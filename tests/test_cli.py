r"""Offline tests of era5-dl against a small ARCO-like store.

Run with `uv run pytest`.
"""

import json
import numpy as np
import os
import pytest
import shlex
import subprocess
import zarr

from era5_dl import cli as tool

LEVELS = [50, 500, 850, 1000]
VARIABLES = ["temperature", "2m_temperature", "sea_surface_temperature"]
START = np.datetime64("2025-01-30T00", "h")
HOURS = 24 * 5  # until 2025-02-03T23; the time axis runs past the published data


def field(var, times, levels=None):
    r"""Deterministic values of the fake source."""
    offset = VARIABLES.index(var) * 1e5
    t = (times - START).astype(np.float32)[:, None, None]
    lat, lon = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    base = offset + 10 * t + lat + 0.1 * lon
    if var == "sea_surface_temperature":
        base = np.where(lat == 0, np.nan, base)
    if levels is None:
        return base.astype(np.float32)
    return np.stack([base + level for level in levels], axis=1).astype(np.float32)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=2)
    times = START + np.arange(HOURS) * tool.HOUR

    def add(name, values, dims, chunks=None, **attrs):
        array = group.create_array(
            name,
            shape=values.shape,
            chunks=chunks or values.shape,
            dtype=values.dtype,
            fill_value=np.nan if values.dtype.kind == "f" else None,
            attributes={"_ARRAY_DIMENSIONS": dims, **attrs},
        )
        array[...] = values

    add(
        "time",
        (times - tool.EPOCH).astype(np.int64),
        ["time"],
        chunks=(50,),
        units="hours since 1900-01-01 00:00:00",
        calendar="proleptic_gregorian",
    )
    add("level", np.array([1, *LEVELS], dtype=np.int64), ["level"], units="hPa")
    add("latitude", np.array([90, 0, -90], dtype=np.float32), ["latitude"])
    add("longitude", np.array([0, 90, 180, 270], dtype=np.float32), ["longitude"])
    add(
        "temperature",
        field("temperature", times, [1, *LEVELS]),
        ["time", "level", "latitude", "longitude"],
        chunks=(1, 5, 3, 4),
        units="K",
        long_name="Temperature",
    )
    for var in VARIABLES[1:]:
        add(var, field(var, times), ["time", "latitude", "longitude"], chunks=(1, 3, 4), units="K")
    group.attrs.update({
        "valid_time_start": "1940-01-01",
        "valid_time_stop": "2025-01-31",
        "valid_time_stop_era5t": "2025-02-02",
        "last_updated": "2025-02-03 03:00:00+00:00",
    })
    zarr.consolidate_metadata(path, zarr_format=2)
    return str(path)


def run(source, output, *extra, start="2025-01-31T20", end="2025-02-01T03"):
    return tool.main([
        "download",
        "--output",
        str(output),
        "--start",
        start,
        "--end",
        end,
        "--variables",
        ",".join(VARIABLES),
        "--levels",
        "500,1000",
        "--source",
        source,
        "--era5t",
        "--workers",
        "3",
        "--batch",
        "3",
        *extra,
    ])


def check_month(path, start, count):
    times = np.datetime64(start, "h") + np.arange(count) * tool.HOUR
    group = zarr.open_group(path, mode="r", zarr_format=2)
    assert group.attrs["era5_complete"] is True
    np.testing.assert_array_equal(group["level"][:], [500, 1000])
    np.testing.assert_array_equal(
        tool.decode_times(group["time"][:], group["time"].attrs["units"]), times
    )
    np.testing.assert_array_equal(
        group["temperature"][:], field("temperature", times, [500, 1000])
    )
    for var in VARIABLES[1:]:
        np.testing.assert_array_equal(group[var][:], field(var, times))
    return group


def test_dates_and_months():
    assert tool.parse_time("2025") == np.datetime64("2025-01-01T00")
    assert tool.parse_end("2024-02") == np.datetime64("2024-02-29T23")
    assert tool.parse_end("2025-02-01T05") == np.datetime64("2025-02-01T05")
    for text in ("2025-1", "2025-01-01 05", "2025-13", "tomorrow"):
        with pytest.raises(Exception, match="invalid date"):
            tool.parse_time(text)
    months = tool.plan_months(
        tool.parse_time("2025-01-31T09"), tool.parse_end("2025-02-01T06"), step=6
    )
    assert [label for label, _ in months] == ["2025-01", "2025-02"]
    assert [str(t) for t in months[0][1]] == ["2025-01-31T12", "2025-01-31T18"]
    assert [str(t) for t in months[1][1]] == ["2025-02-01T00", "2025-02-01T06"]
    assert tool.array_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_download_matches_source(source, tmp_path):
    output = tmp_path / "archive"
    assert run(source, output) is None
    january = check_month(output / "2025-01.zarr", "2025-01-31T20", 4)
    february = check_month(output / "2025-02.zarr", "2025-02-01T00", 4)
    assert january.attrs["era5_preliminary"] is False
    assert february.attrs["era5_preliminary"] is True
    assert json.loads((output / "archive.json").read_text())["variables"] == VARIABLES
    assert sorted(path.name for path in (output / "reports").iterdir()) == [
        "2025-01.json",
        "2025-02.json",
    ]
    assert not list(output.glob(".*.partial")) and not list(output.glob(".*.progress"))
    # One timestamp per chunk, compressed like the source.
    assert february["temperature"].chunks == (1, 2, 3, 4)
    xr = pytest.importorskip("xarray")
    data = xr.open_zarr(output / "2025-01.zarr", chunks=None)
    assert data.temperature.dims == ("time", "level", "latitude", "longitude")
    assert str(data.time.values[0])[:13] == "2025-01-31T20"


def test_coverage_and_selection_errors(source, tmp_path, capsys):
    output = tmp_path / "archive"
    # Without --era5t the published (final) data ends on 2025-01-31.
    argv = ["download", "-o", str(output), "--start", "2025-02-01", "--end", "2025-02-01"]
    assert tool.main([*argv, "--source", source, "--variables", "2m_temperature"]) == 1
    assert "add --era5t" in capsys.readouterr().err
    assert not output.exists()  # a rejected request leaves nothing behind
    assert tool.main([*argv, "--source", source, "--variables", "2m_temprature", "--era5t"]) == 1
    assert "Did you mean 2m_temperature" in capsys.readouterr().err
    levels = ["--variables", "temperature", "--levels", "1,2", "--era5t"]
    assert tool.main([*argv, "--source", source, *levels]) == 1
    assert "Unknown levels [2]" in capsys.readouterr().err
    late = [*argv[:-1], "2025-02-04", "--source", source, "--variables", "2m_temperature"]
    assert tool.main([*late, "--era5t"]) == 1
    assert "but the source covers" in capsys.readouterr().err


def test_resume_after_crash(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "archive"
    read, calls = tool.Source.read, []

    def flaky(self, var, time_index, level_index):
        calls.append(var)
        if len(calls) > 3:
            raise ConnectionError("network down")
        return read(self, var, time_index, level_index)

    monkeypatch.setattr(tool.Source, "read", flaky)
    with pytest.raises(RuntimeError, match="failed"):
        run(source, output, "--workers", "1", "--batch", "2", "--retries", "1")
    progress = output / ".2025-01.progress"
    assert (output / ".2025-01.zarr.partial").exists()
    assert json.loads((progress / "temperature.json").read_text())["done"] == "1111"
    assert json.loads((progress / "2m_temperature.json").read_text())["done"] == "1100"
    assert tool.main(["status", "-o", str(output)]) is None
    assert "interrupted (" in capsys.readouterr().out

    calls.clear()
    monkeypatch.setattr(tool.Source, "read", lambda *args: (calls.append(args[1]), read(*args))[1])
    assert run(source, output, "--workers", "1", "--batch", "2") is None
    # January resumes with 1 + 2 missing batches; February needs all 6.
    assert len(calls) == 3 + 6
    check_month(output / "2025-01.zarr", "2025-01-31T20", 4)
    check_month(output / "2025-02.zarr", "2025-02-01T00", 4)
    assert tool.main(["status", "-o", str(output)]) is None
    assert "2/2 months complete" in capsys.readouterr().out


def test_complete_months_are_skipped_offline(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "archive"
    assert run(source, output) is None

    def unreachable(*args, **kwargs):
        raise AssertionError("the source should not be opened")

    monkeypatch.setattr(tool.Source, "__init__", unreachable)
    assert run(source, output) is None
    assert capsys.readouterr().err.count("already complete") == 2


def test_archive_and_months_are_immutable(source, tmp_path, capsys):
    output = tmp_path / "archive"
    assert run(source, output, end="2025-01-31T23") is None
    # Other variables need another output directory.
    argv = ["download", "-o", str(output), "--start", "2025-01-31", "--end", "2025-01-31"]
    assert tool.main([*argv, "--source", source, "--variables", "2m_temperature"]) == 1
    assert "holds a different selection" in capsys.readouterr().err
    # A completed partial month cannot be enlarged.
    assert run(source, output, start="2025-01-31T18", end="2025-01-31T23") == 1
    assert "Months are immutable" in capsys.readouterr().err
    # Extending with new months is fine.
    assert run(source, output, start="2025-02-01T00", end="2025-02-01T03") is None
    check_month(output / "2025-02.zarr", "2025-02-01T00", 4)


def test_submit(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "archive"
    assert run(source, output, end="2025-01-31T23") is None
    capsys.readouterr()
    argv = [
        "submit", "-o", str(output), "--start", "2025-01-31T20", "--end", "2025-02-01T03",
        "--variables", ",".join(VARIABLES), "--levels", "500,1000", "--source", source,
        "--era5t", "--cpus", "2", "--mem", "4G", "--time", "00:10:00",
    ]  # fmt: skip
    assert tool.main([*argv, "--dry-run", "--", "--account=me", "--qos=low"]) is None
    command = shlex.split(capsys.readouterr().out)
    assert command[:2] == ["sbatch", "--parsable"]
    assert "--array=1%12" in command and "--account=me" in command and "--qos=low" in command
    assert "--cpus-per-task=2" in command and "--mem=4G" in command

    submitted = []
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda command, **kwargs: (
            submitted.append(command),
            subprocess.CompletedProcess(command, 0, "4242;cluster\n", ""),
        )[1],
    )
    assert tool.main(argv) is None
    assert "submitted job array 4242" in capsys.readouterr().err
    assert (output / "logs").is_dir()
    # The wrapped command runs on its own, as a Slurm array task would.
    wrap = submitted[0][-1].removeprefix("--wrap=")
    environment = {**os.environ, "SLURM_ARRAY_TASK_ID": "1"}
    monkeypatch.undo()
    subprocess.run(["sh", "-c", wrap], env=environment, check=True)
    check_month(output / "2025-02.zarr", "2025-02-01T00", 4)
    assert tool.main(argv) is None
    assert "all 2 months are complete" in capsys.readouterr().err


def test_plan(source, tmp_path, capsys):
    output = tmp_path / "archive"
    assert run(source, output, end="2025-01-31T23") is None
    capsys.readouterr()
    argv = ["plan", "-o", str(output), "--start", "2025-01-31T20", "--end", "2025-02-01T03"]
    extra = ["--variables", ",".join(VARIABLES), "--levels", "500,1000", "--source", source]
    assert tool.main([*argv, *extra, "--era5t"]) is None
    out = capsys.readouterr().out
    assert "2025-01" in out and "complete" in out and "to do" in out
    assert "2 months (1 not complete), 8 timestamps" in out
