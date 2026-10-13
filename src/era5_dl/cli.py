r"""Resumable, parallel ERA5 downloads from Google's public ARCO bucket.

A time range is stored as one Zarr store per calendar month, so a Slurm job array
can download months in parallel. Rerunning the same command resumes interrupted
months and skips completed ones. See README.md for a tutorial.
"""

import argparse
import difflib
import errno
import fcntl
import json
import numpy as np
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zarr

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from numcodecs import Blosc, blosc
from pathlib import Path

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Variables and pressure levels of the Appa archive. Any ARCO selection works.
DEFAULT_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "total_precipitation",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "slope_of_sub_gridscale_orography",
    "standard_deviation_of_orography",
    "land_sea_mask",
]
# fmt: off
DEFAULT_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
ALL_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000,
]
# fmt: on
STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
COORDINATES = ("time", "level", "latitude", "longitude")
COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
EPOCH = np.datetime64("1900-01-01T00", "h")
HOUR = np.timedelta64(1, "h")
LOG_INTERVAL = 300  # seconds between progress lines


class Error(Exception):
    r"""A problem reported to the user without a traceback."""


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", file=sys.stderr, flush=True)


def duration(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def size(count):
    return f"{count / 1e12:.2f} TB" if count >= 1e12 else f"{count / 1e9:.1f} GB"


def parse_time(text, end=False):
    r"""Parse YYYY, YYYY-MM, YYYY-MM-DD or YYYY-MM-DDTHH (UTC).

    An end date includes its whole period: 2025-03 ends at 2025-03-31T23.
    """
    unit = {4: "Y", 7: "M", 10: "D", 13: "h"}.get(len(text))
    try:
        if unit is None or (unit == "h" and text[10] != "T"):
            raise ValueError
        value = np.datetime64(text, unit)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid date {text!r}; use YYYY, YYYY-MM, YYYY-MM-DD or YYYY-MM-DDTHH"
        ) from None
    if end:
        return (value + np.timedelta64(1, unit)).astype("datetime64[h]") - HOUR
    return value.astype("datetime64[h]")


def parse_end(text):
    return parse_time(text, end=True)


def metadata_time(text, end=False):
    r"""Parse a coverage timestamp from source metadata, e.g. "2026-06-30"."""
    text = text.strip().replace(" ", "T")
    if len(text) == 10:
        return parse_time(text, end)
    return np.datetime64(text).astype("datetime64[h]")


def parse_variables(text):
    if text == "default":
        return list(DEFAULT_VARIABLES)
    variables = [item.strip() for item in text.split(",") if item.strip()]
    if not variables or len(set(variables)) != len(variables):
        raise argparse.ArgumentTypeError("expected distinct, comma-separated variable names")
    return variables


def parse_levels(text):
    if text in ("default", "all"):
        return list(DEFAULT_LEVELS if text == "default" else ALL_LEVELS)
    try:
        levels = [int(item) for item in text.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated levels in hPa") from None
    if not levels or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("expected distinct, comma-separated levels in hPa")
    return levels


def decode_times(values, units):
    r"""Decode CF times, e.g. "hours since 1900-01-01 00:00:00", to hourly datetimes."""
    unit, _, origin = units.partition(" since ")
    step = {"days": "D", "hours": "h", "minutes": "m", "seconds": "s"}[unit.strip()]
    origin = np.datetime64(origin.strip().replace(" ", "T"))
    return (origin + values.astype(f"timedelta64[{step}]")).astype("datetime64[h]")


def plan_months(start, end, step):
    r"""Select the timestamps in [start, end] and split them at calendar months.

    Timestamps are aligned to UTC midnight, so a 6-hour step keeps 00, 06, 12 and
    18 UTC, whatever the start hour.
    """
    if start > end:
        raise Error("--start must not be after --end.")
    hours = np.arange(start, end + HOUR, HOUR)
    hours = hours[(hours - EPOCH).astype(np.int64) % step == 0]
    if hours.size == 0:
        raise Error("No timestamps selected; widen the dates or reduce --step-hours.")
    months = hours.astype("datetime64[M]")
    bounds = np.flatnonzero(months[1:] != months[:-1]) + 1
    return [(str(chunk[0].astype("datetime64[M]")), chunk) for chunk in np.split(hours, bounds)]


def array_spec(indices):
    r"""Compress sorted indices for sbatch --array, e.g. [0, 1, 2, 5] -> "0-2,5"."""
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ",".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


def write_json(path, content):
    r"""Atomically replace a JSON file, so an interruption keeps the old version."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w") as handle:
        json.dump(content, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


_warned_locks = threading.Event()


@contextmanager
def file_lock(path, blocking=False):
    r"""Hold an exclusive lock on path, if the filesystem supports file locking."""
    with open(path, "a") as handle:
        locked = True
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise Error(f"Another download holds {path}; is a job still running?") from None
        except OSError as error:
            if error.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOLCK):
                raise
            if not _warned_locks.is_set():
                _warned_locks.set()
                log(
                    "warning: file locking is unsupported on this filesystem; "
                    "never run two downloads of the same month at once"
                )
            locked = False
        try:
            yield
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


class Source:
    r"""Read-only remote ERA5 store with its decoded time axis and coverage."""

    def __init__(self, url, era5t=False):
        log(f"opening {url}")
        options = {"token": "anon"} if url.startswith("gs://") else None
        try:
            self.group = zarr.open_group(
                url, mode="r", zarr_format=2, use_consolidated=True, storage_options=options
            )
            times = self.group["time"]
            self.times = decode_times(times[:], times.attrs["units"])
        except Exception as error:
            raise Error(
                f"Cannot read {url}: {type(error).__name__}: {error}\n"
                "This node needs outbound HTTPS access to storage.googleapis.com."
            ) from error
        self.url = url
        self.era5t = era5t
        self.arrays = {}
        attrs = dict(self.group.attrs)
        self.first, self.last = self.times[0], self.times[-1]
        if "valid_time_start" in attrs:
            self.first = max(self.first, metadata_time(attrs["valid_time_start"]))
        self.final = self.last
        if "valid_time_stop" in attrs:
            self.final = min(self.last, metadata_time(attrs["valid_time_stop"], end=True))
        self.preliminary = self.last
        if "valid_time_stop_era5t" in attrs:
            self.preliminary = min(
                self.last, metadata_time(attrs["valid_time_stop_era5t"], end=True)
            )
        self.last = self.preliminary if era5t else self.final
        self.updated = attrs.get("last_updated")

    def array(self, name):
        if name not in self.arrays:
            self.arrays[name] = self.group[name]
        return self.arrays[name]

    def dims(self, name):
        return list(self.array(name).attrs["_ARRAY_DIMENSIONS"])

    def check(self, variables, levels, times):
        r"""Validate a selection before reading any field."""
        names = set(self.group.array_keys())
        for var in variables:
            if var not in names or var in COORDINATES:
                close = difflib.get_close_matches(var, sorted(names - set(COORDINATES)), n=3)
                hint = f" Did you mean {', '.join(close)}?" if close else ""
                raise Error(f"Unknown variable {var!r}.{hint} List them with `variables`.")
            if self.dims(var)[0] != "time":
                raise Error(f"{var} has no time dimension; only time series are supported.")
        if any("level" in self.dims(var) for var in variables):
            available = self.array("level")[:].tolist()
            unknown = [level for level in levels if level not in available]
            if unknown:
                raise Error(f"Unknown levels {unknown}; available levels: {available}")
        if times[0] < self.first or times[-1] > self.last:
            hint = ""
            if not self.era5t and times[-1] <= self.preliminary:
                hint = (
                    f" Preliminary ERA5T data reaches {self.preliminary}; "
                    "add --era5t to include it."
                )
            raise Error(
                f"Requested {times[0]} to {times[-1]}, but the source covers "
                f"{self.first} to {self.last}.{hint}"
            )
        self.indices(times)

    def indices(self, times):
        index = np.searchsorted(self.times, times)
        valid = index < self.times.size
        if not valid.all() or not np.array_equal(self.times[index], times):
            raise Error("The source has missing or misaligned timestamps in this range.")
        return index

    def level_index(self, levels):
        available = self.array("level")[:].tolist()
        index = [available.index(level) for level in levels]
        return slice(None) if index == list(range(len(available))) else np.asarray(index)

    def frame_bytes(self, var, levels):
        r"""Uncompressed bytes of one timestamp of var after level selection."""
        array, size = self.array(var), self.array(var).dtype.itemsize
        for dim, length in zip(self.dims(var)[1:], array.shape[1:], strict=True):
            size *= len(levels) if dim == "level" else length
        return size

    def read(self, var, time_index, level_index):
        if np.all(np.diff(time_index) == 1):
            time_index = slice(int(time_index[0]), int(time_index[-1]) + 1)
        selection = tuple(
            time_index if dim == "time" else level_index if dim == "level" else slice(None)
            for dim in self.dims(var)
        )
        return self.array(var).get_orthogonal_selection(selection)


def selection(args):
    r"""Archive-wide settings; every month in an output directory must share them."""
    return {
        "format": 1,
        "source": args.source,
        "variables": args.variables,
        "levels": args.levels,
        "step_hours": args.step_hours,
    }


def month_request(schema, times):
    return {
        "selection": schema,
        "start": str(times[0]),
        "end": str(times[-1]),
        "timestamps": int(times.size),
    }


def check_root(root, schema):
    r"""Reject an archive directory created with another selection."""
    manifest = root / "archive.json"
    if manifest.exists() and json.loads(manifest.read_text()) != schema:
        raise Error(
            f"{root} holds a different selection:\n{manifest.read_text()}"
            "Use another --output for other variables, levels, time steps or sources."
        )


def prepare_root(root, schema):
    r"""Create the archive directory and its manifest, once the selection is valid."""
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".archive.lock", blocking=True):
        check_root(root, schema)
        if not (root / "archive.json").exists():
            write_json(root / "archive.json", schema)


def published(root, label, request):
    r"""Whether the month is complete. A different completed request is an error."""
    final = root / f"{label}.zarr"
    if not final.exists():
        return False
    stored = dict(zarr.open_group(final, mode="r", zarr_format=2).attrs).get("era5_download")
    if stored != request:
        raise Error(
            f"{final} already holds {stored['start'] if stored else '?'} to "
            f"{stored['end'] if stored else '?'}, which differs from this request. Months are "
            "immutable: use whole calendar months, or another --output."
        )
    return True


def fraction_done(root, label):
    r"""Fraction of the uncompressed bytes of an interrupted month already saved."""
    staging, progress = root / f".{label}.zarr.partial", root / f".{label}.progress"
    try:
        request = json.loads((progress / "request.json").read_text())
        group = zarr.open_group(staging, mode="r", zarr_format=2, use_consolidated=False)
        done = total = 0
        for var in request["selection"]["variables"]:
            array, path = group[var], progress / f"{var}.json"
            bits = json.loads(path.read_text())["done"] if path.exists() else ""
            done += bits.count("1") * array.nbytes / array.shape[0]
            total += array.nbytes
        return done / total
    except (OSError, KeyError, ValueError, zarr.errors.BaseZarrError):
        return 0.0


def month_state(root, label, request):
    if (root / f"{label}.zarr").exists():
        try:
            published(root, label, request)
        except Error:
            return "conflict: holds other dates"
        return "complete"
    marker = root / f".{label}.progress" / "request.json"
    if marker.exists():
        if json.loads(marker.read_text()) != request:
            return "conflict: started with other dates"
        return f"partial ({fraction_done(root, label):.0%})"
    return "to do"


def create_store(path, source, variables, levels, times, request):
    r"""Create an empty monthly store whose chunks each hold one timestamp."""
    group = zarr.open_group(path, mode="w", zarr_format=2)

    def coordinate(name, values, attrs):
        array = group.create_array(
            name,
            shape=values.shape,
            chunks=values.shape,
            dtype=values.dtype,
            fill_value=None,
            compressors=COMPRESSOR,
            attributes=attrs,
        )
        array[...] = values

    coordinate(
        "time",
        (times - EPOCH).astype(np.int64),
        {
            "_ARRAY_DIMENSIONS": ["time"],
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        },
    )
    dims = {dim for var in variables for dim in source.dims(var)[1:]}
    for name in sorted(dims):
        array = source.array(name)
        values = np.asarray(levels, dtype=array.dtype) if name == "level" else array[:]
        coordinate(name, values, dict(array.attrs))
    for var in variables:
        array = source.array(var)
        shape = tuple(
            times.size if dim == "time" else len(levels) if dim == "level" else length
            for dim, length in zip(source.dims(var), array.shape, strict=True)
        )
        group.create_array(
            var,
            shape=shape,
            chunks=(1, *shape[1:]),
            dtype=array.dtype,
            fill_value=array.fill_value,
            compressors=COMPRESSOR,
            attributes=dict(array.attrs),
        )
    group.attrs.update({
        "era5_download": request,
        "era5_source_last_updated": source.updated,
        "era5_preliminary": bool(times[-1] > source.final),
    })
    return group


def download_month(root, label, times, schema, source, args):
    r"""Download one month, resuming from its checkpoints, then publish it."""
    request = month_request(schema, times)
    final = root / f"{label}.zarr"
    staging = root / f".{label}.zarr.partial"
    progress = root / f".{label}.progress"
    with file_lock(root / f".{label}.lock"):
        if published(root, label, request):
            log(f"{label}: already complete, skipping")
            # A job killed right after publishing may leave its checkpoints behind.
            shutil.rmtree(progress, ignore_errors=True)
            return
        marker = progress / "request.json"
        if marker.exists():
            if json.loads(marker.read_text()) != request:
                raise Error(
                    f"{label} was started with other dates. Rerun it with the original "
                    f"--start/--end, or delete {staging} and {progress} to restart it."
                )
            if not staging.exists():
                raise Error(f"{staging} is missing; delete {progress} to restart {label}.")
            group = zarr.open_group(staging, mode="r+", zarr_format=2, use_consolidated=False)
            log(f"{label}: resuming ({fraction_done(root, label):.0%} done)")
        else:
            # Without the marker nothing was committed: (re)initialize from scratch.
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(progress, ignore_errors=True)
            group = create_store(staging, source, args.variables, args.levels, times, request)
            progress.mkdir()
            write_json(marker, request)
            log(f"{label}: started, {times.size} timestamps")
        started = time.monotonic()
        try:
            fetch_month(label, group, progress, source, times, args)
        except BaseException:
            log(f"{label}: interrupted; rerun the same command to resume")
            raise
        group.attrs["era5_complete"] = True
        zarr.consolidate_metadata(staging, zarr_format=2)
        stored = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        (root / "reports").mkdir(exist_ok=True)
        write_json(
            root / "reports" / f"{label}.json",
            {
                "month": label,
                "start": request["start"],
                "end": request["end"],
                "timestamps": int(times.size),
                "workers": args.workers,
                "batch": args.batch,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "stored_bytes": stored,
                "finished": datetime.now().isoformat(timespec="seconds"),
            },
        )
        staging.rename(final)
        shutil.rmtree(progress)
        log(f"{label}: complete, {size(stored)} in {final}")


def fetch_month(label, group, progress, source, times, args):
    r"""Fetch every missing (variable, time batch) with a pool of workers.

    Each variable has a bitmap of saved timestamps. A batch is marked only after
    its chunks are written, so a killed job rewrites at most the batches in flight.
    """
    variables = sorted(args.variables, key=lambda var: -source.frame_bytes(var, args.levels))
    arrays = {var: group[var] for var in variables}
    index = source.indices(times)
    level_index = None
    if any("level" in source.dims(var) for var in variables):
        level_index = source.level_index(args.levels)
    done, locks, frame = {}, {}, {}
    for var in variables:
        path = progress / f"{var}.json"
        bits = json.loads(path.read_text())["done"] if path.exists() else "0" * times.size
        if len(bits) != times.size:
            raise Error(f"Corrupted checkpoint {path}; delete it to refetch {var}.")
        done[var] = bytearray(bits.encode())
        locks[var] = threading.Lock()
        frame[var] = source.frame_bytes(var, args.levels)
    tasks = []
    for var in variables:
        for start in range(0, times.size, args.batch):
            stop = min(start + args.batch, times.size)
            if b"0" in done[var][start:stop]:
                tasks.append((var, start, stop))
    total = sum(frame[var] * times.size for var in variables)
    saved = sum(frame[var] * done[var].count(b"1") for var in variables)
    initial = saved
    counter = threading.Lock()
    cancel = threading.Event()

    def work(var, start, stop):
        nonlocal saved
        for attempt in range(args.retries):
            if cancel.is_set():
                return
            try:
                values = source.read(var, index[start:stop], level_index)
                axes = tuple(range(1, values.ndim))
                if not np.isfinite(values).any(axis=axes).all():
                    raise ValueError("the source returned an entirely missing frame")
                arrays[var][start:stop] = values
                break
            except Exception as error:
                if attempt + 1 == args.retries:
                    raise RuntimeError(f"{label}/{var}[{start}:{stop}] failed") from error
                delay = min(10 * 2**attempt, 300)
                log(
                    f"{label}/{var}[{start}:{stop}]: {type(error).__name__}: {error}; "
                    f"retry {attempt + 1}/{args.retries - 1} in {delay}s"
                )
                if cancel.wait(delay):
                    return
        with locks[var]:
            done[var][start:stop] = b"1" * (stop - start)
            write_json(progress / f"{var}.json", {"done": done[var].decode()})
            finished = b"0" not in done[var]
        with counter:
            saved += frame[var] * (stop - start)
        if finished:
            log(f"{label}/{var}: done")

    started = last = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        pending = {pool.submit(work, *task) for task in tasks}
        try:
            while pending:
                finished, pending = wait(pending, timeout=60, return_when=FIRST_EXCEPTION)
                for future in finished:
                    future.result()
                now = time.monotonic()
                if pending and now - last >= LOG_INTERVAL:
                    last = now
                    rate = (saved - initial) / (now - started)
                    eta = duration((total - saved) / rate) if rate > 0 else "?"
                    log(
                        f"{label}: {saved / 1e9:.1f}/{total / 1e9:.1f} GB "
                        f"({saved / total:.0%}), {rate / 1e6:.0f} MB/s, ETA {eta}"
                    )
        except BaseException:
            cancel.set()
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def command_plan(args, months, schema, root):
    source = Source(args.source, args.era5t)
    source.check(args.variables, args.levels, np.concatenate([times for _, times in months]))
    print(f"source     {source.url}")
    print(f"           ERA5 until {source.final}, preliminary ERA5T until {source.preliminary}")
    print(
        f"selection  {len(args.variables)} variables, {len(args.levels)} levels, "
        f"every {args.step_hours} h"
    )
    frame = sum(source.frame_bytes(var, args.levels) for var in args.variables)
    print(f"\n{'month':8} {'timestamps':>10} {'size':>9}  status")
    states = []
    for label, times in months:
        state = month_state(root, label, month_request(schema, times))
        states.append(state)
        print(f"{label:8} {times.size:>10} {size(frame * times.size):>9}  {state}")
    count = sum(times.size for _, times in months)
    todo = sum(state != "complete" for state in states)
    print(
        f"\n{len(months)} months ({todo} not complete), {count} timestamps, "
        f"{size(frame * count)} uncompressed"
    )


def command_download(args, months, schema, root):
    if args.month_index is not None:
        if not 0 <= args.month_index < len(months):
            raise Error(f"--month-index must be in [0, {len(months) - 1}].")
        months = [months[args.month_index]]
    todo = []
    for label, times in months:
        if published(root, label, month_request(schema, times)):
            log(f"{label}: already complete, skipping")
        else:
            todo.append((label, times))
    if not todo:
        return
    source = Source(args.source, args.era5t)
    source.check(args.variables, args.levels, np.concatenate([times for _, times in todo]))
    prepare_root(root, schema)
    # Workers already parallelize; Blosc threads would oversubscribe the CPUs.
    blosc.set_nthreads(1)
    if args.parallel_months == 1:
        for label, times in todo:
            download_month(root, label, times, schema, source, args)
        return
    with ThreadPoolExecutor(args.parallel_months) as pool:
        futures = {
            pool.submit(download_month, root, label, times, schema, source, args): label
            for label, times in todo
        }
        failed = []
        for future, label in futures.items():
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - other months keep going
                log(f"{label}: failed: {error}")
                failed.append(label)
    if failed:
        raise Error(f"{len(failed)} months failed ({', '.join(failed)}); rerun to resume.")


def command_submit(args, months, schema, root, sbatch_args):
    pending = [
        i
        for i, (label, times) in enumerate(months)
        if not published(root, label, month_request(schema, times))
    ]
    if not pending:
        log(f"all {len(months)} months are complete")
        return
    source = Source(args.source, args.era5t)
    source.check(args.variables, args.levels, np.concatenate([months[i][1] for i in pending]))
    forwarded = [
        sys.executable, "-u", "-m", "era5_dl", "download",
        "--output", str(root),
        "--start", str(args.start),
        "--end", str(args.end),
        "--variables", ",".join(args.variables),
        "--levels", ",".join(map(str, args.levels)),
        "--step-hours", str(args.step_hours),
        "--source", args.source,
        "--workers", str(args.workers),
        "--batch", str(args.batch),
        "--retries", str(args.retries),
    ]  # fmt: skip
    if args.era5t:
        forwarded.append("--era5t")
    command = [
        "sbatch", "--parsable",
        f"--job-name={args.job_name}",
        f"--array={array_spec(pending)}%{args.max_parallel}",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        "--export=ALL",
        f"--output={root}/logs/%x_%A_%a.log",
        *sbatch_args,
        f"--wrap={shlex.join(forwarded)} --month-index $SLURM_ARRAY_TASK_ID",
    ]  # fmt: skip
    if args.dry_run:
        print(shlex.join(command))
        return
    prepare_root(root, schema)
    (root / "logs").mkdir(exist_ok=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise Error(f"sbatch failed:\n{result.stderr.strip()}\nCommand: {shlex.join(command)}")
    job = result.stdout.strip().split(";")[0]
    log(f"submitted job array {job}: {len(pending)} months, at most {args.max_parallel} at once")
    print(f"  queue:    squeue -j {job}")
    print(
        f"  progress: {shlex.join([sys.executable, '-m', 'era5_dl', 'status', '-o', str(root)])}"
    )
    print(f"  logs:     {root}/logs/{args.job_name}_{job}_<task>.log")
    print(f"  cancel:   scancel {job}  (rerun this command later to resume)")


def command_status(args):
    root = Path(args.output).resolve()
    manifest = root / "archive.json"
    if not manifest.exists():
        raise Error(f"No archive in {root}.")
    schema = json.loads(manifest.read_text())
    print(f"archive    {root}")
    print(f"source     {schema['source']}")
    print(
        f"selection  {len(schema['variables'])} variables, {len(schema['levels'])} levels, "
        f"every {schema['step_hours']} h"
    )
    labels = sorted(
        {path.name[: -len(".zarr")] for path in root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].zarr")}
        | {
            path.name[1 : -len(".progress")]
            for path in root.glob(".[0-9][0-9][0-9][0-9]-[0-9][0-9].progress")
        }
    )
    if not labels:
        print("\nNo months started yet.")
        return
    print(f"\n{'month':8} {'timestamps':>10} {'stored':>9}  status")
    stored = 0
    for label in labels:
        report = root / "reports" / f"{label}.json"
        if (root / f"{label}.zarr").exists() and report.exists():
            report = json.loads(report.read_text())
            stored += report["stored_bytes"]
            print(
                f"{label:8} {report['timestamps']:>10} {size(report['stored_bytes']):>9}  "
                f"complete in {duration(report['elapsed_seconds'])}"
            )
            continue
        if (root / f"{label}.zarr").exists():
            print(f"{label:8} {'':>10} {'':>9}  complete")
            continue
        try:
            with file_lock(root / f".{label}.lock"):
                activity = "interrupted"
        except Error:
            activity = "running"
        print(f"{label:8} {'':>10} {'':>9}  {activity} ({fraction_done(root, label):.0%})")
    complete = sum((root / f"{label}.zarr").exists() for label in labels)
    print(f"\n{complete}/{len(labels)} months complete, {size(stored)} stored")


def command_variables(args):
    source = Source(args.source)
    print(f"\n{'name':48} {'kind':12} {'units':12} description")
    for name in sorted(source.group.array_keys()):
        if name in COORDINATES:
            continue
        attrs = source.array(name).attrs
        kind = "pressure" if "level" in source.dims(name) else "single"
        print(f"{name:48} {kind:12} {attrs.get('units', ''):12} {attrs.get('long_name', '')}")


def parser():
    selection_parser = argparse.ArgumentParser(add_help=False)
    group = selection_parser.add_argument_group("selection")
    group.add_argument("-o", "--output", required=True, help="archive directory")
    group.add_argument(
        "--start", required=True, type=parse_time, help="YYYY, YYYY-MM, YYYY-MM-DD or ...THH"
    )
    group.add_argument(
        "--end", required=True, type=parse_end, help="inclusive: 2025 ends 2025-12-31T23"
    )
    group.add_argument(
        "--variables",
        default="default",
        type=parse_variables,
        help="comma-separated names (see the variables command), or 'default'",
    )
    group.add_argument(
        "--levels",
        default="default",
        type=parse_levels,
        help="comma-separated hPa, 'default' (13 levels) or 'all' (37)",
    )
    group.add_argument("--step-hours", default=1, type=int, choices=STEPS, help="time step")
    group.add_argument(
        "--era5t", action="store_true", help="allow preliminary data after final ERA5"
    )
    group.add_argument("--source", default=ARCO_URL, help=argparse.SUPPRESS)

    work_parser = argparse.ArgumentParser(add_help=False)
    group = work_parser.add_argument_group("performance")
    group.add_argument(
        "--workers", default=8, type=int, help="concurrent fetches per month (default 8)"
    )
    group.add_argument(
        "--batch", default=2, type=int, help="timestamps per fetch and checkpoint (default 2)"
    )
    group.add_argument(
        "--retries", default=5, type=int, help="attempts per batch before failing (default 5)"
    )

    main_parser = argparse.ArgumentParser(
        prog="era5-dl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = main_parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "plan",
        parents=[selection_parser],
        help="validate a selection and show sizes and month status",
    )
    download = commands.add_parser(
        "download",
        parents=[selection_parser, work_parser],
        help="download on this machine (or inside your own job)",
    )
    download.add_argument(
        "--parallel-months", default=1, type=int, help="months downloaded at once (default 1)"
    )
    download.add_argument("--month-index", type=int, help=argparse.SUPPRESS)
    submit = commands.add_parser(
        "submit",
        parents=[selection_parser, work_parser],
        help="submit a Slurm job array, one task per incomplete month",
        epilog="Arguments after -- go to sbatch, e.g. -- --account=X --partition=Y",
    )
    submit.add_argument("--max-parallel", default=12, type=int, help="months running at once")
    submit.add_argument("--cpus", default=8, type=int, help="CPUs per task (default 8)")
    submit.add_argument("--mem", default="16G", help="memory per task (default 16G)")
    submit.add_argument("--time", default="2-00:00:00", help="time limit per task")
    submit.add_argument("--job-name", default="era5")
    submit.add_argument("--dry-run", action="store_true", help="print the sbatch command only")
    status = commands.add_parser("status", help="show completed and in-progress months")
    status.add_argument("-o", "--output", required=True, help="archive directory")
    variables = commands.add_parser("variables", help="list the variables of the source")
    variables.add_argument("--source", default=ARCO_URL, help=argparse.SUPPRESS)
    return main_parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    sbatch_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, sbatch_args = argv[:split], argv[split + 1 :]
    main_parser = parser()
    args = main_parser.parse_args(argv)
    if sbatch_args and args.command != "submit":
        main_parser.error("arguments after -- are only accepted by submit")
    try:
        if args.command == "status":
            return command_status(args)
        if args.command == "variables":
            return command_variables(args)
        if args.command != "plan":
            for name in ("workers", "batch", "retries", "parallel_months", "max_parallel"):
                if getattr(args, name, 1) < 1:
                    raise Error(f"--{name.replace('_', '-')} must be positive.")
        months = plan_months(args.start, args.end, args.step_hours)
        schema = selection(args)
        root = Path(args.output).resolve()
        check_root(root, schema)
        if args.command == "plan":
            return command_plan(args, months, schema, root)
        if args.command == "download":
            return command_download(args, months, schema, root)
        return command_submit(args, months, schema, root, sbatch_args)
    except Error as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
