# era5-dl

A small command-line tool that downloads hourly 0.25° ERA5 from Google's public
[ARCO bucket](https://github.com/google-research/arco-era5) onto a cluster filesystem.

- **One Zarr store per calendar month.** On Slurm, each month runs as one task of a job array.
- **Resumable.** Rerun the same command: finished months are skipped, and interrupted
  months continue from their last checkpoint. This holds even after `kill -9` or a time limit.
- **Parallel.** Several concurrent fetches per month, and several months at once.
- **Extendable.** Add earlier or later years to the same directory later.
- **Minimal dependencies.** Only `zarr>=3`, `gcsfs` and `numpy`, locked with [uv](https://docs.astral.sh/uv/).
  No Google credentials needed.

## 1. Install

Install uv if needed (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Then clone this repo
somewhere the compute nodes can see, such as your home directory:

```bash
git clone git@github.com:iSach/era5-dl.git && cd era5-dl
uv sync                          # creates .venv from uv.lock, fetching Python 3.12 if needed
uv run era5-dl variables | head  # checks that the bucket is reachable
```

All commands below are run from this directory.

## 2. Try a few hours locally

```bash
uv run era5-dl download -o /scratch/$USER/era5-test --start 2025-01-01T00 --end 2025-01-01T03
```

This writes about 1 GB in under a minute.

## 3. Plan, then submit to Slurm

```bash
uv run era5-dl plan   -o /scratch/$USER/era5 --start 2025 --end 2025
uv run era5-dl submit -o /scratch/$USER/era5 --start 2025 --end 2025 -- --account=ACCOUNT --partition=PARTITION
```

`plan` checks the selection and prints the size and status of each month without writing anything.
`submit` sends one array task per incomplete month. Everything after `--` goes to `sbatch` unchanged.

- **Defaults per task:** `--cpus 8 --mem 16G --time 2-00:00:00`.
- **Concurrency:** at most `--max-parallel 12` months run at once.
- **Preview:** `--dry-run` prints the `sbatch` command without submitting it.
- **Environment:** jobs run `.venv/bin/python -m era5_dl` from this repo, so keep it on a shared filesystem.
- **Network:** compute nodes need outbound HTTPS to `storage.googleapis.com`.

## 4. Monitor

```bash
uv run era5-dl status -o /scratch/$USER/era5   # complete / running / interrupted, with %
squeue -j JOBID
tail -f /scratch/$USER/era5/logs/era5_JOBID_*.log        # progress, MB/s and ETA every 5 min
```

## 5. Resume and extend

- **Resume** after a time limit, `scancel`, node failure or network outage by rerunning the same `submit` command.
  Failed network reads are retried 5 times before the task gives up.
  Don't submit twice while tasks are still running: a lock makes the second run of a month fail immediately.
- **Extend** by using the same `-o` with new dates, e.g. `--start 2024 --end 2024`.
- **Keep the selection fixed:** variables, levels and time step must stay the same within one directory
  (they are recorded in `archive.json`). Use another directory for a different selection.
- **Finished months are never modified**, so use whole calendar months. To redownload a month, delete `YYYY-MM.zarr`.

## 6. What to download

| Option | Meaning |
|---|---|
| `--start`, `--end` | Inclusive UTC dates: `YYYY`, `YYYY-MM`, `YYYY-MM-DD` or `YYYY-MM-DDTHH` |
| `--variables` | Comma-separated ARCO names (see `uv run era5-dl variables`). Default: the 16 below |
| `--levels` | Comma-separated hPa, `all` (37), or the default 13: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |
| `--step-hours` | 1 (default), 2, 3, 4, 6, 8, 12 or 24, aligned to 00 UTC |
| `--era5t` | Also allow the preliminary ERA5T data that follows the final ERA5 (about 3 months behind real time) |
| `--workers`, `--batch` | Concurrent fetches per month (default 8), and timestamps per fetch and checkpoint (default 2) |

The default variables are:

- **Single level:** `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`,
  `mean_sea_level_pressure`, `sea_surface_temperature`, `total_precipitation`
- **Pressure level:** `temperature`, `u_component_of_wind`, `v_component_of_wind`, `geopotential`,
  `specific_humidity`
- **Static fields, stored hourly like the others:** `angle_of_sub_gridscale_orography`,
  `anisotropy_of_sub_gridscale_orography`, `slope_of_sub_gridscale_orography`,
  `standard_deviation_of_orography`, `land_sea_mask`

Each pressure-level chunk in the bucket holds all 37 levels. Choosing fewer levels saves
disk space but not download time.

## 7. Read the data

The downloader doesn't need xarray, but reading is easiest with it. For example, run
`uv run --with xarray --with dask python`, then:

```python
import glob, xarray as xr

month = xr.open_zarr("/scratch/era5/2025-01.zarr")
everything = xr.open_mfdataset(
    sorted(glob.glob("/scratch/era5/[0-9]*.zarr")), engine="zarr"
)  # needs dask
```

Layout: `archive.json` (the selection), `YYYY-MM.zarr` (completed months, Zarr v2, one chunk
per variable and timestamp), `reports/` (time and size of each month) and `logs/`. Hidden
`.YYYY-MM.zarr.partial` and `.YYYY-MM.progress` directories hold months in progress.

## 8. Sizes and speed

With the default selection, one hour is 0.32 GB uncompressed and about 0.2 GB on disk.
That comes to about 150 GB per month, or 1.77 TB for 2025.

Speed depends on your network. One task fetched a full day in about a minute from our login node.
On our compute nodes, an earlier version took 15 hours for all 12 months of 2025 running in parallel.

## Tests

```bash
uv run pytest
```

These tests run offline against a small fake bucket.
