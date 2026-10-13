# era5-dl

Downloads ERA5 (hourly, 0.25°) from Google's public bucket to a Slurm cluster, one Zarr file per month.

## Setup

```bash
git clone https://github.com/iSach/era5-dl && cd era5-dl
uv sync
```

You need [uv](https://docs.astral.sh/uv/). Clone the repo on a shared filesystem, and make sure the compute nodes have internet access.

## Download

```bash
uv run era5-dl submit -o /scratch/$USER/era5 --start 2025 --end 2025 -- --account=ACCOUNT --partition=PARTITION
```

This submits one Slurm job per month. Anything after `--` is passed to `sbatch`.

```bash
uv run era5-dl status -o /scratch/$USER/era5    # progress
```

- **If jobs stop** (time limit, crash, `scancel`), run the same `submit` command again. It continues where it stopped.
- **To add years**, run it again with the same `-o` and new dates.

A year of data is about 1.8 TB.

## Read

```python
import xarray as xr  # e.g. in: uv run --with xarray python

ds = xr.open_zarr("/scratch/me/era5/2025-01.zarr")
```

## Options

| Option | Meaning |
|---|---|
| `--variables a,b,c` | which variables (`uv run era5-dl variables` lists them all) |
| `--levels 500,850` | which pressure levels, or `all` |
| `--step-hours 6` | every 6 hours instead of every hour |
| `--era5t` | also get the latest preliminary data |
| `plan` instead of `submit` | show what would be downloaded and its size |
| `download` instead of `submit` | run on the current machine, without Slurm |

The default variables are temperature, u/v wind, geopotential and specific humidity on 13 pressure
levels (50–1000 hPa); 2 m temperature, 10 m u/v wind, sea-level pressure, sea surface temperature and
total precipitation; plus land-sea mask and orography fields.
