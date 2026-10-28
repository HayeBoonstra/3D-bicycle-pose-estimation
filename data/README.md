# Project data layout

Heavy datasets can live on a secondary drive while still appearing under `data/` in this repo.

## Secondary SSD (recommended)

Default mount: `/mnt/SmallSSD` (~217 GB free). Good for Blender renders and PoseMamba pickles; keep the git repo on your main disk.

One-time setup (creates folders on the SSD + symlinks into `data/`):

```bash
bash synthetic_data/pipeline/setup_secondary_data_disk.sh
```

Then generate data (auto-uses the SSD when `/mnt/SmallSSD` is mounted and writable):

```bash
bash synthetic_data/pipeline/generate_blender_posemamba_dataset.sh
```

Force SSD or main disk:

```bash
USE_SECONDARY_SSD=1 bash synthetic_data/pipeline/generate_blender_posemamba_dataset.sh
USE_SECONDARY_SSD=0 bash synthetic_data/pipeline/generate_blender_posemamba_dataset.sh
```

Custom mount point:

```bash
SECONDARY_DATA_ROOT=/mnt/MyDrive/bicycle-data bash synthetic_data/pipeline/setup_secondary_data_disk.sh
```

## Paths

| Artifact | On SSD (default) | In repo (symlink) |
|----------|------------------|------------------|
| Raw Blender clips | `/mnt/SmallSSD/3D-bicycle-pose-estimation/raw_blender_posemamba/` | `data/raw_blender_posemamba/` |
| PoseMamba pickles | `/mnt/SmallSSD/3D-bicycle-pose-estimation/posemamba_training_sequences/` | `data/posemamba_training_sequences/` |

Training still uses `SEQUENCE_ROOT` / `DATASET_TAG=detected2d` — point `start_training.sh` at the same sequence root (symlink works).

## Opening files in the IDE

`/mnt/...` is fine for Linux but awkward in some UIs. The symlinks under `data/` are the intended way to browse SSD data inside the project.

If symlinks fail (existing real folders in `data/`), move them to the SSD and re-run `setup_secondary_data_disk.sh`.
