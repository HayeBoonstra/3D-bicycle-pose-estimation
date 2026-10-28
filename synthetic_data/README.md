# Synthetic data generation

MuJoCo simulates bicycle trajectories; Blender renders RGB clips and exports J=18 2D/3D labels. Downstream scripts build PoseMamba pickles (T=243, stride 81).

## Layout

- `mujoco/` — physics, trajectories, optional MuJoCo-direct export (no RGB)
- `blender/` — scene assets, runtime scripts (`Animate_script.py`, `extract_2D_annotation.py`, …)
- `pipeline/` — orchestration (`generate_blender_posemamba_dataset.sh` is the main entry)

Large Blender assets (HDRIs, `.blend` scenes) may be gitignored; use `pipeline/setup_secondary_data_disk.sh` for `/mnt/SmallSSD` layout.

## Canonical flow (detected 2D, deployment-matched)

```bash
bash synthetic_data/pipeline/generate_blender_posemamba_dataset.sh
```

Then (from repo root, other conda envs): RF-DETR + RTMPose export and pickle build — see `training/pose3d/README.md` (`build_sequences.py`, `export_*`).

Legacy COCO-only path for old 2D training: `pipeline/legacy_coco/`.

Shared keypoint names: `common/bicycle_keypoint_schema.py`.
