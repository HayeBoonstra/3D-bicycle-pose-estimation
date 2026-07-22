"""Shared utilities for thesis results evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

Input2DSource = Literal["detected", "gt"]

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Secondary SSD layout used by data_generation_pipeline_tools/setup_secondary_data_disk.sh
SSD_ROOT = Path("/mnt/SmallSSD/3D-bicycle-pose-estimation")


def resolve_data_path(
    relative: str,
    *,
    env_var: str | None = None,
) -> Path:
    """Resolve a data path: env override, repo data/, then SSD mirror."""
    if env_var:
        override = __import__("os").environ.get(env_var)
        if override:
            p = Path(override)
            if p.is_dir():
                return p.resolve()
            raise FileNotFoundError(f"{env_var} is not a directory: {p}")

    repo_path = (REPO_ROOT / relative).resolve()
    if repo_path.is_dir():
        return repo_path

    ssd_path = (SSD_ROOT / relative.split("data/", 1)[-1]).resolve()
    if ssd_path.is_dir():
        return ssd_path

    return repo_path  # fall back for clearer error messages


def default_raw_root() -> Path:
    return resolve_data_path("data/raw_blender_posemamba", env_var="RAW_ROOT")


def default_sequence_root() -> Path:
    return resolve_data_path("data/posemamba_training_sequences", env_var="DATA_ROOT")


def default_detected2d_test_dir() -> Path:
    return default_sequence_root() / "PoseMamba_f243s81_detected2d/BICYCLE/test"


def default_gt2d_test_dir() -> Path:
    return default_sequence_root() / "PoseMamba_f243s81/BICYCLE/test"


def _corpus_subdir_from_config(cfg: dict) -> str | None:
    data_root = cfg.get("data_root")
    if data_root:
        name = Path(str(data_root)).name
        if name.startswith("PoseMamba_f"):
            return name
    clip_len = cfg.get("clip_len")
    stride = cfg.get("data_stride")
    if clip_len is not None and stride is not None:
        tag = "detected2d" if cfg.get("gt_2d") is False else ""
        base = f"PoseMamba_f{int(clip_len)}s{int(stride)}"
        return f"{base}_{tag}" if tag else base
    return None


def _gt2d_corpus_subdir_from_config(cfg: dict) -> str | None:
    """Map a checkpoint config to the oracle (GT-2D) corpus folder name."""
    subdir = _corpus_subdir_from_config(cfg)
    if subdir is None:
        return None
    if subdir.endswith("_detected2d"):
        return subdir[: -len("_detected2d")]
    return subdir


def _resolve_test_dir(subdir: str, *, checkpoint_label: str, cfg_path: Path) -> Path:
    test_dir = default_sequence_root() / subdir / "BICYCLE" / "test"
    if test_dir.is_dir():
        return test_dir.resolve()
    raise FileNotFoundError(
        f"test pickle dir not found for {checkpoint_label}: {test_dir}\n"
        f"  (from {cfg_path.name})"
    )


def test_dir_for_input_2d(checkpoint: Path, input_2d: Input2DSource) -> Path:
    """Resolve BICYCLE/test pickles for detected-2D or oracle GT-2D evaluation."""
    if input_2d == "detected":
        return detected2d_test_dir_for_checkpoint(checkpoint)

    ckpt = checkpoint.resolve()
    for name in ("config.yaml", "train_config.yaml"):
        cfg_path = ckpt.parent / name
        if not cfg_path.is_file():
            continue
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        subdir = _gt2d_corpus_subdir_from_config(cfg)
        if subdir:
            return _resolve_test_dir(subdir, checkpoint_label=ckpt.parent.name, cfg_path=cfg_path)
    return default_gt2d_test_dir()


def detected2d_test_dir_for_checkpoint(checkpoint: Path) -> Path:
    """Resolve BICYCLE/test pickles matching a checkpoint's training corpus."""
    ckpt = checkpoint.resolve()
    for name in ("config.yaml", "train_config.yaml"):
        cfg_path = ckpt.parent / name
        if not cfg_path.is_file():
            continue
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        subdir = _corpus_subdir_from_config(cfg)
        if subdir:
            return _resolve_test_dir(subdir, checkpoint_label=ckpt.parent.name, cfg_path=cfg_path)
    return default_detected2d_test_dir()


# Capacity ablation experiments (S/B/L/X) for GT-2D oracle eval and comparison tables.
CAPACITY_EXPERIMENTS = ("capacity_s", "capacity_b", "capacity_l", "capacity_x")
# Models trained on the GT-projected 2D corpus (PoseMamba_f243s81_gt).
CAPACITY_GT_TRAINING_EXPERIMENTS = (
    "capacity_s_gt",
    "capacity_b_gt",
    "capacity_l_gt",
    "capacity_x_gt",
)
HEADLINE_EXPERIMENT = "capacity_l"

# Temporal window ablations (training clip length T).
WINDOW_EXPERIMENTS = ("window_t27", "window_t81", "window_t121", "window_t162")
# T=243 baseline: same PoseMamba-B setup on f243s81_detected2d (not retrained in run_window_ablations.sh).
WINDOW_BASELINE_EXPERIMENT = "capacity_b"
WINDOW_ABLATION_PLOT_EXPERIMENTS = WINDOW_EXPERIMENTS + (WINDOW_BASELINE_EXPERIMENT,)

# Joint groups for per-group error reporting.
JOINT_GROUPS = {
    "wheels": [8, 9, 10, 11, 12, 13, 14, 15],
    "frame": [0, 1, 2, 3, 4],
    "steering": [5, 6, 7, 16, 17],
}

# Clips above this per-clip MPJPE are excluded from aggregate 3D/dynamics metrics
# (likely out-of-distribution scenes absent from training).
DEFAULT_MAX_CLIP_MPJPE_MM = 70.0
DEFAULT_SEQUENCE_FPS = 60.0

# Steer-loss ablations that failed to train sensibly; omit from summary tables/plots.
EXCLUDED_ABLATION_EXPERIMENTS = frozenset(
    {
        "dyn_steer",
        "dyn_steer_roll",
        "dyn_steer_roll_vel",
    }
)


def unique_clip_ids_ordered(clip_ids) -> list[str]:
    ids: list[str] = []
    for c in clip_ids:
        s = str(c)
        if s not in ids:
            ids.append(s)
    return ids


def frame_mask_for_clips(clip_ids, accepted_clip_ids: set[str]) -> np.ndarray:
    accepted = {str(c) for c in accepted_clip_ids}
    return np.array([str(c) in accepted for c in clip_ids])


def first_clip_mask(clip_ids, accepted_clip_ids: set[str] | None = None):
    """Return (clip_id, boolean frame mask) for the first clip in timeline order."""
    if clip_ids is None:
        return None, None
    accepted = None if accepted_clip_ids is None else {str(c) for c in accepted_clip_ids}
    for cid in unique_clip_ids_ordered(clip_ids):
        if accepted is None or cid in accepted:
            return cid, np.array([str(c) == cid for c in clip_ids])
    return None, None


def contiguous_segment_slices(frame_idx) -> list[slice]:
    """Return slices for every run where source frame indices increase by 1."""
    idx = np.asarray(frame_idx, dtype=np.int64)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    run_starts = np.concatenate([[0], breaks + 1])
    run_ends = np.concatenate([breaks + 1, [len(idx)]])
    return [slice(int(s), int(e)) for s, e in zip(run_starts, run_ends)]


def longest_contiguous_slice(frame_idx) -> slice:
    """Slice into the longest run where source frame indices increase by 1.

    Test corpora use window-level splits: a single clip_id in preds_3d.npz often
    concatenates scattered windows (gaps of hundreds of frames). Plotting those
    arrays with np.arange(n) falsely shows vertical jumps in both pred and GT.
    """
    segments = contiguous_segment_slices(frame_idx)
    if not segments:
        return slice(0, 0)
    return max(segments, key=lambda seg: seg.stop - seg.start)


def mm_from_m(value_m: float) -> float:
    return float(value_m) * 1000.0


def bbox_xywh_to_xyxy(bbox: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU for xyxy boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def group_mean(values_per_joint: np.ndarray, group_name: str) -> float:
    idx = JOINT_GROUPS[group_name]
    return float(np.mean(values_per_joint[idx]))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    r = pearson_r(x, y)
    return float(r * r) if np.isfinite(r) else float("nan")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
