#!/usr/bin/env bash
# Greenlight: render mujoco_composite_00004 (243 f, fixed near-horizontal cam) → full pipeline → composite video.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/Thesis results visualizations/Full pipeline visualization with dynamics}"

TRAJ_FULL="${TRAJ_FULL:-/mnt/SmallSSD/3D-bicycle-pose-estimation/mujoco_blender_trajectories/mujoco_composite_00004_60hz.csv}"
TRAJ_TRIM="${OUT_ROOT}/mujoco_composite_00004_f243.csv"
FRAMES_N="${FRAMES_N:-243}"

SCENE_ID="${SCENE_ID:-evening_street_scene}"
BLEND="${REPO_ROOT}/Blender files/Scenes/evening_street_scene.blend"
BIKE="${BIKE:-Swapfiets}"
RIDER="${RIDER:-Josh}"
CAMERA_SEED="${CAMERA_SEED:-42424204}"
CLIP_ID="${CLIP_ID:-clip_${SCENE_ID}_mujoco_composite_00004_${CAMERA_SEED}_greenlight}"

RAW_CLIP="${OUT_ROOT}/raw/${CLIP_ID}"
PIPELINE_DIR="${OUT_ROOT}/pipeline"
VIS_DIR="${OUT_ROOT}/vis"

BLENDER="${BLENDER:-$HOME/Desktop/blender/blender}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_COMPOSITE="${SKIP_COMPOSITE:-0}"
VIS_FPS="${VIS_FPS:-30}"
GREENLIGHT_FPS="${GREENLIGHT_FPS:-30}"

CONDA_RFDETR="${CONDA_RFDETR:-rfdetr}"
CONDA_MMPOSE="${CONDA_MMPOSE:-mmpose}"
CONDA_POSEMAMBA="${CONDA_POSEMAMBA:-posemamba}"

_run_py() {
  local env_name="$1"
  shift
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    python "$@"
  else
    conda run -n "${env_name}" python "$@"
  fi
}

resolve_blender() {
  if [[ -x "${BLENDER}" ]]; then
    echo "${BLENDER}"
    return 0
  fi
  for candidate in "${BLENDER_PATH:-}" "$HOME/Desktop/blender/blender" /usr/bin/blender; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

mkdir -p "${OUT_ROOT}/raw" "${PIPELINE_DIR}" "${VIS_DIR}"

echo "[greenlight] trimming trajectory to ${FRAMES_N} rows → ${TRAJ_TRIM}"
python3 - <<PY
import csv
from pathlib import Path

src = Path("${TRAJ_FULL}")
dst = Path("${TRAJ_TRIM}")
n = int("${FRAMES_N}")
rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
if len(rows) < n:
    raise SystemExit(f"trajectory has {len(rows)} rows, need {n}")
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows[:n])
print(f"Wrote {n} rows to {dst}")
PY

if [[ "${SKIP_RENDER}" != "1" ]]; then
  RESOLVED_BLENDER="$(resolve_blender)" || {
    echo "error: Blender not found at ${BLENDER}" >&2
    exit 1
  }
  echo "[greenlight] Blender: ${RESOLVED_BLENDER}"

  export CAMERA_MODE=fixed
  export CAMERA_ELEVATION_MEAN_DEG="${CAMERA_ELEVATION_MEAN_DEG:-88}"
  export CAMERA_ELEVATION_STD_DEG="${CAMERA_ELEVATION_STD_DEG:-4}"
  export CAMERA_MIN_DISTANCE="${CAMERA_MIN_DISTANCE:-6.0}"
  export CAMERA_MAX_DISTANCE="${CAMERA_MAX_DISTANCE:-22.0}"
  export CAMERA_MIN_BBOX_AREA_FRAC="${CAMERA_MIN_BBOX_AREA_FRAC:-0.001}"
  export CAMERA_MAX_BBOX_AREA_FRAC="${CAMERA_MAX_BBOX_AREA_FRAC:-0.80}"
  export CAMERA_MIN_VISIBLE_KEYPOINTS="${CAMERA_MIN_VISIBLE_KEYPOINTS:-12}"
  export CAMERA_MIN_VISIBLE_FRAME_RATIO="${CAMERA_MIN_VISIBLE_FRAME_RATIO:-0.55}"
  export CAMERA_MIN_HEIGHT_ABOVE_TARGET="${CAMERA_MIN_HEIGHT_ABOVE_TARGET:--0.5}"
  export CAMERA_MIN_VIEW_PITCH_DEG="${CAMERA_MIN_VIEW_PITCH_DEG:--25}"

  mkdir -p "${RAW_CLIP}"
  RENDER_OK=0
  for try_seed in $(seq "${CAMERA_SEED}" $((CAMERA_SEED + 24))); do
    echo "[greenlight] render attempt camera_seed=${try_seed} ..."
    if "${RESOLVED_BLENDER}" --background "${BLEND}" \
      --python "${REPO_ROOT}/data_generation_pipeline_tools/render_clip.py" -- \
      --clip-id "${CLIP_ID}" \
      --scene-id "${SCENE_ID}" \
      --camera-seed "${try_seed}" \
      --trajectory-csv "${TRAJ_TRIM}" \
      --camera-mode fixed \
      --frame-start 1 \
      --frame-end "${FRAMES_N}" \
      --no-sync-camera-window \
      --sync-window-size "${FRAMES_N}" \
      --out "${RAW_CLIP}" \
      --bike "${BIKE}" \
      --rider "${RIDER}" \
      --camera-target k_handlebar_middle \
      --no-quiet-mode \
      --encode-video; then
      if [[ -f "${RAW_CLIP}/keypoints_3d.jsonl" ]]; then
        CAMERA_SEED="${try_seed}"
        RENDER_OK=1
        echo "[greenlight] render succeeded with camera_seed=${try_seed}"
        break
      fi
    fi
    echo "[greenlight] render failed for seed ${try_seed}, retrying..."
    rm -rf "${RAW_CLIP}/frames" "${RAW_CLIP}/keypoints_3d.jsonl" 2>/dev/null || true
  done
  if [[ "${RENDER_OK}" -ne 1 ]]; then
    echo "error: Blender render failed after seed retries (fixed camera + trajectory span)." >&2
    exit 1
  fi
else
  echo "[greenlight] SKIP_RENDER=1"
fi

FRAMES_DIR="${RAW_CLIP}/frames"
if [[ ! -d "${FRAMES_DIR}" ]]; then
  echo "error: missing frames dir ${FRAMES_DIR}" >&2
  exit 1
fi
NIMG=$(find "${FRAMES_DIR}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
if [[ "${NIMG}" -ne "${FRAMES_N}" ]]; then
  echo "error: expected ${FRAMES_N} frames, found ${NIMG} in ${FRAMES_DIR}" >&2
  exit 1
fi

if [[ "${SKIP_PIPELINE}" != "1" ]]; then
  echo "[greenlight] running full detection pipeline..."
  "${REPO_ROOT}/1_full_detection_pipeline/run_full_pipeline.sh" \
    --frames-dir "${FRAMES_DIR}" \
    --output-dir "${PIPELINE_DIR}" \
    --mmpose-checkpoint "${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth" \
    --lifter-checkpoint "${REPO_ROOT}/1_full_detection_pipeline/posemamba_X_best_epoch.bin" \
    --lifter-config "${REPO_ROOT}/1_full_detection_pipeline/PoseMamba_train_bicycle_X.generated.yaml" \
    --no-visualize \
    --resume

  echo "[greenlight] rendering stage visualization MP4s..."
  _run_py "${CONDA_MMPOSE}" "${REPO_ROOT}/1_full_detection_pipeline/visualize_intermediates.py" \
    --output-dir "${PIPELINE_DIR}" --fps "${VIS_FPS}" --resume --stages detections 2d
  _run_py "${CONDA_POSEMAMBA}" "${REPO_ROOT}/1_full_detection_pipeline/visualize_intermediates.py" \
    --output-dir "${PIPELINE_DIR}" --fps "${VIS_FPS}" --resume --stages 3d
else
  echo "[greenlight] SKIP_PIPELINE=1"
fi

for name in detections keypoints_2d; do
  src="${PIPELINE_DIR}/vis/${name}"
  if [[ -d "${src}" ]]; then
    ln -sfn "${src}" "${VIS_DIR}/${name}" 2>/dev/null || cp -a "${src}" "${VIS_DIR}/${name}"
  fi
done
if [[ -f "${PIPELINE_DIR}/vis/keypoints_3d.mp4" ]]; then
  cp -f "${PIPELINE_DIR}/vis/keypoints_3d.mp4" "${VIS_DIR}/keypoints_3d.mp4"
fi
if [[ -f "${PIPELINE_DIR}/vis/detections/detections.mp4" ]]; then
  cp -f "${PIPELINE_DIR}/vis/detections/detections.mp4" "${VIS_DIR}/detections.mp4"
fi
if [[ -f "${PIPELINE_DIR}/vis/keypoints_2d/keypoints_2d.mp4" ]]; then
  cp -f "${PIPELINE_DIR}/vis/keypoints_2d/keypoints_2d.mp4" "${VIS_DIR}/keypoints_2d.mp4"
fi

MANIFEST="${OUT_ROOT}/manifest.json"
python3 - <<PY
import json
from pathlib import Path
manifest = {
    "clip_id": "${CLIP_ID}",
    "camera_seed": int("${CAMERA_SEED}"),
    "trajectory_trim_csv": "${TRAJ_TRIM}",
    "raw_clip_dir": "${RAW_CLIP}",
    "pipeline_dir": "${PIPELINE_DIR}",
    "vis_dir": "${VIS_DIR}",
    "frames": int("${FRAMES_N}"),
    "camera_mode": "fixed",
}
Path("${MANIFEST}").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote {manifest['clip_id']} manifest")
PY

if [[ "${SKIP_COMPOSITE}" != "1" ]]; then
  echo "[greenlight] building 2D + 3D/dynamics meeting videos..."
  _run_py "${CONDA_POSEMAMBA}" \
    "${REPO_ROOT}/Thesis results visualizations/make_full_pipeline_dynamics_video.py" \
    --raw-clip-dir "${RAW_CLIP}" \
    --pipeline-dir "${PIPELINE_DIR}" \
    --out-2d-mp4 "${VIS_DIR}/greenlight_2d_keypoints.mp4" \
    --out-3d-dynamics-mp4 "${VIS_DIR}/greenlight_3d_dynamics.mp4" \
    --trajectory-csv "${TRAJ_TRIM}" \
    --fps "${GREENLIGHT_FPS}" \
    --out-width 1280
fi

echo "[greenlight] done → ${OUT_ROOT}"
