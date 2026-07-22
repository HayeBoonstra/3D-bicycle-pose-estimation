#!/usr/bin/env bash
# Render 3 visually distinct Blender scenes for qualitative evaluation figures.
#
# Outputs under data/qualitative_eval/raw/ with frames/ retained (no cleanup).
# Does not build PoseMamba training pickles.
#
# Usage:
#   ./experiments/build_qualitative_eval_corpus.sh
#   SKIP_RENDER=1 ./experiments/build_qualitative_eval_corpus.sh   # detection only
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/qualitative_eval/raw}"
TRAJECTORY_ROOT="${TRAJECTORY_ROOT:-${REPO_ROOT}/data/qualitative_eval/trajectories}"
TRAJECTORY_MANIFEST="${TRAJECTORY_MANIFEST:-${TRAJECTORY_ROOT}/manifest.csv}"
SYNC_WINDOW_SIZE="${SYNC_WINDOW_SIZE:-243}"
SEED="${SEED:-42}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_DETECTION="${SKIP_DETECTION:-0}"
BLENDER="${BLENDER:-blender}"
CONDA_POSEMAMBA="${CONDA_POSEMAMBA:-posemamba}"
CONDA_RFDETR="${CONDA_RFDETR:-rfdetr}"
CONDA_MMPOSE="${CONDA_MMPOSE:-mmpose}"

SCENES=(
  "docks_scene|docks_scene.blend|kids bike|KachujinS|101"
  "tree_lined_scene|tree_lined_scene.blend|gentlemen bike|Sophie|202"
  "evening_street_scene|evening_street_scene.blend|Swapfiets|Josh|303"
)

resolve_blender_bin() {
  local requested="$1"
  if command -v "$requested" >/dev/null 2>&1; then
    command -v "$requested"
    return 0
  fi
  if [[ "$requested" == "blender" ]]; then
    local candidate
    for candidate in \
      "${BLENDER_PATH:-}" \
      "$HOME/Desktop/blender/blender" \
      "/usr/bin/blender" \
      "/snap/bin/blender"; do
      if [[ -n "$candidate" && -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done
  elif [[ -x "$requested" ]]; then
    echo "$requested"
    return 0
  fi
  return 1
}

_run_py() {
  local env_name="$1"
  shift
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    python "$@"
  else
    conda run -n "${env_name}" python "$@"
  fi
}

mkdir -p "${RAW_ROOT}" "${TRAJECTORY_ROOT}"

echo "[qualitative] RAW_ROOT=${RAW_ROOT}"
echo "[qualitative] TRAJECTORY_ROOT=${TRAJECTORY_ROOT}"
echo "[qualitative] SYNC_WINDOW_SIZE=${SYNC_WINDOW_SIZE}"

_run_py "${CONDA_POSEMAMBA}" "${REPO_ROOT}/data_generation_pipeline_tools/ensure_mujoco_trajectories.py" \
  --trajectory-root "${TRAJECTORY_ROOT}" \
  --trajectory-manifest "${TRAJECTORY_MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --pattern composite \
  --num-trajectories 1 \
  --trajectory-frames "${SYNC_WINDOW_SIZE}" \
  --display-hz 60 \
  --seed-base "${SEED}" \
  --no-always-regenerate \
  --no-append

TRAJ_CSV="$(
  python3 - "${TRAJECTORY_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
with manifest.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty trajectory manifest")
print(rows[0]["trajectory_csv"])
PY
)"
if [[ ! -f "${TRAJ_CSV}" ]]; then
  echo "error: trajectory CSV not found: ${TRAJ_CSV}" >&2
  exit 1
fi
echo "[qualitative] trajectory: ${TRAJ_CSV}"

if [[ "${SKIP_RENDER}" != "1" ]]; then
  if ! RESOLVED_BLENDER="$(resolve_blender_bin "${BLENDER}")"; then
    echo "error: Blender executable not found: ${BLENDER}" >&2
    exit 1
  fi
  echo "[qualitative] blender: ${RESOLVED_BLENDER}"

  export CAMERA_MODE="${CAMERA_MODE:-track}"
  export CAMERA_MIN_DISTANCE="${CAMERA_MIN_DISTANCE:-4.0}"
  export CAMERA_MAX_DISTANCE="${CAMERA_MAX_DISTANCE:-10.0}"
  export CAMERA_MIN_BBOX_AREA_FRAC="${CAMERA_MIN_BBOX_AREA_FRAC:-0.01}"
  export CAMERA_MAX_BBOX_AREA_FRAC="${CAMERA_MAX_BBOX_AREA_FRAC:-0.80}"
  export CAMERA_MIN_VISIBLE_KEYPOINTS="${CAMERA_MIN_VISIBLE_KEYPOINTS:-14}"
  export CAMERA_MIN_VISIBLE_FRAME_RATIO="${CAMERA_MIN_VISIBLE_FRAME_RATIO:-0.9}"
  export CAMERA_FIT_MARGIN="${CAMERA_FIT_MARGIN:-1.25}"

  for entry in "${SCENES[@]}"; do
    IFS='|' read -r scene_id blend bike rider camera_seed <<<"${entry}"
    clip_id="clip_${scene_id}_qual_00000001"
    clip_dir="${RAW_ROOT}/${clip_id}"
    if [[ -f "${clip_dir}/keypoints_3d.jsonl" && -d "${clip_dir}/frames" ]]; then
      echo "[qualitative] skip render (exists): ${clip_id}"
      continue
    fi
    echo "[qualitative] rendering ${clip_id} ..."
    "${RESOLVED_BLENDER}" --background "${REPO_ROOT}/Blender files/Scenes/${blend}" \
      --python "${REPO_ROOT}/data_generation_pipeline_tools/render_clip.py" -- \
      --clip-id "${clip_id}" \
      --scene-id "${scene_id}" \
      --camera-seed "${camera_seed}" \
      --trajectory-csv "${TRAJ_CSV}" \
      --sync-window-size "${SYNC_WINDOW_SIZE}" \
      --out "${clip_dir}" \
      --bike "${bike}" \
      --rider "${rider}"
  done
else
  echo "[qualitative] SKIP_RENDER=1 — using existing clips under ${RAW_ROOT}"
fi

if [[ "${SKIP_DETECTION}" != "1" ]]; then
  _run_py "${CONDA_RFDETR}" "${REPO_ROOT}/3d_keypoint_detector_training/export_clip_detections.py" \
    --raw-root "${RAW_ROOT}" \
    --rfdetr-model "${RFDETR_MODEL:-rfdetr-2xlarge}" \
    --det-confidence "${DET_CONFIDENCE:-0.3}" \
    --resume

  _run_py "${CONDA_MMPOSE}" "${REPO_ROOT}/3d_keypoint_detector_training/export_detected_2d.py" \
    --raw-root "${RAW_ROOT}" \
    --mmpose-config "${MMPOSE_CONFIG:-${REPO_ROOT}/2d_keypoint_detector_training/rtmpose_bicycle_full.py}" \
    --mmpose-checkpoint "${MMPOSE_CHECKPOINT:-${REPO_ROOT}/training_outputs/mmpose_bicycle_rtmpose_l_gpu/best_coco_AP_epoch_175.pth}" \
    --pose-mode "${POSE_MODE:-detection_bbox}" \
    --device "${MMPOSE_DEVICE:-cuda:0}" \
    --resume
fi

echo "[qualitative] Done. Clips:"
find "${RAW_ROOT}" -maxdepth 1 -type d -name 'clip_*_qual_*' | sort
