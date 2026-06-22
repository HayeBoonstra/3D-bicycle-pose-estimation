#!/usr/bin/env bash
# Re-export keypoints_3d.jsonl (+ per_frame_annotations) for existing raw clips using manifest.csv.
# Skips PNG rendering. Requires the same trajectory CSV files used for the original render.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/mnt/SmallSSD/3D-bicycle-pose-estimation/raw_blender_posemamba}"
MANIFEST="${MANIFEST:-${RAW_ROOT}/manifest.csv}"
BLENDER="${BLENDER:-blender}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: manifest not found: ${MANIFEST}" >&2
  exit 1
fi

resolve_blender_bin() {
  if command -v "${BLENDER}" >/dev/null 2>&1; then
    command -v "${BLENDER}"
    return 0
  fi
  if [[ -x "${BLENDER}" ]]; then
    printf '%s\n' "${BLENDER}"
    return 0
  fi
  return 1
}

RESOLVED_BLENDER="$(resolve_blender_bin)" || {
  echo "error: Blender not found (${BLENDER})" >&2
  exit 1
}

tail -n +2 "${MANIFEST}" | while IFS=',' read -r clip_id scene_id blend trajectory_id trajectory_csv camera_seed _rest; do
  [[ -z "${clip_id}" ]] && continue
  out_dir="${RAW_ROOT}/${clip_id}"
  if [[ ! -d "${out_dir}" ]]; then
    echo "[reexport] skip missing clip dir: ${clip_id}" >&2
    continue
  fi
  echo "[reexport] ${clip_id}"
  "${RESOLVED_BLENDER}" --background "${blend}" \
    --python "${REPO_ROOT}/data_generation_pipeline_tools/render_clip.py" -- \
    --clip-id "${clip_id}" \
    --scene-id "${scene_id}" \
    --camera-seed "${camera_seed}" \
    --out "${out_dir}" \
    --trajectory-csv "${trajectory_csv}" \
    --annotations-only \
    --no-quiet-mode
done
