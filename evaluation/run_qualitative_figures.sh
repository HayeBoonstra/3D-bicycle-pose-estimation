#!/usr/bin/env bash
# End-to-end qualitative evaluation figure pipeline.
#
# 1. Render 3 Blender scenes (frames kept)
# 2. Run full detection pipeline with capacity_l per clip
# 3. Export detected-2D overlay PNGs
# 4. Build side-by-side composite figures (RGB+2D | 3D pred+GT overlay)
#
# Usage:
#   ./evaluation/run_qualitative_figures.sh
#   SKIP_RENDER=1 ./evaluation/run_qualitative_figures.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/qualitative_eval/raw}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}/data/qualitative_eval/pipeline}"
FIGURES_ROOT="${FIGURES_ROOT:-${REPO_ROOT}/data/qualitative_eval/figures}"
LIFTER_CHECKPOINT="${LIFTER_CHECKPOINT:-${REPO_ROOT}/posemamba_weights/capacity_l/best_epoch.bin}"
LIFTER_CONFIG="${LIFTER_CONFIG:-${REPO_ROOT}/experiments/configs/capacity_l.yaml}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_CORPUS="${SKIP_CORPUS:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_COMPOSITE="${SKIP_COMPOSITE:-0}"

if [[ ! -f "${LIFTER_CHECKPOINT}" ]]; then
  echo "error: lifter checkpoint not found: ${LIFTER_CHECKPOINT}" >&2
  exit 1
fi

echo "[qualitative-figures] RAW_ROOT=${RAW_ROOT}"
echo "[qualitative-figures] PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "[qualitative-figures] FIGURES_ROOT=${FIGURES_ROOT}"
echo "[qualitative-figures] checkpoint=${LIFTER_CHECKPOINT}"

if [[ "${SKIP_CORPUS}" != "1" ]]; then
  SKIP_RENDER="${SKIP_RENDER}" bash "${REPO_ROOT}/experiments/build_qualitative_eval_corpus.sh"
fi

mkdir -p "${PIPELINE_ROOT}" "${FIGURES_ROOT}"

if [[ "${SKIP_PIPELINE}" != "1" ]]; then
  for clip_dir in "${RAW_ROOT}"/clip_*_qual_*; do
    [[ -d "${clip_dir}/frames" ]] || continue
    clip_name="$(basename "${clip_dir}")"
    out_dir="${PIPELINE_ROOT}/${clip_name}"
    echo "[qualitative-figures] pipeline -> ${clip_name}"
    bash "${REPO_ROOT}/1_full_detection_pipeline/run_full_pipeline.sh" \
      --frames-dir "${clip_dir}/frames" \
      --output-dir "${out_dir}" \
      --lifter-checkpoint "${LIFTER_CHECKPOINT}" \
      --lifter-config "${LIFTER_CONFIG}" \
      --no-visualize \
      --resume

    conda run -n mmpose python "${REPO_ROOT}/1_full_detection_pipeline/visualize_intermediates.py" \
      --output-dir "${out_dir}" \
      --stages 2d \
      --no-video \
      --resume

    overlay_dest="${FIGURES_ROOT}/overlays/${clip_name}"
    mkdir -p "${overlay_dest}"
    if [[ -d "${out_dir}/vis/keypoints_2d" ]]; then
      cp -n "${out_dir}/vis/keypoints_2d/"*.jpg "${overlay_dest}/" 2>/dev/null || true
    fi
  done
fi

if [[ "${SKIP_COMPOSITE}" != "1" ]]; then
  conda run -n posemamba python "${REPO_ROOT}/evaluation/make_qualitative_composite.py" \
    --raw-root "${RAW_ROOT}" \
    --pipeline-root "${PIPELINE_ROOT}" \
    --out "${FIGURES_ROOT}" \
    --frame-id auto \
    --scene-frame docks_scene=205 \
    --grid
fi

echo "[qualitative-figures] Done."
echo "  Composites: ${FIGURES_ROOT}/composite_*.png"
echo "  2D overlays: ${FIGURES_ROOT}/overlays/"
echo "  Manifest: ${FIGURES_ROOT}/manifest.json"
