#!/usr/bin/env bash
# Put large datasets on a secondary drive and link them into the repo for easy browsing.
#
# Example:
#   bash data_generation_pipeline_tools/setup_secondary_data_disk.sh
#   USE_SECONDARY_SSD=1 bash data_generation_pipeline_tools/generate_blender_posemamba_dataset.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SECONDARY_DATA_ROOT="${SECONDARY_DATA_ROOT:-/mnt/SmallSSD/3D-bicycle-pose-estimation}"
CREATE_SYMLINKS="${CREATE_SYMLINKS:-1}"

RAW_ON_SSD="${SECONDARY_DATA_ROOT}/raw_blender_posemamba"
SEQUENCES_ON_SSD="${SECONDARY_DATA_ROOT}/posemamba_training_sequences"

if [[ ! -d "/mnt/SmallSSD" ]]; then
  echo "error: /mnt/SmallSSD is not mounted. Mount the drive or set SECONDARY_DATA_ROOT." >&2
  exit 1
fi

if ! mkdir -p "${RAW_ON_SSD}" "${SEQUENCES_ON_SSD}" 2>/dev/null; then
  echo "error: cannot create directories under ${SECONDARY_DATA_ROOT}" >&2
  echo "  Fix permissions, e.g.: sudo chown -R \"\${USER}:\${USER}\" /mnt/SmallSSD" >&2
  exit 1
fi

echo "[setup] SSD data root: ${SECONDARY_DATA_ROOT}"
echo "[setup]   raw clips:    ${RAW_ON_SSD}"
echo "[setup]   sequences:  ${SEQUENCES_ON_SSD}"
df -h /mnt/SmallSSD | tail -1

_link_into_repo() {
  local name="$1"
  local target="$2"
  local link="${REPO_ROOT}/data/${name}"

  if [[ -L "${link}" ]]; then
    echo "[setup] symlink ${link} -> $(readlink -f "${link}")"
    return 0
  fi
  if [[ -e "${link}" ]]; then
    echo "[setup] WARN: ${link} exists and is not a symlink." >&2
    echo "       Move it onto the SSD, then re-run this script:" >&2
    echo "         mv \"${link}\" \"${target}\"" >&2
    echo "         bash data_generation_pipeline_tools/setup_secondary_data_disk.sh" >&2
    return 1
  fi
  ln -sfn "${target}" "${link}"
  echo "[setup] linked ${link} -> ${target}"
}

if [[ "${CREATE_SYMLINKS}" == "1" ]]; then
  _link_into_repo "raw_blender_posemamba" "${RAW_ON_SSD}" || true
  _link_into_repo "posemamba_training_sequences" "${SEQUENCES_ON_SSD}" || true
  echo "[setup] Open data/raw_blender_posemamba in the IDE (symlink to SSD)."
fi

cat <<EOF

Next runs (data on SSD):
  USE_SECONDARY_SSD=1 bash data_generation_pipeline_tools/generate_blender_posemamba_dataset.sh

Or export once in your shell:
  export USE_SECONDARY_SSD=1
  export SECONDARY_DATA_ROOT="${SECONDARY_DATA_ROOT}"

EOF
