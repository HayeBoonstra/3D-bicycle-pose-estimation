#!/usr/bin/env bash
set -euo pipefail

# Run commands in the MMPose conda environment without leaking compiler
# sysconfig overrides from other project workflows (e.g. PoseMamba, RF-DETR).

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command ...>"
  echo "Example: $0 mim train mmpose rtmpose_bicycle_full.py --work-dir ..."
  exit 2
fi

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

conda run -n mmpose "$@"
