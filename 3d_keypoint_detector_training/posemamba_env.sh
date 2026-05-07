#!/usr/bin/env bash
set -euo pipefail

# Run commands in a clean PoseMamba conda environment without leaking
# compiler/sysconfig overrides into other project workflows (e.g. RTMPose).

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command ...>"
  echo "Example: $0 python train.py --help"
  exit 2
fi

unset _PYTHON_SYSCONFIGDATA_NAME
unset CC
unset CXX
unset CUDAHOSTCXX

conda run -n posemamba "$@"

