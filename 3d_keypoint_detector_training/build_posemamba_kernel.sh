#!/usr/bin/env bash
set -euo pipefail

# Build PoseMamba selective_scan extension in an isolated subshell/env.
# This avoids polluting your interactive shell and other training environments.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$ROOT_DIR/PoseMamba/kernels/selective_scan"

(
  unset _PYTHON_SYSCONFIGDATA_NAME
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
  export CUDAHOSTCXX=/usr/bin/g++-11

  conda run -n posemamba bash -lc "
    export CUDA_HOME=\"\$CONDA_PREFIX\"
    export PATH=\"\$CUDA_HOME/bin:\$PATH\"
    export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}\"
    cd \"$KERNEL_DIR\"
    python -m pip install -e .
  "
)

echo "PoseMamba selective_scan build complete."

