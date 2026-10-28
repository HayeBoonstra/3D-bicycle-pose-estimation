#!/usr/bin/env bash
# Stop MMPose / torch.distributed training and verify GPUs are idle.
set -euo pipefail

echo "[stop] Sending SIGTERM to training processes..."
pkill -f "mmpose/.mim/tools/train.py" 2>/dev/null || true
pkill -f "torch.distributed.launch" 2>/dev/null || true
pkill -f "torch.distributed.run" 2>/dev/null || true
pkill -f "mim train mmpose" 2>/dev/null || true

for _ in 1 2 3 4 5; do
  sleep 1
  if ! pgrep -f "mmpose/.mim/tools/train.py|torch.distributed.(launch|run)" >/dev/null 2>&1; then
    break
  fi
done

if pgrep -f "mmpose/.mim/tools/train.py|torch.distributed.(launch|run)" >/dev/null 2>&1; then
  echo "[stop] Still running — sending SIGKILL..."
  pkill -9 -f "mmpose/.mim/tools/train.py" 2>/dev/null || true
  pkill -9 -f "torch.distributed.launch" 2>/dev/null || true
  pkill -9 -f "torch.distributed.run" 2>/dev/null || true
  sleep 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  nvidia-smi
  echo ""
  mapfile -t _used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
  for line in "${_used[@]}"; do
    idx="${line%%,*}"
    mib="${line##*,}"
    mib="${mib// /}"
    if [[ "${mib}" -gt 500 ]]; then
      echo "[stop] WARNING: GPU ${idx} still reports ${mib} MiB used."
      if [[ "${mib}" -gt 2000 ]]; then
        echo "[stop]   No process listed but high usage = orphaned CUDA memory (often GPU 0 after a crash)."
        echo "[stop]   Try: fuser -v /dev/nvidia${idx}  # find hidden holders"
        echo "[stop]   Or ask admin: sudo nvidia-smi --gpu-reset -i ${idx}"
        echo "[stop]   Or reboot the machine if memory won't clear."
      fi
    fi
  done
else
  echo "[stop] nvidia-smi not found; check GPU memory manually."
fi

if pgrep -f "mmpose/.mim/tools/train.py|torch.distributed.(launch|run)" >/dev/null 2>&1; then
  echo "[stop] ERROR: training processes still present:"
  pgrep -af "mmpose|torch.distributed" || true
  exit 1
fi

echo "[stop] Done."
