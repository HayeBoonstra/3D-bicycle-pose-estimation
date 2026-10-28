#!/usr/bin/env python3
"""Stage 4: roll / steer from 3D keypoints (conda env: posemamba or base)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.bicycle_dynamics_angles import (  # noqa: E402
    bicycle_roll_angle,
    bicycle_steer_angle,
)
from inference.pipeline_io import KEYPOINTS_3D_NAME


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: bicycle roll and steer from 3D keypoints.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    kp3d_path = output_dir / KEYPOINTS_3D_NAME
    dynamics_npz = output_dir / "dynamics.npz"
    dynamics_json = output_dir / "dynamics.json"

    if args.resume and dynamics_npz.is_file() and dynamics_json.is_file():
        print(f"[stage4] skip (exists): {dynamics_npz}")
        return

    if not kp3d_path.is_file():
        raise FileNotFoundError(f"Missing 3D keypoints: {kp3d_path}")

    data = np.load(kp3d_path)
    keypoints_3d = np.asarray(data["keypoints_3d"], dtype=np.float32)
    steer = bicycle_steer_angle(keypoints_3d)
    roll = bicycle_roll_angle(keypoints_3d)
    frames = np.arange(keypoints_3d.shape[0], dtype=np.int32)

    np.savez_compressed(
        dynamics_npz,
        steer=steer.astype(np.float32),
        roll=roll.astype(np.float32),
        frame_index=frames,
    )
    payload = {
        "steer_rad": steer.tolist(),
        "roll_rad": roll.tolist(),
        "frame_index": frames.tolist(),
    }
    dynamics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[stage4] wrote {dynamics_npz} and {dynamics_json}")


if __name__ == "__main__":
    main()
