#!/usr/bin/env python3
"""Merge trajectory manifest with history and generate only missing/new CSVs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.pipeline_clip_registry import (  # noqa: E402
    TRAJECTORY_MANIFEST_FIELDS,
    merge_trajectory_manifest,
    trajectory_file_sha256,
    write_csv_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pattern", default="composite")
    parser.add_argument("--num-trajectories", type=int, required=True)
    parser.add_argument("--trajectory-frames", type=int, default=2187)
    parser.add_argument("--display-hz", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--physics-hz", type=int, default=200)
    parser.add_argument("--always-regenerate", action="store_true")
    parser.add_argument("--no-always-regenerate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add num-trajectories NEW trajectories on top of existing ones (default).",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Use the legacy fixed-count contract (ensure indices 0..N-1).",
    )
    parser.add_argument("--gen-retries", type=int, default=8)
    parser.add_argument("--bicycle-test", type=Path, default=REPO_ROOT / "Mujoco_bicycle_path_generator" / "bicycle_test.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-target-velocity-mps", type=float, default=3.5)
    parser.add_argument("--max-target-velocity-mps", type=float, default=8.0)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=25.0)
    parser.add_argument("--composite-profile", default="stable")
    parser.add_argument("--max-roll-deg", type=float, default=40.0)
    parser.add_argument("--segment-min-seconds", type=float, default=3.0)
    parser.add_argument("--segment-max-seconds", type=float, default=7.0)
    return parser.parse_args()


def _run_bicycle_test(args: argparse.Namespace, row: dict[str, str], trajectory_seed: int) -> None:
    csv_path = Path(row["trajectory_csv"])
    cmd = [
        args.python,
        str(args.bicycle_test),
        "--pattern",
        row.get("pattern", args.pattern),
        "--seed",
        str(trajectory_seed),
        "--trajectory-frames",
        str(args.trajectory_frames),
        "--physics-hz",
        str(args.physics_hz),
        "--display-hz",
        str(args.display_hz),
        "--min-target-velocity-mps",
        str(args.min_target_velocity_mps),
        "--max-target-velocity-mps",
        str(args.max_target_velocity_mps),
        "--max-yaw-rate-deg-s",
        str(args.max_yaw_rate_deg_s),
        "--composite-profile",
        args.composite_profile,
        "--max-roll-deg",
        str(args.max_roll_deg),
        "--segment-min-seconds",
        str(args.segment_min_seconds),
        "--segment-max-seconds",
        str(args.segment_max_seconds),
        "--output-path",
        str(csv_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    append_mode = not args.no_append  # default True, override only with --no-append
    rows, logs = merge_trajectory_manifest(
        trajectory_root=args.trajectory_root,
        manifest_path=args.trajectory_manifest,
        pattern=args.pattern,
        num_trajectories=args.num_trajectories,
        frames=args.trajectory_frames,
        display_hz=args.display_hz,
        seed_base=args.seed_base,
        always_regenerate=bool(args.always_regenerate) and not args.no_always_regenerate,
        overwrite=args.overwrite,
        raw_root=args.raw_root,
        append=append_mode,
    )
    for line in logs:
        print(line)

    final_rows: list[dict[str, str]] = []
    for row in rows:
        out_row = {k: row.get(k, "") for k in TRAJECTORY_MANIFEST_FIELDS}
        if row.get("should_generate") != "1":
            csv_path = Path(out_row["trajectory_csv"])
            if csv_path.is_file():
                out_row["sha256"] = trajectory_file_sha256(csv_path)
            final_rows.append(out_row)
            continue

        trajectory_id = out_row["trajectory_id"]
        trajectory_idx = int(trajectory_id.rsplit("_", 1)[-1])
        seed_base = int(out_row.get("seed") or (args.seed_base + trajectory_idx))
        gen_ok = False
        used_seed = seed_base
        for attempt in range(args.gen_retries):
            trajectory_seed = seed_base + 10007 * attempt
            try:
                _run_bicycle_test(args, out_row, trajectory_seed)
                gen_ok = True
                used_seed = trajectory_seed
                break
            except subprocess.CalledProcessError:
                print(f"[ensure-trajectories] {trajectory_id} attempt {attempt + 1} failed", file=sys.stderr)
        if not gen_ok:
            raise SystemExit(f"failed to generate trajectory {trajectory_id} after {args.gen_retries} attempts")

        csv_path = Path(out_row["trajectory_csv"])
        out_row["sha256"] = trajectory_file_sha256(csv_path)
        out_row["seed"] = str(used_seed)
        final_rows.append(out_row)
        print(f"[ensure-trajectories] generated {trajectory_id} -> {csv_path.name} sha256={out_row['sha256'][:12]}...")

    write_csv_manifest(args.trajectory_manifest, TRAJECTORY_MANIFEST_FIELDS, final_rows)


if __name__ == "__main__":
    main()
