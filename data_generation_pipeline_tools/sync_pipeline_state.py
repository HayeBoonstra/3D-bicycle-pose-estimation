#!/usr/bin/env python3
"""Reconcile trajectory/raw manifests with on-disk clips (incremental pipeline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.pipeline_clip_registry import (  # noqa: E402
    RAW_MANIFEST_FIELDS,
    audit_raw_root,
    backfill_render_config,
    clip_to_manifest_row,
    merge_trajectory_manifest,
    scan_raw_clips,
    trajectory_status_for_clip,
    write_csv_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync MuJoCo/raw manifests with existing clips.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--pattern", default="composite")
    parser.add_argument("--num-trajectories", type=int, default=0)
    parser.add_argument("--trajectory-frames", type=int, default=2187)
    parser.add_argument("--display-hz", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--always-regenerate", action="store_true")
    parser.add_argument("--no-always-regenerate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append num-trajectories NEW trajectories to the manifest (default).",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Legacy fixed-count contract (ensure indices 0..N-1).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print clip/trajectory audit JSON and exit (no manifest rewrite).",
    )
    parser.add_argument(
        "--rebuild-raw-manifest",
        action="store_true",
        help="Rebuild raw_blender_posemamba/manifest.csv from on-disk clips only.",
    )
    parser.add_argument(
        "--backfill-render-config",
        action="store_true",
        help="Write trajectory_csv + trajectory_sha256 into each clip render_config.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional raw manifest.csv for trajectory_csv when render_config lacks it.",
    )
    return parser.parse_args()


def _load_manifest_trajectory_map(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    import csv

    mapping: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip_id = (row.get("clip_id") or "").strip()
            trajectory_csv = (row.get("trajectory_csv") or "").strip()
            if clip_id and trajectory_csv:
                mapping[clip_id] = trajectory_csv
    return mapping


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    trajectory_root = args.trajectory_root.expanduser().resolve()

    audit = audit_raw_root(raw_root)
    audit["clips"] = [
        {
            "clip_id": c.clip_id,
            "trajectory_csv": c.trajectory_csv,
            "trajectory_sha256": c.trajectory_sha256[:16] + "..." if c.trajectory_sha256 else "",
            "has_dynamics_gt": c.has_dynamics_gt,
            "n_frames": c.n_frames,
        }
        for c in scan_raw_clips(raw_root)
    ]
    print(json.dumps(audit, indent=2))

    if args.audit_only:
        return

    if args.backfill_render_config:
        manifest_map = _load_manifest_trajectory_map(
            args.manifest or (raw_root / "manifest.csv")
        )
        updated = 0
        for clip in scan_raw_clips(raw_root):
            csv_raw = clip.trajectory_csv or manifest_map.get(clip.clip_id, "")
            ok, msg = backfill_render_config(
                clip,
                trajectory_csv=Path(csv_raw) if csv_raw else None,
            )
            status = trajectory_status_for_clip(clip)
            print(f"[backfill] {clip.clip_id}: {msg} (post-check={status})")
            if ok:
                updated += 1
        print(f"[sync] backfilled render_config for {updated} clip(s)")
        return

    if args.rebuild_raw_manifest:
        rows = [clip_to_manifest_row(c) for c in scan_raw_clips(raw_root)]
        write_csv_manifest(raw_root / "manifest.csv", RAW_MANIFEST_FIELDS, rows)
        print(f"[sync] rebuilt {raw_root / 'manifest.csv'} ({len(rows)} clips)")
        return

    if args.num_trajectories < 1:
        raise SystemExit("--num-trajectories is required unless using --audit-only, --backfill-render-config, or --rebuild-raw-manifest")

    always_regenerate = bool(args.always_regenerate) and not args.no_always_regenerate
    append_mode = not args.no_append
    _, logs = merge_trajectory_manifest(
        trajectory_root=trajectory_root,
        manifest_path=args.trajectory_manifest,
        pattern=args.pattern,
        num_trajectories=args.num_trajectories,
        frames=args.trajectory_frames,
        display_hz=args.display_hz,
        seed_base=args.seed_base,
        always_regenerate=always_regenerate,
        overwrite=args.overwrite,
        raw_root=raw_root,
        append=append_mode,
    )
    for line in logs:
        print(line)


if __name__ == "__main__":
    main()
