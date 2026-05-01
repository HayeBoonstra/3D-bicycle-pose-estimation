"""Build temporal training sequences for 2D->3D lifting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.io_utils import dump_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal lifting sequences.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=27)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Placeholder index; actual 2D/3D pairing depends on your training export format.
    payload = {
        "dataset_root": str(args.dataset_root),
        "window_size": args.window_size,
        "note": "Populate with 2D/3D aligned temporal windows from metadata and 2D predictions.",
    }
    dump_json(args.out, payload)
    print(f"Wrote sequence manifest to {args.out}")


if __name__ == "__main__":
    main()

