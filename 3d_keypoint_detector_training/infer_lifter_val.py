"""Compatibility wrapper.

Use `3D_lifting_inference.py` as the main entrypoint for sequence inference.
This wrapper forwards execution there so old commands keep working.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).resolve().with_name("3D_lifting_inference.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
