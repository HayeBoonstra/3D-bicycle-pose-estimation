"""Load per-frame bicycle dynamics (steer, roll) from MuJoCo trajectory CSV exports."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


def roll_deg_from_quat_wxyz(rw: float, rx: float, ry: float, rz: float) -> float:
    """Body roll about X in degrees (matches bicycle_test rejection metric, xyz Euler)."""
    sinr_cosp = 2.0 * (rw * rx + ry * rz)
    cosr_cosp = 1.0 - 2.0 * (rx * rx + ry * ry)
    return math.degrees(math.atan2(sinr_cosp, cosr_cosp))


def _parse_trajectory_row(row: dict[str, str]) -> dict[str, float]:
    def _f(key: str) -> float:
        raw = row.get(key)
        if raw in {None, ""}:
            raise KeyError(f"trajectory CSV row missing required column {key!r}")
        return float(raw)

    rw, rx, ry, rz = _f("rw"), _f("rx"), _f("ry"), _f("rz")
    return {
        "steer_deg": _f("steer_angle"),
        "roll_deg": roll_deg_from_quat_wxyz(rw, rx, ry, rz),
        "rear_wheel_deg": _f("rear_wheel_angle"),
        "front_wheel_deg": _f("front_wheel_angle"),
        "crank_deg": _f("crank_angle"),
    }


def load_trajectory_dynamics_csv(path: Path) -> list[dict[str, float]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "steer_angle" not in reader.fieldnames:
            raise ValueError(f"Trajectory CSV missing steer_angle column: {path}")
        for row in reader:
            rows.append(_parse_trajectory_row(row))
    if not rows:
        raise ValueError(f"Trajectory CSV is empty: {path}")
    return rows


def dynamics_gt_payload(
    trajectory_csv: Path,
    frame_index: int,
    *,
    trajectory_rows: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    csv_path = Path(trajectory_csv).expanduser().resolve()
    rows = trajectory_rows if trajectory_rows is not None else load_trajectory_dynamics_csv(csv_path)
    if frame_index < 0 or frame_index >= len(rows):
        raise IndexError(
            f"frame_index {frame_index} out of range for trajectory {csv_path} ({len(rows)} rows)"
        )
    row = rows[frame_index]
    return {
        "steer_deg": float(row["steer_deg"]),
        "roll_deg": float(row["roll_deg"]),
        "source": "mujoco_trajectory_csv",
        "trajectory_csv": str(csv_path),
        "trajectory_row": int(frame_index),
    }


def assert_trajectory_frame_count(trajectory_csv: Path, num_frames: int) -> list[dict[str, float]]:
    rows = load_trajectory_dynamics_csv(trajectory_csv)
    if len(rows) != num_frames:
        raise ValueError(
            f"Trajectory row count ({len(rows)}) != clip frame count ({num_frames}) for {trajectory_csv}"
        )
    return rows
