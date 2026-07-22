#!/usr/bin/env python3
"""Publication figures for MuJoCo bicycle trajectory angle patterns.

Generates top-down path plots and heading/speed profile panels for every
pattern used in synthetic data generation (see Mujoco_bicycle_path_generator/
trajectory_patterns.py). Hard-turn and composite patterns are omitted from
the per-pattern grid; composite randomization is shown in a separate overlay.

Outputs (PDF + PNG) are written next to this script.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_DIR = REPO_ROOT / "Mujoco_bicycle_path_generator"
if str(MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_DIR))

from trajectory_patterns import (  # noqa: E402
    _segment_yaw,
    _speed_segment,
    _stabilize_angle_profile,
    build_angle_and_velocity_profiles,
    random_from_numpy,
)

PATTERNS_FOR_THESIS = (
    "straight",
    "circle-left",
    "circle-right",
    "zigzag",
    "avoidance",
    "lane-change",
    "soft-turn-left",
    "soft-turn-right",
    "u-turn-left",
    "u-turn-right",
    "figure-eight",
    "accelerate",
    "decelerate",
    "coast",
)

PATTERN_LABELS = {
    "straight": "Straight",
    "circle-left": "Circle (left)",
    "circle-right": "Circle (right)",
    "zigzag": "Zigzag",
    "avoidance": "Avoidance",
    "lane-change": "Lane change",
    "soft-turn-left": "Soft turn (left)",
    "soft-turn-right": "Soft turn (right)",
    "u-turn-left": "U-turn (left)",
    "u-turn-right": "U-turn (right)",
    "figure-eight": "Figure-eight",
    "accelerate": "Accelerate",
    "decelerate": "Decelerate",
    "coast": "Coast",
}


def resample_to_display(yaw, speed, physics_hz, display_hz):
    if physics_hz == display_hz:
        return yaw, speed
    physics_dt = 1.0 / physics_hz
    duration = (len(yaw) - 1) * physics_dt
    n_display = int(round(duration * display_hz)) + 1
    t_display = np.linspace(0.0, duration, n_display)
    t_physics = np.arange(len(yaw)) * physics_dt
    yaw_display = np.interp(t_display, t_physics, yaw)
    speed_display = np.interp(t_display, t_physics, speed)
    return yaw_display, speed_display


def generate_profile(
    pattern,
    *,
    seed,
    trajectory_frames,
    physics_hz,
    display_hz,
    min_speed,
    max_speed,
):
    physics_frames = int(math.ceil(trajectory_frames * physics_hz / display_hz))
    rng = np.random.default_rng(seed)
    py_rng = random_from_numpy(rng)
    start_speed = float(rng.uniform(min_speed, max_speed))
    yaw = _segment_yaw(pattern, physics_frames, py_rng, start_yaw=0.0, amplitude_deg=25.0)
    yaw = _stabilize_angle_profile(yaw, physics_hz, max_yaw_rate_deg_s=40.0)
    speed = _speed_segment(pattern, physics_frames, py_rng, start_speed, min_speed, max_speed)
    return resample_to_display(yaw, speed, physics_hz, display_hz)


def integrate_topdown_path(yaw_deg, speed_mps, dt_s, *, normalize_start_heading=True):
    """Integrate heading + speed into a top-down path starting at the origin."""
    if normalize_start_heading:
        heading_rad = np.deg2rad(yaw_deg - yaw_deg[0])
    else:
        heading_rad = np.deg2rad(yaw_deg)
    x = np.zeros(len(yaw_deg), dtype=float)
    y = np.zeros(len(yaw_deg), dtype=float)
    for idx in range(1, len(yaw_deg)):
        v = float(speed_mps[idx - 1])
        theta = heading_rad[idx - 1]
        x[idx] = x[idx - 1] + v * dt_s * np.sin(theta)
        y[idx] = y[idx - 1] + v * dt_s * np.cos(theta)
    return x, y


def generate_composite_profile(
    *,
    seed,
    trajectory_frames,
    physics_hz,
    display_hz,
    min_speed,
    max_speed,
    segment_min_seconds,
    segment_max_seconds,
    composite_profile,
    max_yaw_rate_deg_s,
):
    physics_frames = int(math.ceil(trajectory_frames * physics_hz / display_hz))
    yaw, speed, _segments = build_angle_and_velocity_profiles(
        pattern="composite",
        physics_frames=physics_frames,
        seed=seed,
        min_target_velocity=min_speed,
        max_target_velocity=max_speed,
        segment_min_seconds=segment_min_seconds,
        segment_max_seconds=segment_max_seconds,
        physics_hz=physics_hz,
        composite_profile=composite_profile,
    )
    yaw = _stabilize_angle_profile(yaw, physics_hz, max_yaw_rate_deg_s=max_yaw_rate_deg_s)
    return resample_to_display(yaw, speed, physics_hz, display_hz)


def _style_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.axvline(0.0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Lateral displacement (m)", fontsize=8)
    ax.set_ylabel("Forward displacement (m)", fontsize=8)


def plot_topdown_grid(profiles, output_stem: Path, dpi: int):
    n = len(profiles)
    ncols = 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    path_color = "#1a5276"
    paths = []
    for pattern, yaw, speed, dt_s in profiles:
        x, y = integrate_topdown_path(yaw, speed, dt_s)
        paths.append((pattern, x, y))

    x_min = min(float(x.min()) for _, x, _ in paths)
    x_max = max(float(x.max()) for _, x, _ in paths)
    y_min = min(float(y.min()) for _, _, y in paths)
    y_max = max(float(y.max()) for _, _, y in paths)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    half_span *= 1.08
    shared_xlim = (center_x - half_span, center_x + half_span)
    shared_ylim = (center_y - half_span, center_y + half_span)

    for ax, (pattern, x, y) in zip(axes_flat, paths):
        ax.plot(x, y, color=path_color, linewidth=1.6, solid_capstyle="round")
        ax.scatter(x[0], y[0], s=28, color="#27ae60", zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(x[-1], y[-1], s=28, color="#c0392b", zorder=3, edgecolors="white", linewidths=0.5)
        ax.set_title(PATTERN_LABELS.get(pattern, pattern), fontsize=9, fontweight="medium")
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)
        _style_axes(ax)

    for ax in axes_flat[len(profiles) :]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60", markersize=7, label="Start"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b", markersize=7, label="End"),
        Line2D([0], [0], color=path_color, linewidth=1.6, label="Integrated path"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "MuJoCo synthetic trajectory patterns (top-down, heading-normalised)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    for ext in ("pdf", "png"):
        out = output_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def plot_randomized_composites_topdown(
    paths,
    output_stem: Path,
    dpi: int,
    *,
    composite_profile: str,
    segment_min_seconds: float,
    segment_max_seconds: float,
):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)
    cmap = plt.get_cmap("turbo")
    n = len(paths)

    x_min = y_min = math.inf
    x_max = y_max = -math.inf
    for idx, (x, y) in enumerate(paths):
        color = cmap(idx / max(n - 1, 1))
        ax.plot(x, y, color=color, linewidth=1.2, alpha=0.72, solid_capstyle="round")
        x_min = min(x_min, float(x.min()))
        x_max = max(x_max, float(x.max()))
        y_min = min(y_min, float(y.min()))
        y_max = max(y_max, float(y.max()))

    ax.scatter(0.0, 0.0, s=42, color="#27ae60", zorder=4, edgecolors="white", linewidths=0.6)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    half_span = max(half_span * 1.08, 1.0)
    ax.set_xlim(center_x - half_span, center_x + half_span)
    ax.set_ylim(center_y - half_span, center_y + half_span)
    _style_axes(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60", markersize=7, label="Shared start"),
        Line2D([0], [0], color="#888888", linewidth=1.2, alpha=0.72, label="Composite sample"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    fig.suptitle(
        f"Composite trajectory randomization ({n} samples, top-down)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    ax.set_title(
        f"Segments drawn from {composite_profile} pool, "
        f"{segment_min_seconds:.0f}–{segment_max_seconds:.0f} s each",
        fontsize=9,
        color="#555555",
        pad=8,
    )

    for ext in ("pdf", "png"):
        out = output_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def plot_profile_grid(profiles, output_stem: Path, dpi: int):
    n = len(profiles)
    ncols = 5
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (pattern, yaw, speed, dt_s) in zip(axes_flat, profiles):
        time_s = np.arange(len(yaw)) * dt_s
        relative_yaw = yaw - yaw[0]
        ax2 = ax.twinx()
        ax.plot(time_s, relative_yaw, color="#2471a3", linewidth=1.4)
        ax2.plot(time_s, speed, color="#d35400", linewidth=1.2, alpha=0.85)
        ax.set_title(PATTERN_LABELS.get(pattern, pattern), fontsize=9, fontweight="medium")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Heading Δ (°)", fontsize=8, color="#2471a3")
        ax2.set_ylabel("Speed (m/s)", fontsize=8, color="#d35400")
        ax.tick_params(axis="y", labelcolor="#2471a3", labelsize=7)
        ax2.tick_params(axis="y", labelcolor="#d35400", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    for ax in axes_flat[len(profiles) :]:
        ax.axis("off")

    fig.suptitle(
        "Desired heading and target speed profiles per pattern",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    for ext in ("pdf", "png"):
        out = output_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--trajectory-frames", type=int, default=729, help="Export length at display Hz (per-pattern grid)")
    parser.add_argument(
        "--composite-trajectory-frames",
        type=int,
        default=2100,
        help="Export length at display Hz for composite overlay trajectories",
    )
    parser.add_argument("--physics-hz", type=int, default=200)
    parser.add_argument("--display-hz", type=int, default=60)
    parser.add_argument("--min-speed", type=float, default=2.0)
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (incremented per pattern)")
    parser.add_argument(
        "--composite-count",
        type=int,
        default=30,
        help="Number of randomized composite trajectories for the overlay figure",
    )
    parser.add_argument(
        "--composite-profile",
        choices=("stable", "full"),
        default="stable",
        help="Segment pool for composite trajectories (matches bicycle_test.py)",
    )
    parser.add_argument("--segment-min-seconds", type=float, default=2.0)
    parser.add_argument("--segment-max-seconds", type=float, default=7.0)
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=40.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dt_s = 1.0 / args.display_hz

    profiles = []
    for offset, pattern in enumerate(PATTERNS_FOR_THESIS):
        yaw, speed = generate_profile(
            pattern,
            seed=args.seed + offset,
            trajectory_frames=args.trajectory_frames,
            physics_hz=args.physics_hz,
            display_hz=args.display_hz,
            min_speed=args.min_speed,
            max_speed=args.max_speed,
        )
        profiles.append((pattern, yaw, speed, dt_s))

    plot_topdown_grid(profiles, args.output_dir / "trajectory_patterns_topdown", dpi=args.dpi)
    plot_profile_grid(profiles, args.output_dir / "trajectory_patterns_profiles", dpi=args.dpi)

    composite_paths = []
    for offset in range(args.composite_count):
        yaw, speed = generate_composite_profile(
            seed=args.seed + 10_000 + offset,
            trajectory_frames=args.composite_trajectory_frames,
            physics_hz=args.physics_hz,
            display_hz=args.display_hz,
            min_speed=args.min_speed,
            max_speed=args.max_speed,
            segment_min_seconds=args.segment_min_seconds,
            segment_max_seconds=args.segment_max_seconds,
            composite_profile=args.composite_profile,
            max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        )
        composite_paths.append(
            integrate_topdown_path(yaw, speed, dt_s, normalize_start_heading=False)
        )
    plot_randomized_composites_topdown(
        composite_paths,
        args.output_dir / "trajectory_patterns_composite_randomized",
        dpi=args.dpi,
        composite_profile=args.composite_profile,
        segment_min_seconds=args.segment_min_seconds,
        segment_max_seconds=args.segment_max_seconds,
    )


if __name__ == "__main__":
    main()
