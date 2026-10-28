"""Desired-yaw and speed profile generators for MuJoCo bicycle trajectories."""

from __future__ import annotations

import random

import numpy as np

ANGLE_PATTERNS = (
    "straight",
    "circle-left",
    "circle-right",
    "zigzag",
    "avoidance",
    "lane-change",
    "soft-turn-left",
    "soft-turn-right",
    "hard-turn-left",
    "hard-turn-right",
    "u-turn-left",
    "u-turn-right",
    "figure-eight",
    "accelerate",
    "decelerate",
    "coast",
    "composite",
)


def _clamp(value, low, high):
    return float(np.clip(value, low, high))


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _lateral_amplitude_m(amplitude_deg):
    return abs(float(amplitude_deg)) * 0.12


def _path_to_yaw(lateral, forward, start_yaw=0.0):
    """Heading (degrees) from a local path with +forward as the initial tangent axis."""
    dlat = np.gradient(np.asarray(lateral, dtype=float))
    dfwd = np.gradient(np.asarray(forward, dtype=float))
    heading = np.rad2deg(np.arctan2(dlat, dfwd))
    continuous = np.rad2deg(np.unwrap(np.deg2rad(heading)))
    return continuous - continuous[0] + float(start_yaw)


def _path_maneuver(pattern, frames, rng, start_yaw, progress, smooth):
    """Build yaw from a path whose lateral/forward coordinates share the same metre scale."""
    forward_length_m = float(rng.uniform(22.0, 32.0))
    forward = progress * forward_length_m

    if pattern == "lane-change":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        lane_offset_m = float(rng.uniform(1.6, 2.4))
        lateral = sign * lane_offset_m * smooth
        return _path_to_yaw(lateral, forward, start_yaw)

    if pattern == "avoidance":
        sign = 1.0 if rng.random() < 0.5 else -1.0
        swerve_m = float(rng.uniform(1.0, 2.0))
        lateral = sign * swerve_m * np.sin(np.pi * progress)
        return _path_to_yaw(lateral, forward, start_yaw)

    if pattern == "zigzag":
        n_cycles = int(rng.randint(3, 5))
        cone_lateral_m = float(rng.uniform(1.0, 1.8))
        # sin(πp) envelope: enter/exit straight, lateral returns to centre at the end.
        lateral = (
            cone_lateral_m
            * np.sin(2.0 * np.pi * n_cycles * progress)
            * np.sin(np.pi * progress)
        )
        return _path_to_yaw(lateral, forward, start_yaw)

    raise ValueError(f"Not a path maneuver: {pattern}")


def _segment_yaw(pattern, frames, rng, start_yaw=0.0, amplitude_deg=None):
    frames = max(1, int(frames))
    progress = np.linspace(0.0, 1.0, frames)
    smooth = np.asarray([_smoothstep(value) for value in progress], dtype=float)
    if amplitude_deg is None:
        amplitude_deg = float(rng.uniform(10.0, 35.0))

    if pattern == "straight":
        return np.full(frames, start_yaw, dtype=float)
    if pattern == "circle-left":
        return start_yaw + np.linspace(0.0, -360.0, frames)
    if pattern == "circle-right":
        return start_yaw + np.linspace(0.0, 360.0, frames)
    if pattern == "soft-turn-left":
        return start_yaw - abs(amplitude_deg) * smooth
    if pattern == "soft-turn-right":
        return start_yaw + abs(amplitude_deg) * smooth
    if pattern == "hard-turn-left":
        return start_yaw - np.linspace(0.0, max(abs(amplitude_deg), 90.0), frames)
    if pattern == "hard-turn-right":
        return start_yaw + np.linspace(0.0, max(abs(amplitude_deg), 90.0), frames)
    if pattern == "u-turn-left":
        return start_yaw - 180.0 * smooth
    if pattern == "u-turn-right":
        return start_yaw + 180.0 * smooth
    if pattern == "lane-change":
        return _path_maneuver(pattern, frames, rng, start_yaw, progress, smooth)
    if pattern == "avoidance":
        return _path_maneuver(pattern, frames, rng, start_yaw, progress, smooth)
    if pattern == "zigzag":
        return _path_maneuver(pattern, frames, rng, start_yaw, progress, smooth)
    if pattern == "figure-eight":
        amplitude_m = _lateral_amplitude_m(amplitude_deg)
        t = np.linspace(0.0, 2.0 * np.pi, frames)
        lat_raw = amplitude_m * np.sin(t)
        fwd_raw = amplitude_m * np.sin(t) * np.cos(t)
        heading0 = np.arctan2(amplitude_m * np.cos(t[0]), amplitude_m * np.cos(2.0 * t[0]))
        cos_r = float(np.cos(-heading0))
        sin_r = float(np.sin(-heading0))
        lateral = lat_raw * cos_r - fwd_raw * sin_r
        forward = lat_raw * sin_r + fwd_raw * cos_r
        return _path_to_yaw(lateral, forward, start_yaw)
    if pattern in {"accelerate", "decelerate", "coast"}:
        return start_yaw + 0.25 * abs(amplitude_deg) * np.sin(2.0 * np.pi * progress)
    raise ValueError(f"Unknown angle pattern: {pattern}")


def _speed_segment(pattern, frames, rng, start_speed, min_speed, max_speed):
    frames = max(1, int(frames))
    progress = np.linspace(0.0, 1.0, frames)
    smooth = np.asarray([_smoothstep(value) for value in progress], dtype=float)
    target = float(rng.uniform(min_speed, max_speed))
    if pattern == "accelerate":
        target = max(target, start_speed + rng.uniform(0.5, 2.5))
    elif pattern == "decelerate":
        target = min(target, max(min_speed, start_speed * rng.uniform(0.35, 0.75)))
    elif pattern == "coast":
        target = max(min_speed, start_speed * rng.uniform(0.55, 0.90))
    return start_speed + (target - start_speed) * smooth


def _stabilize_angle_profile(angle_array_deg, physics_hz, max_yaw_rate_deg_s=40.0, smooth_window=9):
    angles = np.asarray(angle_array_deg, dtype=float).copy()
    if angles.size <= 1:
        return angles
    continuous = np.rad2deg(np.unwrap(np.deg2rad(angles)))
    if smooth_window > 1 and continuous.size >= smooth_window:
        kernel = np.ones(int(smooth_window), dtype=float)
        kernel /= float(np.sum(kernel))
        continuous = np.convolve(continuous, kernel, mode="same")
    max_step = float(max_yaw_rate_deg_s) / float(max(1, physics_hz))
    limited = np.empty_like(continuous)
    limited[0] = continuous[0]
    for idx in range(1, continuous.size):
        delta = continuous[idx] - limited[idx - 1]
        delta = _clamp(delta, -max_step, max_step)
        limited[idx] = limited[idx - 1] + delta
    return limited


def random_from_numpy(rng):
    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))
    return py_rng


def build_angle_and_velocity_profiles(
    *,
    pattern,
    physics_frames,
    seed,
    min_target_velocity,
    max_target_velocity,
    segment_min_seconds,
    segment_max_seconds,
    physics_hz,
    composite_profile,
):
    rng = np.random.default_rng(seed)
    py_rng = random_from_numpy(rng)
    start_yaw = float(rng.uniform(-180.0, 180.0))
    start_speed = float(rng.uniform(min_target_velocity, max_target_velocity))

    if pattern != "composite":
        yaw = _segment_yaw(pattern, physics_frames, py_rng, start_yaw=start_yaw)
        yaw = _stabilize_angle_profile(yaw, physics_hz)
        speed = _speed_segment(
            pattern, physics_frames, py_rng, start_speed, min_target_velocity, max_target_velocity
        )
        return yaw, speed, [pattern]

    if composite_profile == "stable":
        choices = [
            "straight",
            "circle-left",
            "circle-right",
            "zigzag",
            "avoidance",
            "lane-change",
            "soft-turn-left",
            "soft-turn-right",
            "figure-eight",
            "accelerate",
            "decelerate",
            "coast",
        ]
    else:
        choices = [item for item in ANGLE_PATTERNS if item not in {"composite"}]
    yaw_parts = []
    speed_parts = []
    used = []
    remaining = int(physics_frames)
    current_yaw = start_yaw
    current_speed = start_speed
    while remaining > 0:
        segment_seconds = float(rng.uniform(segment_min_seconds, segment_max_seconds))
        segment_frames = max(1, min(remaining, int(round(segment_seconds * physics_hz))))
        segment_pattern = str(rng.choice(choices))
        segment_yaw = _segment_yaw(segment_pattern, segment_frames, py_rng, start_yaw=current_yaw)
        segment_speed = _speed_segment(
            segment_pattern,
            segment_frames,
            py_rng,
            current_speed,
            min_target_velocity,
            max_target_velocity,
        )
        yaw_parts.append(segment_yaw)
        speed_parts.append(segment_speed)
        used.append(segment_pattern)
        current_yaw = float(segment_yaw[-1])
        current_speed = float(segment_speed[-1])
        remaining -= segment_frames
    yaw_all = np.concatenate(yaw_parts)
    yaw_all = _stabilize_angle_profile(yaw_all, physics_hz)
    speed_all = np.concatenate(speed_parts)
    return yaw_all, speed_all, used
