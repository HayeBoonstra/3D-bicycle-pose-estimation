import os
import json
import math
from pathlib import Path
from world_contructor import World
# Use Qt with Wayland when tkinter is missing (avoids xcb plugin load failure on Linux)
try:
    import tkinter  # noqa: F401
    _use_tk = True
except ImportError:
    _use_tk = False
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import matplotlib
matplotlib.use("TkAgg" if _use_tk else "QtAgg")

import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import glfw
import matplotlib.pyplot as plt
import argparse
import traceback
import sys


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

def controller(model, data):
    velocity_controller(model, data, target_velocity)


def _ctrl_bounds(model, idx, fallback=(-1.0e6, 1.0e6)):
    try:
        if idx < model.actuator_ctrlrange.shape[0]:
            low, high = model.actuator_ctrlrange[idx]
            return float(low), float(high)
    except Exception:
        pass
    return fallback


def _clamp(value, low, high):
    return float(np.clip(value, low, high))


def steering_angle_controller(model, data, angle_array, i, plot_data):
    Kp1 = 40
    Kp2 = 40

    quat = data.qpos[3:7]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # x, y, z, w
    euler = rot.as_euler('xyz')

    yaw_angle = np.rad2deg(euler[2])
    desired_turn_rate = angle_array[i]

    # Shortest angular difference so control doesn't see ±360° jumps at wrap
    # if absolute value is used than the controller will explode at ±180°
    delta_turn = (desired_turn_rate - yaw_angle + 180) % 360 - 180
    # Hard clamp the yaw error command to avoid force spikes that destabilize simulation.
    delta_turn = _clamp(delta_turn, -45.0, 45.0)
    Kp_turn = 1.5

    Ki_turn = 0.2
    try: 
        integral_turn += delta_turn * physics_dt
    except:
        integral_turn = 0

    Kd_turn = 1
    try: 
        delta_turn += Kd_turn * (delta_turn - delta_turn_prev)
    except:
        delta_turn_prev = delta_turn

    delta_turn += Ki_turn * integral_turn
    # Define force in the local/bicycle frame (lateral-Y, no X/Z)
    force_local = np.array([0, Kp_turn * delta_turn, 0], dtype=float)
    force_local[1] = _clamp(force_local[1], -40.0, 40.0)
    # Convert the force to the world frame
    force_world = rot.apply(force_local)
    # Apply the force at the seat site via xfrc_applied (force at point = force at CoM + torque)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "seat_site")
    body_id = model.site_bodyid[site_id]
    site_pos = data.site_xpos[site_id]
    body_com = data.xpos[body_id]
    torque = np.cross(site_pos - body_com, force_world)
    data.xfrc_applied[body_id][:3] = force_world
    data.xfrc_applied[body_id][3:6] = torque

    quat = data.qpos[3:7]
    euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
    roll_angle = euler[0]
    steer_cmd = -roll_angle * Kp1 - Kp2 * (data.qpos[8])
    steer_low, steer_high = _ctrl_bounds(model, 1, fallback=(-1.0, 1.0))
    data.ctrl[1] = _clamp(steer_cmd, steer_low, steer_high)

    if len(plot_data["time"]) < len(angle_array):
        # Unwrap actual yaw for continuous plot (and consistent scale with desired)
        if "_yaw_raw_prev" not in plot_data:
            plot_data["_yaw_raw_prev"] = yaw_angle
            plot_data["_yaw_unwrapped_prev"] = yaw_angle
        delta_yaw = yaw_angle - plot_data["_yaw_raw_prev"]
        if delta_yaw > 180:
            delta_yaw -= 360
        elif delta_yaw < -180:
            delta_yaw += 360
        yaw_unwrapped = plot_data["_yaw_unwrapped_prev"] + delta_yaw
        plot_data["_yaw_raw_prev"] = yaw_angle
        plot_data["_yaw_unwrapped_prev"] = yaw_unwrapped

        plot_data["time"].append(i * physics_dt)
        plot_data["desired_yaw_angle"].append(desired_turn_rate)
        plot_data["actual_yaw_angle"].append(yaw_unwrapped)
        plot_data["applied_force"].append(np.linalg.norm(force_world))

        plot_data["global_position"].append(np.array(data.qpos[0:3]))

def velocity_controller(model, data, target_velocity):
    # Compute the local forward (bicycle body X) velocity
    # Get the orientation quaternion of the freejoint (qpos[3:7])
    quat = data.qpos[3:7]
    # Convert quaternion to rotation matrix
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # x, y, z, w
    # Get global linear velocity in world frame
    vel_world = data.qvel[0:3]
    # Transform world velocity to body frame
    vel_body = rot.inv().apply(vel_world)
    current_velocity = vel_body[0]  # forward (body X) velocity
    error =  target_velocity - current_velocity
    Kp = 35
    drive_cmd = Kp * error
    drive_low, drive_high = _ctrl_bounds(model, 0, fallback=(-1.0, 1.0))
    data.ctrl[0] = _clamp(drive_cmd, drive_low, drive_high)


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _segment_yaw(pattern, frames, rng, start_yaw=0.0, amplitude_deg=None):
    frames = max(1, int(frames))
    progress = np.linspace(0.0, 1.0, frames)
    smooth = np.asarray([_smoothstep(value) for value in progress], dtype=float)
    if amplitude_deg is None:
        amplitude_deg = float(rng.uniform(10.0, 35.0))

    if pattern == "straight":
        return np.full(frames, start_yaw, dtype=float)
    if pattern == "circle-left":
        return start_yaw + np.linspace(0.0, abs(amplitude_deg), frames)
    if pattern == "circle-right":
        return start_yaw - np.linspace(0.0, abs(amplitude_deg), frames)
    if pattern == "soft-turn-left":
        return start_yaw + abs(amplitude_deg) * smooth
    if pattern == "soft-turn-right":
        return start_yaw - abs(amplitude_deg) * smooth
    if pattern == "hard-turn-left":
        return start_yaw + np.linspace(0.0, max(abs(amplitude_deg), 90.0), frames)
    if pattern == "hard-turn-right":
        return start_yaw - np.linspace(0.0, max(abs(amplitude_deg), 90.0), frames)
    if pattern == "u-turn-left":
        return start_yaw + 180.0 * smooth
    if pattern == "u-turn-right":
        return start_yaw - 180.0 * smooth
    if pattern == "zigzag":
        cycles = float(rng.uniform(1.0, 3.0))
        return start_yaw + abs(amplitude_deg) * np.sin(2.0 * np.pi * cycles * progress)
    if pattern in {"avoidance", "lane-change"}:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        return start_yaw + sign * abs(amplitude_deg) * np.sin(2.0 * np.pi * progress)
    if pattern == "figure-eight":
        return start_yaw + abs(amplitude_deg) * np.sin(4.0 * np.pi * progress)
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
    # Convert wrapped angles to continuous heading space, smooth, then re-limit slew rate.
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
        speed = _speed_segment(pattern, physics_frames, py_rng, start_speed, min_target_velocity, max_target_velocity)
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


def random_from_numpy(rng):
    import random

    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)))
    return py_rng

def extract_transform_data(model, data):
    def _qpos(idx, default=np.nan):
        return float(data.qpos[idx]) if idx < data.qpos.shape[0] else float(default)

    tx = data.qpos[0]
    ty = data.qpos[1]
    tz = data.qpos[2]
    quat = np.array([_qpos(3), _qpos(4), _qpos(5), _qpos(6)], dtype=float)  # MuJoCo freejoint: w, x, y, z
    rw = quat[0]  # stored as w
    rx = quat[1]  # stored as x
    ry = quat[2]  # stored as y
    rz = quat[3]  # stored as z
    rear_wheel_angle = _qpos(7)
    steer_angle = _qpos(8)
    front_wheel_angle = _qpos(9)
    crank_angle = _qpos(10)
    left_pedal_angle = _qpos(11)
    right_pedal_angle = _qpos(12)
    return tx, ty, tz, rw, rx, ry, rz, rear_wheel_angle, steer_angle, front_wheel_angle, crank_angle, left_pedal_angle, right_pedal_angle


parser = argparse.ArgumentParser(description="Bicycle simulation launch options")
parser.add_argument("--plotting", dest="plotting", action="store_true", help="Enable plotting")
parser.set_defaults(plotting=False)
parser.add_argument("--viewer", dest="viewer", action="store_true", help="Enable viewer")
parser.set_defaults(viewer=False)
parser.add_argument("--filename", type=str, default="transform_data", help="Base name of the output csv file")
parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for exported trajectory CSVs")
parser.add_argument("--output-path", type=Path, help="Exact output CSV path. Overrides --output-dir/--filename suffixing.")
parser.add_argument("--pattern", choices=ANGLE_PATTERNS, default="composite", help="Desired-yaw angle-array pattern")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument(
    "--trajectory-frames",
    type=int,
    default=2187,
    help="Target number of exported frames at DISPLAY_HZ. Physics steps are scaled from this.",
)
parser.add_argument("--physics-hz", type=int, default=200)
parser.add_argument("--display-hz", type=int, default=60)
parser.add_argument("--target-velocity-mps", type=float, default=None, help="Fixed target velocity in m/s")
parser.add_argument("--min-target-velocity-mps", type=float, default=2.0)
parser.add_argument("--max-target-velocity-mps", type=float, default=5.0)
parser.add_argument("--max-yaw-rate-deg-s", type=float, default=40.0)
parser.add_argument("--segment-min-seconds", type=float, default=2.0)
parser.add_argument("--segment-max-seconds", type=float, default=7.0)
parser.add_argument(
    "--composite-profile",
    choices=("stable", "full"),
    default="stable",
    help="stable excludes hard-turn/u-turn segments to reduce fall-overs; full keeps all patterns.",
)
parser.add_argument(
    "--max-roll-deg",
    type=float,
    default=45.0,
    help="Reject trajectory if absolute bicycle roll exceeds this (after resampling).",
)
args = parser.parse_args()
launch_options = {
    "plotting": args.plotting,
    "viewer": args.viewer,
    "filename": args.filename
}

world = World()
model_xml = world.create_world()
world.save_world("world.xml")  # save the world XML file

model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# Time control: physics simulation is resampled to display/export FPS.
PHYSICS_HZ = int(args.physics_hz)
DISPLAY_HZ = int(args.display_hz)
if PHYSICS_HZ < 1 or DISPLAY_HZ < 1:
    raise ValueError("--physics-hz and --display-hz must be >= 1")
if args.trajectory_frames < 2:
    raise ValueError("--trajectory-frames must be >= 2")
if args.segment_min_seconds <= 0 or args.segment_max_seconds < args.segment_min_seconds:
    raise ValueError("segment duration bounds are invalid")
if args.min_target_velocity_mps > args.max_target_velocity_mps:
    raise ValueError("--min-target-velocity-mps must be <= --max-target-velocity-mps")
physics_dt = 1.0 / PHYSICS_HZ
display_dt = 1.0 / DISPLAY_HZ
model.opt.timestep = physics_dt
def on_key(keycode):
    global space_pressed
    if keycode == glfw.KEY_SPACE:
        space_pressed = True
        print("Space pressed")

def update_camera(viewer, data):
    cam = viewer.cam
    cam.lookat[:] = data.qpos[0:3]

i = 0
physics_frames = int(math.ceil(args.trajectory_frames * PHYSICS_HZ / DISPLAY_HZ))
angle_array, velocity_profile, pattern_segments = build_angle_and_velocity_profiles(
    pattern=args.pattern,
    physics_frames=physics_frames,
    seed=args.seed,
    min_target_velocity=args.min_target_velocity_mps,
    max_target_velocity=args.max_target_velocity_mps,
    segment_min_seconds=args.segment_min_seconds,
    segment_max_seconds=args.segment_max_seconds,
    physics_hz=PHYSICS_HZ,
    composite_profile=args.composite_profile,
)
angle_array = _stabilize_angle_profile(angle_array, PHYSICS_HZ, max_yaw_rate_deg_s=args.max_yaw_rate_deg_s)
if args.target_velocity_mps is not None:
    velocity_profile = np.full_like(angle_array, float(args.target_velocity_mps), dtype=float)
target_velocity = float(velocity_profile[0])
data.qvel[0] = target_velocity

next_display_time = time.perf_counter()
next_physics_time = time.perf_counter()
push_impulse = 20 # Ns
space_pressed = False
force = push_impulse / physics_dt

plot_data = {
    "time": [],
    "desired_yaw_angle": [],
    "actual_yaw_angle": [],
    "applied_force": [],
    "global_position": [],
}

transform_data = {
    "time": [],
    "tx": [],
    "ty": [],
    "tz": [],
    "rw": [],
    "rx": [],
    "ry": [],
    "rz": [],
    "rear_wheel_angle": [],
    "steer_angle": [],
    "front_wheel_angle": [],
    "crank_angle": [],
    "left_pedal_angle": [],
    "right_pedal_angle": [],
}

def save_transform_data(model, data, i, transform_data):
    tx, ty, tz, rw, rx, ry, rz, rear_wheel_angle, steer_angle, front_wheel_angle, crank_angle, left_pedal_angle, right_pedal_angle = extract_transform_data(model, data)
    # Use a monotonically increasing capture time (i can wrap for controllers).
    transform_data["time"].append(len(transform_data["time"]) * physics_dt)
    transform_data["tx"].append(tx)
    transform_data["ty"].append(ty)
    transform_data["tz"].append(tz)
    transform_data["rw"].append(rw)
    transform_data["rx"].append(rx)
    transform_data["ry"].append(ry)
    transform_data["rz"].append(rz)
    for key, rad in (
        ("rear_wheel_angle", rear_wheel_angle),
        ("steer_angle", steer_angle),
        ("front_wheel_angle", front_wheel_angle),
        ("crank_angle", crank_angle),
        ("left_pedal_angle", left_pedal_angle),
        ("right_pedal_angle", right_pedal_angle),
    ):
        transform_data[key].append(float(np.rad2deg(rad)))

def resize_transform_data(transform_data, target_frame_rate):
    actual_frame_rate = 1/physics_dt
    if actual_frame_rate == target_frame_rate:
        print("Actual frame rate is equal to target frame rate")
        return transform_data
    
    t_in = np.asarray(transform_data.get("time", []), dtype=float)
    if t_in.size < 2:
        return transform_data

    duration = t_in[-1] - t_in[0]
    if duration <= 0:
        return transform_data

    n_out = int(round(duration * float(target_frame_rate)))
    n_out = max(2, n_out)
    t_out = np.linspace(t_in[0], t_in[-1], n_out, dtype=float)

    transform_data["time"] = t_out.tolist()

    quat_keys = ("rx", "ry", "rz", "rw")  # already x, y, z, w (SciPy order) from extract
    if all(k in transform_data for k in quat_keys) and all(len(transform_data[k]) == t_in.shape[0] for k in quat_keys):
        q_xyzw = np.stack([np.asarray(transform_data[k], dtype=float) for k in quat_keys], axis=1)
        q_xyzw = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
        rots = R.from_quat(q_xyzw)
        slerp = Slerp(t_in, rots)
        q_out_xyzw = slerp(t_out).as_quat()  # x, y, z, w
        transform_data["rx"] = q_out_xyzw[:, 0].tolist()
        transform_data["ry"] = q_out_xyzw[:, 1].tolist()
        transform_data["rz"] = q_out_xyzw[:, 2].tolist()
        transform_data["rw"] = q_out_xyzw[:, 3].tolist()

    for k, v in list(transform_data.items()):
        if k == "time":
            continue
        if k in quat_keys:
            continue
        y_in = np.asarray(v, dtype=float)
        if y_in.size == 0:
            transform_data[k] = []
            continue
        if y_in.shape[0] != t_in.shape[0]:
            raise ValueError(
                f"transform_data['{k}'] length {y_in.shape[0]} does not match time length {t_in.shape[0]}"
            )

        y2 = y_in.reshape(y_in.shape[0], -1)
        y2_out = np.empty((t_out.shape[0], y2.shape[1]), dtype=float)
        for col in range(y2.shape[1]):
            y2_out[:, col] = np.interp(t_out, t_in, y2[:, col])
        y_out = y2_out.reshape((t_out.shape[0],) + y_in.shape[1:])
        transform_data[k] = y_out.tolist()
    return transform_data

def save_transform_data_csv(transform_data, csv_path):
    import csv

    keys = [k for k in transform_data.keys() if k != "time"]
    fieldnames = ["time"] + keys

    n = len(transform_data.get("time", []))
    for k in keys:
        if len(transform_data.get(k, [])) != n:
            raise ValueError(f"Column '{k}' length {len(transform_data.get(k, []))} does not match time length {n}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            row = {"time": transform_data["time"][i]}
            for k in keys:
                row[k] = transform_data[k][i]
            w.writerow(row)

def align_model_to_ground(
    model,
    data,
    height_offsets=None,
    z_index=2,
    vertical_force_dof_index=2,
    verbose=True,
):
    """
    Sweep a small Z offset and choose the value that makes the vertical generalized
    force (Fz in freejoint coordinates) closest to zero.

    This is a heuristic to avoid the bicycle "jumping down" at initialization.
    """
    if height_offsets is None:
        height_offsets = np.linspace(-0.1, 0.1, 2001)

    height_offsets = np.asarray(height_offsets, dtype=float)
    if height_offsets.ndim != 1:
        raise ValueError("height_offsets must be a 1D array-like")

    # In a MuJoCo freejoint, qpos[z_index] is the body's Z position (x,y,z,qw,qx,qy,qz).
    # qfrc_inverse for the freejoint generalized forces is ordered [Fx,Fy,Fz,Twx,Twy,Twz,...]
    # so index `vertical_force_dof_index` corresponds to Fz.
    if z_index >= data.qpos.shape[0]:
        raise IndexError(f"z_index={z_index} out of bounds for data.qpos shape {data.qpos.shape}")
    if vertical_force_dof_index >= data.qfrc_inverse.shape[0]:
        raise IndexError(
            "vertical_force_dof_index="
            f"{vertical_force_dof_index} out of bounds for data.qfrc_inverse shape {data.qfrc_inverse.shape}"
        )

    base_qpos = data.qpos.copy()
    base_qvel = data.qvel.copy()
    base_ctrl = data.ctrl.copy() if getattr(data, "ctrl", None) is not None else None

    # Use static inverse dynamics: zero velocities/acceleration and clear externally
    # applied forces so the sweep isn't biased by controllers.
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    if getattr(data, "ctrl", None) is not None:
        data.ctrl[:] = 0.0
    if getattr(data, "xfrc_applied", None) is not None:
        data.xfrc_applied[:] = 0.0

    mujoco.mj_forward(model, data)

    vertical_forces = np.empty(height_offsets.shape[0], dtype=float)
    for k, offset in enumerate(height_offsets):
        data.qpos[:] = base_qpos
        data.qpos[z_index] += float(offset)

        # Keep evaluation "static" regardless of what was in the global state.
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        if getattr(data, "ctrl", None) is not None:
            data.ctrl[:] = 0.0
        if getattr(data, "xfrc_applied", None) is not None:
            data.xfrc_applied[:] = 0.0

        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)
        fz = float(data.qfrc_inverse[vertical_force_dof_index])
        if not np.isfinite(fz):
            fz = np.inf
        vertical_forces[k] = fz

    best_idx = int(np.argmin(np.abs(vertical_forces)))
    best_offset = float(height_offsets[best_idx]) + 0.003

    # Apply the chosen offset and restore the original simulation velocities/control.
    data.qpos[:] = base_qpos
    data.qpos[z_index] += best_offset
    data.qvel[:] = base_qvel
    data.qacc[:] = 0.0
    if getattr(data, "ctrl", None) is not None and base_ctrl is not None:
        data.ctrl[:] = base_ctrl
    if getattr(data, "xfrc_applied", None) is not None:
        data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)

    if verbose:
        print(
            "align_model_to_ground: "
            f"best_offset={best_offset:.6g}, min|Fz|={abs(vertical_forces[best_idx]):.6g}"
        )
    return best_offset


# Calibrate once before the simulation starts.
align_model_to_ground(model, data)


if launch_options["viewer"]:
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    try:
        while viewer.is_running() and not space_pressed and i < len(angle_array):
            mujoco.mj_step(model, data)
            # Sync viewer and camera at 60Hz only
            now = time.perf_counter()
            if now >= next_display_time:
                viewer.sync()
                update_camera(viewer, data)
                next_display_time += display_dt
                if next_display_time < now:
                    next_display_time = now + display_dt

            # Sleep only the remaining time until next physics tick (accounts for computation time)
            now = time.perf_counter()
            sleep_time = next_physics_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_physics_time += physics_dt
            if next_physics_time < time.perf_counter():
                next_physics_time = time.perf_counter()
            steering_angle_controller(model, data, angle_array, i, plot_data)
            velocity_controller(model, data, float(velocity_profile[i]))
            save_transform_data(model, data, i, transform_data)
            i += 1
    except:
        print("error in viewer mode")
        traceback.print_exc()
    finally:
        viewer.close()
    
else:
    while i < len(angle_array):
        mujoco.mj_step(model, data)
        steering_angle_controller(model, data, angle_array, i, plot_data)
        velocity_controller(model, data, float(velocity_profile[i]))
        save_transform_data(model, data, i, transform_data)
        i += 1


transform_data = resize_transform_data(transform_data, DISPLAY_HZ)
if args.output_path is not None:
    output_csv = args.output_path
else:
    output_csv = args.output_dir / f"{launch_options['filename']}_{DISPLAY_HZ}hz.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)
save_transform_data_csv(transform_data, output_csv)

q_wxyz = np.stack(
    [
        np.asarray(transform_data.get("rw", []), dtype=float),
        np.asarray(transform_data.get("rx", []), dtype=float),
        np.asarray(transform_data.get("ry", []), dtype=float),
        np.asarray(transform_data.get("rz", []), dtype=float),
    ],
    axis=1,
)
if q_wxyz.size == 0:
    raise RuntimeError("No quaternion data generated in trajectory export.")
q_xyzw = np.stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], axis=1)
roll_deg = np.rad2deg(R.from_quat(q_xyzw).as_euler("xyz", degrees=False)[:, 0])
max_abs_roll_deg = float(np.max(np.abs(roll_deg)))
if max_abs_roll_deg > float(args.max_roll_deg):
    print(
        f"Rejected trajectory: max |roll|={max_abs_roll_deg:.2f}deg exceeds "
        f"--max-roll-deg={float(args.max_roll_deg):.2f}deg"
    )
    sys.exit(3)

manifest_path = output_csv.with_suffix(".json")
manifest_path.write_text(
    json.dumps(
        {
            "trajectory_csv": str(output_csv),
            "pattern": args.pattern,
            "pattern_segments": pattern_segments,
            "seed": args.seed,
            "physics_hz": PHYSICS_HZ,
            "display_hz": DISPLAY_HZ,
            "requested_display_frames": args.trajectory_frames,
            "exported_display_frames": len(transform_data.get("time", [])),
            "physics_frames": int(len(angle_array)),
            "target_velocity_mps": args.target_velocity_mps,
            "min_target_velocity_mps": args.min_target_velocity_mps,
            "max_target_velocity_mps": args.max_target_velocity_mps,
            "max_abs_roll_deg": max_abs_roll_deg,
            "max_roll_deg_limit": float(args.max_roll_deg),
            "composite_profile": args.composite_profile,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"Wrote trajectory CSV: {output_csv}")
if launch_options["plotting"]:
    plt.subplot(2, 1, 1)
    plt.plot(plot_data["time"], plot_data["desired_yaw_angle"], label="Desired Yaw Angle")
    plt.plot(plot_data["time"], plot_data["actual_yaw_angle"], label="Actual Yaw Angle")
    plt.plot(plot_data["time"], plot_data["applied_force"], label="Applied Force")
    plt.legend()
    plt.subplot(2, 1, 2)
    global_positions = np.array(plot_data["global_position"])
    if global_positions.shape[0] > 0 and global_positions.shape[1] >= 2:
        plt.plot(global_positions[:, 0], global_positions[:, 1], label="Global Position")
    plt.legend()
    plt.show() 