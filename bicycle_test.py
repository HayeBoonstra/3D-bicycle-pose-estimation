import os

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

def controller(model, data):
    velocity_controller(model, data)

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
    force_local = np.array([0, Kp_turn * delta_turn, 0])
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
    data.ctrl[1] = -roll_angle * Kp1 - Kp2 * (data.qpos[8])

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

def velocity_controller(model, data):
    target_velocity_kmh = 15
    target_velocity = target_velocity_kmh / 3.6
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
    Kp = 50
    data.ctrl[0] = Kp * error

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
    rear_wheel_angle = _qpos(8)
    steer_angle = _qpos(9)
    front_wheel_angle = _qpos(10)
    crank_angle = _qpos(11)
    left_pedal_angle = _qpos(12)
    right_pedal_angle = _qpos(13)
    return tx, ty, tz, rw, rx, ry, rz, rear_wheel_angle, steer_angle, front_wheel_angle, crank_angle, left_pedal_angle, right_pedal_angle

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.set_mjcb_control(controller)

# Time control: physics at 200Hz, display at 60Hz
PHYSICS_HZ = 200
DISPLAY_HZ = 60
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
data.qvel[0] = 3
angle_array = np.concatenate((
    np.zeros(200),
    np.linspace(0,30, 100),
    np.linspace(30, -30, 200),
    np.linspace(-30, 0, 100),
))

# circular_path = np.linspace(0, 2 * np.pi, 2000)
# angle_array = np.rad2deg(circular_path)
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
    transform_data["rear_wheel_angle"].append(rear_wheel_angle)
    transform_data["steer_angle"].append(steer_angle)
    transform_data["front_wheel_angle"].append(front_wheel_angle)
    transform_data["crank_angle"].append(crank_angle)
    transform_data["left_pedal_angle"].append(left_pedal_angle)
    transform_data["right_pedal_angle"].append(right_pedal_angle)

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
    
viewer_mode = False
if viewer_mode:
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
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
            save_transform_data(model, data, i, transform_data)
            i += 1
    
else:
    while i < len(angle_array):
        mujoco.mj_step(model, data)
        steering_angle_controller(model, data, angle_array, i, plot_data)
        save_transform_data(model, data, i, transform_data)
        i += 1


transform_data = resize_transform_data(transform_data, DISPLAY_HZ)
save_transform_data_csv(transform_data, f"transform_data_{DISPLAY_HZ}hz.csv")
plot_mode = False
if plot_mode:
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