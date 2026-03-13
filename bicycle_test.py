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
import glfw
import matplotlib.pyplot as plt

def controller(model, data):
    velocity_controller(model, data)

def steering_angle_controller(model, data, angle_array, i, plot_data):
    # Extract the lean/roll angle from data.qpos and apply it to data.ctrl[1] (steering joint)
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
    Kp_turn = 10
    
    # Define force in the local/bicycle frame (lateral-Y, no X/Z)
    force_local = np.array([0, Kp_turn * delta_turn, 0])
    # Convert the force to the world frame
    force_world = rot.apply(force_local)
    # Apply the force to the freejoint (first 3 are force X/Y/Z)
    data.qfrc_applied[0:3] = force_world

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
        plot_data["applied_force"].append(force_world[1])

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
    np.linspace(0, 90, 600),
    np.repeat(90, 400),
    np.linspace(90, 0, 400),
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

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running() and not space_pressed:
        i += 1
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
        if i >= len(angle_array):
            i = 0
        steering_angle_controller(model, data, angle_array, i, plot_data)
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