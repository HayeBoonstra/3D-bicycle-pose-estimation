import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def controller(model, data):
    steering_angle_controller(model, data)
    velocity_controller(model, data)

def steering_angle_controller(model, data):
    # Roll stability + Stanley lateral control. Stanley: δ = θe + atan(k*e_fa/vx),
    # with e_fa = cross track error from front axle to path, θe = θ − θp (heading error).
    Kp1 = 0
    Kp2 = 0
    K_steer = 10

    quat = data.qpos[3:7]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = rot.as_euler('xyz')
    roll_angle = euler[0]
    yaw = euler[2]
    steering_angle = data.qpos[8]

    vel_world = data.qvel[0:3]
    vel_body = rot.inv().apply(vel_world)
    vx = vel_body[0]  # forward speed (longitudinal)
    v_safe = max(vx, 0.5)

    # 1. Heading error. Path tangent +X → θp = 0. Use θe = θp − θ so positive error = need to steer right.
    theta_e = yaw

    # 2. Cross track error e_fa: from center of front axle to path (y=0). Front axle uses yaw + steering_angle.
    L_FRONT = 0.62  # m, body to front axle (bicycle.xml)
    y_front_axle = data.qpos[1] + L_FRONT * np.sin(yaw + steering_angle)
    e_fa = y_front_axle  # positive when front axle left of path

    # 3. Stanley: δ = θe + atan(k*e_fa/vx). Soften denominator at low speed for stability.
    k = 0.5
    steer_expect = theta_e + np.arctan2(k * e_fa, v_safe)
    steer_expect = np.clip(steer_expect, -np.deg2rad(35), np.deg2rad(35))

    # 4. Roll stabilization + P to track desired steering (negative gain: stable feedback)
    stability_cmd = -Kp1 * roll_angle - Kp2 * steering_angle
    stanley_cmd = -K_steer * (steer_expect - steering_angle)
    data.ctrl[1] = stability_cmd + stanley_cmd

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
    Kp = 20
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

viewer = mujoco.viewer.launch_passive(model, data)
def update_camera(viewer, data):
    cam = viewer.cam
    cam.lookat[:] = data.qpos[0:3]

i = 0
data.qvel[0] = 0
next_display_time = time.perf_counter()
next_physics_time = time.perf_counter()

push_impulse = 10  # Ns
force = push_impulse / physics_dt
while True:
    i += 1
    mujoco.mj_step(model, data)
    if i == 1000:
        data.qfrc_applied[1] = force
        print("Applying force to steer")
    else:
        data.qfrc_applied[1] = 0

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