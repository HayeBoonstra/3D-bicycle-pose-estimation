import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def controller(model, data):
    steering_angle_controller(model, data)
    velocity_controller(model, data)

def steering_angle_controller(model, data):
    # Extract the lean/roll angle from data.qpos and apply it to data.ctrl[1]
    # Assuming roll angle is data.qpos[2] (for a typical freejoint: [x, y, z, qw, qx, qy, qz])
    # For a mujoco freejoint, orientation is quaternion, need to extract roll from quaternion (qw, qx, qy, qz = qpos[3:7])

    Kp1 = 50
    Kp2 = 40

    # Get quaternion from qpos[3:7]
    quat = data.qpos[3:7]
    # Convert quaternion to Euler angles (in radians)
    euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
    roll_angle = euler[0]
    data.ctrl[1] = -roll_angle * Kp1 - Kp2 * (data.qpos[8])
    pass

def velocity_controller(model, data):
    target_velocity = 3
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
    Kp = 100
    data.ctrl[0] = Kp * error

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.set_mjcb_control(controller)

viewer = mujoco.viewer.launch_passive(model, data)
def update_camera(viewer, data):
    cam = viewer.cam
    cam.lookat[:] = data.qpos[0:3]

i = 0
data.qvel[0] = 0
while True:
    i += 1
    mujoco.mj_step(model, data)
    viewer.sync()
    if i%1000 == 0 and i > 0:
        data.qfrc_applied[1] = 1000
        print("Applying force to steer")
        i = 0
    else:
        data.qfrc_applied[1] = 0
    update_camera(viewer, data)
    time.sleep(0.01)