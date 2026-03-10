import mujoco
import mujoco.viewer
import time
import numpy as np

def controller(model, data):
    steering_angle_controller(model, data)
    velocity_controller(model, data)

def steering_angle_controller(model, data):
    # Extract the lean/roll angle from data.qpos and apply it to data.ctrl[1]
    # Assuming roll angle is data.qpos[2] (for a typical freejoint: [x, y, z, qw, qx, qy, qz])
    # For a mujoco freejoint, orientation is quaternion, need to extract roll from quaternion (qw, qx, qy, qz = qpos[3:7])
    from scipy.spatial.transform import Rotation as R
    Kp1 = 2
    Kp2 = 0.5

    # Get quaternion from qpos[3:7]
    quat = data.qpos[3:7]
    # Convert quaternion to Euler angles (in radians)
    euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
    roll_angle = euler[0]
    data.ctrl[1] = -roll_angle * Kp1 - Kp2 * (data.qpos[8])
    pass

def velocity_controller(model, data):
    target_velocity = 2.5
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
    Kp = 10
    data.ctrl[0] = Kp * error

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.set_mjcb_control(controller)

viewer = mujoco.viewer.launch_passive(model, data)

t = 0.1
data.qvel[0] = 1
data.qvel[1] = 0.001
while True:
    t += 0.3
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)