import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from bicycle_constructor import Bicycle
from humanoid_constructor import Humanoid

# Regenerate bicycle.xml and humanoid.xml from constructors (so they include sites) before loading world
def regenerate_models():
    bicycle = Bicycle()
    bicycle.create_bicycle_variables()
    bicycle.save_bicycle_model("bicycle.xml")
    humanoid = Humanoid()
    humanoid.save_humanoid_model("humanoid.xml")

def get_qpos_layout(model):
    """Return (bicycle_nq, humanoid_qpos_start, humanoid_nq) for the merged model."""
    body_torso_id = model.body("torso").id
    # First joint of bicycle is at qpos 0; first joint of humanoid gives humanoid_qpos_start
    humanoid_qpos_start = model.jnt_qposadr[model.body_jntadr[body_torso_id]]
    bicycle_nq = humanoid_qpos_start
    humanoid_nq = model.nq - bicycle_nq
    return bicycle_nq, humanoid_qpos_start, humanoid_nq

# Site name pairs for IK: (humanoid_site, bicycle_site)
ALIGN_SITE_PAIRS = [
    ("pelvis_site", "seat_site"),
    ("left_foot_site", "right_pedal_site"),   # rider left foot on bicycle right pedal (correct spatial match)
    ("right_foot_site", "left_pedal_site"),
    ("left_hand_site", "left_handlebar_site"),
    ("right_hand_site", "right_handlebar_site"),
]

def align_rider_to_bicycle(model, data, bicycle_nq, humanoid_qpos_start, humanoid_nq, maxiter=500):
    """Run IK to align humanoid sites to bicycle sites; only humanoid qpos is optimized."""
    # Fix bicycle at reference pose: upright, pedals at 0
    data.qpos[0:3] = 0, 0, 0.35  # position
    data.qpos[3:7] = 1, 0, 0, 0  # quat (w, x, y, z)
    data.qpos[7:bicycle_nq] = 0  # hinges (steer, pedals, etc.)
    # Get site ids
    site_ids = {}
    for name in ["pelvis_site", "seat_site", "left_foot_site", "left_pedal_site",
                 "right_foot_site", "right_pedal_site", "left_hand_site", "left_handlebar_site",
                 "right_hand_site", "right_handlebar_site"]:
        site_ids[name] = model.site(name).id
    humanoid_site_ids = [site_ids[h] for h, _ in ALIGN_SITE_PAIRS]
    bicycle_site_ids = [site_ids[b] for _, b in ALIGN_SITE_PAIRS]

    def cost(x):
        # x is humanoid qpos; normalize quaternion (first 4 of humanoid = indices 3:7 in humanoid slice)
        x = np.array(x, dtype=np.float64)
        q = data.qpos.copy()
        q[humanoid_qpos_start:humanoid_qpos_start + humanoid_nq] = x
        # Normalize humanoid root quaternion (indices 3:7 of humanoid = qpos indices humanoid_qpos_start+3 : +7)
        quat = q[humanoid_qpos_start + 3:humanoid_qpos_start + 7]
        quat /= np.linalg.norm(quat)
        q[humanoid_qpos_start + 3:humanoid_qpos_start + 7] = quat
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        err = 0.0
        for hi, bi in zip(humanoid_site_ids, bicycle_site_ids):
            err += np.sum((data.site_xpos[hi] - data.site_xpos[bi]) ** 2)
        return err

    # Bounds for humanoid qpos: free joint (pos large, quat normalized in cost), then hinge ranges
    # Collect (qposadr, jtype, range) for joints in humanoid qpos range, then sort by qposadr
    joint_info = []
    for j in range(model.njnt):
        qadr = model.jnt_qposadr[j]
        if qadr < humanoid_qpos_start or qadr >= humanoid_qpos_start + humanoid_nq:
            continue
        joint_info.append((qadr, model.jnt_type[j], model.jnt_range[j]))
    joint_info.sort(key=lambda t: t[0])
    bounds = []
    for qadr, jtype, jrange in joint_info:
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            bounds.extend([(-3, 3), (-3, 3), (-3, 3)])  # position
            bounds.extend([(-1.1, 1.1)] * 4)  # quat (will normalize)
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
            bounds.append((float(jrange[0]), float(jrange[1])))

    x0 = data.qpos[humanoid_qpos_start:humanoid_qpos_start + humanoid_nq].copy()
    # Normalize initial quat
    quat = x0[3:7].copy()
    quat /= np.linalg.norm(quat)
    x0[3:7] = quat
    result = minimize(cost, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter))
    # Write solution back
    x_opt = result.x
    x_opt = np.asarray(x_opt, dtype=np.float64)
    quat = x_opt[3:7].copy()
    quat /= np.linalg.norm(quat)
    x_opt[3:7] = quat
    data.qpos[humanoid_qpos_start:humanoid_qpos_start + humanoid_nq] = x_opt
    mujoco.mj_forward(model, data)
    return result.fun

def align_models(model, data):
    bicycle_nq, humanoid_qpos_start, humanoid_nq = get_qpos_layout(model)
    residual = align_rider_to_bicycle(model, data, bicycle_nq, humanoid_qpos_start, humanoid_nq)
    print(f"Alignment residual (sum squared site errors): {residual:.6f}")

def controller(model, data):
    steering_angle_controller(model, data)
    velocity_controller(model, data)

def steering_angle_controller(model, data):
    # Extract the lean/roll angle from data.qpos and apply it to data.ctrl[1] (steering joint)
    Kp1 = 50
    Kp2 = 40

    quat = data.qpos[3:7]
    euler = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
    roll_angle = euler[0]
    data.ctrl[1] = -roll_angle * Kp1 - Kp2 * (data.qpos[8])
    pass

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

# Regenerate bicycle and humanoid XML from constructors (includes sites), then load world
regenerate_models()
model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
mujoco.set_mjcb_control(controller)

# Align rider to bicycle via IK before simulation (welds in world.xml keep them attached)
align_models(model, data)

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

push_impulse = 20 # Ns
force = push_impulse / physics_dt
while True:
    i += 1
    mujoco.mj_step(model, data)
    if i % 1000 == 0 and i > 0:
        data.qfrc_applied[1] = force
        print("Applying force to steer")
        i = 0
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