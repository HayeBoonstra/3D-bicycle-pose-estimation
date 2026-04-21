# Haye Boonstra, 2026
# Thesis project: 3D pose estimation of bicycle kinematics using neural networks
# A script to rig a Mixamo character's legs with IK in Blender.
#
# The script works for Mixamo characters downloaded via the mixamo website in FBX ASCII format.
#
# The foot bones (LeftFoot/RightFoot) are intentionally left untouched - they will later
# be driven by a pedal-tracking constraint added outside this script.

import bpy
import math
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARMATURE_NAME = "Armature"

LEFT_ANKLE_BONE_KEY = "LeftFoot"
RIGHT_ANKLE_BONE_KEY = "RightFoot"
LEFT_KNEE_BONE_KEY = "LeftLeg"
RIGHT_KNEE_BONE_KEY = "RightLeg"
LEFT_HIP_BONE_KEY = "LeftUpLeg"
RIGHT_HIP_BONE_KEY = "RightUpLeg"

LEFT_ANKLE_CONTROL_BONE_NAME = "left_ankle_control"
RIGHT_ANKLE_CONTROL_BONE_NAME = "right_ankle_control"
LEFT_KNEE_CONTROL_BONE_NAME = "left_knee_control"
RIGHT_KNEE_CONTROL_BONE_NAME = "right_knee_control"

# Auto-scale factors applied to shin/thigh bone lengths.
# The pole is placed far enough in front that a fully flexed knee never ends up
# in front of the pole (which would flip the IK solution to the wrong bend direction).
POLE_FORWARD_THIGH_FACTOR = 1.5
POLE_TAIL_LENGTH_FACTOR = 1.0
ANKLE_TAIL_LENGTH_FACTOR = 1.0  # tail sits one shin-length behind the ankle

CONTROL_BONE_NAMES = (
    LEFT_ANKLE_CONTROL_BONE_NAME,
    RIGHT_ANKLE_CONTROL_BONE_NAME,
    LEFT_KNEE_CONTROL_BONE_NAME,
    RIGHT_KNEE_CONTROL_BONE_NAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_bone_name(armature_data, search_key):
    """Resolve bone name by exact, suffix, then substring match (case-insensitive).

    Handles Mixamo's `mixamorig:LeftFoot` prefix as well as plain `LeftFoot`.
    """
    names = [bone.name for bone in armature_data.bones]
    key_lower = search_key.lower()

    for name in names:
        if name == search_key:
            return name

    suffix_matches = [name for name in names if name.lower().endswith(f":{key_lower}")]
    if suffix_matches:
        return sorted(suffix_matches, key=len)[0]

    contains_matches = [name for name in names if key_lower in name.lower()]
    if contains_matches:
        return sorted(contains_matches, key=len)[0]

    raise KeyError(
        f'Could not find a bone matching "{search_key}" in armature "{armature_data.name}".'
    )


def project_onto_plane(vector, plane_normal):
    """Project vector onto the plane orthogonal to plane_normal."""
    axis = plane_normal.normalized()
    return vector - axis * vector.dot(axis)


def infer_character_up(armature_obj):
    """World +Z expressed in armature local space."""
    up = armature_obj.matrix_world.to_quaternion().inverted() @ Vector((0.0, 0.0, 1.0))
    if up.length < 1e-6:
        return Vector((0.0, 0.0, 1.0))
    return up.normalized()


def infer_character_forward(armature_obj, left_hip_edit, right_hip_edit, up_local):
    """
    Estimate the character's forward direction in armature local space.

    Start from world +Y transformed into armature local space. If that degenerates,
    fall back to the pelvis-right axis crossed with up. Always project onto the
    horizontal plane defined by `up_local` so forward stays horizontal.
    """
    forward = armature_obj.matrix_world.to_quaternion().inverted() @ Vector((0.0, 1.0, 0.0))
    if forward.length < 1e-6:
        pelvis_right = right_hip_edit.head - left_hip_edit.head
        forward = pelvis_right.cross(up_local)
    if forward.length < 1e-6:
        forward = Vector((0.0, 1.0, 0.0))
    forward = project_onto_plane(forward, up_local)
    if forward.length < 1e-6:
        forward = Vector((0.0, 1.0, 0.0))
    return forward.normalized()


def sign_correct_forward(forward, up_local, left_ankle_edit, right_ankle_edit):
    """Flip `forward` if the feet point the opposite way (Mixamo re-imports vary)."""
    left_foot_dir = project_onto_plane(
        left_ankle_edit.tail - left_ankle_edit.head, up_local
    )
    right_foot_dir = project_onto_plane(
        right_ankle_edit.tail - right_ankle_edit.head, up_local
    )
    feet_sum = left_foot_dir + right_foot_dir
    if feet_sum.length > 1e-6 and feet_sum.dot(forward) < 0.0:
        return -forward
    return forward


def remove_bone_if_exists(edit_bones, bone_name):
    """Idempotent cleanup so the script can be re-run without stacking bones."""
    existing = edit_bones.get(bone_name)
    if existing is not None:
        edit_bones.remove(existing)


def create_unparented_bone(edit_bones, name, head, tail):
    """Create a deform-free, unparented bone between head and tail."""
    bone = edit_bones.new(name)
    bone.head = head.copy()
    bone.tail = tail.copy()
    bone.parent = None
    bone.use_deform = False
    bone.use_connect = False
    return bone


def get_or_create_ik_constraint(pose_bone):
    """Reuse an existing IK constraint on `pose_bone`, else create one."""
    for constraint in pose_bone.constraints:
        if constraint.type == 'IK':
            return constraint
    return pose_bone.constraints.new(type='IK')


def signed_angle_around_axis(vector_u, vector_v, axis):
    """Signed angle from vector_u to vector_v around `axis` (right-hand rule)."""
    u = vector_u.normalized()
    v = vector_v.normalized()
    a = axis.normalized()
    return math.atan2(a.dot(u.cross(v)), u.dot(v))


def calibrate_pole_angle_from_bent_pose(
    thigh_pbone,
    shin_pbone,
    ankle_ctrl_pbone,
    knee_ctrl_pbone,
    ik_constraint,
    bend_offset_armature,
    max_iter=10,
    tolerance_rad=1e-4,
    label="leg",
):
    """
    Calibrate `ik_constraint.pole_angle` using a temporarily bent-knee pose.

    Rest-pose-only calibration fails on Mixamo because its legs are nearly straight,
    making the twist around the chain axis underdetermined (+-180 deg ambiguity).
    This function breaks the ambiguity by temporarily translating the ankle control
    upward in armature space (`bend_offset_armature`), forcing the IK solver to
    bend the knee. With a genuinely bent knee, the bend direction is directly
    observable and we can solve pole_angle in closed form per iteration.

    Per iteration:
      1. Force the depsgraph to run the IK solver.
      2. Read solved hip/knee/ankle positions from pose bones.
      3. Project the knee direction and the pole direction onto the plane
         perpendicular to the hip->ankle chain axis.
      4. Signed angle between them (around the chain axis) = needed correction.
      5. Add correction to `pole_angle`. Typically converges in 1 iteration.

    The ankle control is restored to its rest matrix in a `finally` block so a
    crash or early return cannot leave the rig in the bent calibration pose.
    """
    ankle_rest_matrix = ankle_ctrl_pbone.bone.matrix_local.copy()
    bent_matrix = Matrix.Translation(bend_offset_armature) @ ankle_rest_matrix

    ik_constraint.pole_angle = 0.0
    last_err = 0.0

    try:
        ankle_ctrl_pbone.matrix = bent_matrix

        for iteration in range(max_iter):
            bpy.context.view_layer.update()

            hip_pos = thigh_pbone.head.copy()
            knee_pos = shin_pbone.head.copy()  # == thigh.tail on a connected chain
            ankle_pos = shin_pbone.tail.copy()
            pole_pos = knee_ctrl_pbone.head.copy()

            chain = ankle_pos - hip_pos
            if chain.length < 1e-6:
                print(f"[leg-IK] {label}: degenerate chain (hip==ankle); aborting")
                return
            chain_axis = chain.normalized()

            knee_vec = knee_pos - hip_pos
            pole_vec = pole_pos - hip_pos
            knee_proj = knee_vec - chain_axis * knee_vec.dot(chain_axis)
            pole_proj = pole_vec - chain_axis * pole_vec.dot(chain_axis)

            if knee_proj.length < 1e-6:
                print(
                    f"[leg-IK] {label}: knee projection is zero - "
                    "bend offset may be too small to produce a bent knee"
                )
                return
            if pole_proj.length < 1e-6:
                print(f"[leg-IK] {label}: pole lies on chain axis; aborting")
                return

            err = signed_angle_around_axis(knee_proj, pole_proj, chain_axis)
            last_err = err

            if abs(err) < tolerance_rad:
                print(
                    f"[leg-IK] {label}: converged at iter {iteration}, "
                    f"pole_angle = {math.degrees(ik_constraint.pole_angle):+.3f} deg"
                )
                return

            ik_constraint.pole_angle += err

        print(
            f"[leg-IK] {label}: NOT converged after {max_iter} iter, "
            f"pole_angle = {math.degrees(ik_constraint.pole_angle):+.3f} deg, "
            f"residual = {math.degrees(last_err):+.4f} deg"
        )
    finally:
        ankle_ctrl_pbone.matrix = ankle_rest_matrix
        bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
armature_object = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = armature_object
armature_object.select_set(True)

LEFT_ANKLE_BONE_NAME = resolve_bone_name(armature_object.data, LEFT_ANKLE_BONE_KEY)
RIGHT_ANKLE_BONE_NAME = resolve_bone_name(armature_object.data, RIGHT_ANKLE_BONE_KEY)
LEFT_KNEE_BONE_NAME = resolve_bone_name(armature_object.data, LEFT_KNEE_BONE_KEY)
RIGHT_KNEE_BONE_NAME = resolve_bone_name(armature_object.data, RIGHT_KNEE_BONE_KEY)
LEFT_HIP_BONE_NAME = resolve_bone_name(armature_object.data, LEFT_HIP_BONE_KEY)
RIGHT_HIP_BONE_NAME = resolve_bone_name(armature_object.data, RIGHT_HIP_BONE_KEY)


# --- Edit Mode: build the four unparented control bones --------------------
bpy.ops.object.mode_set(mode='EDIT')
edit_bones = armature_object.data.edit_bones

for control_name in CONTROL_BONE_NAMES:
    remove_bone_if_exists(edit_bones, control_name)

left_ankle_bone = edit_bones[LEFT_ANKLE_BONE_NAME]
right_ankle_bone = edit_bones[RIGHT_ANKLE_BONE_NAME]
left_knee_bone = edit_bones[LEFT_KNEE_BONE_NAME]
right_knee_bone = edit_bones[RIGHT_KNEE_BONE_NAME]
left_hip_bone = edit_bones[LEFT_HIP_BONE_NAME]
right_hip_bone = edit_bones[RIGHT_HIP_BONE_NAME]

character_up = infer_character_up(armature_object)
character_forward = infer_character_forward(
    armature_object, left_hip_bone, right_hip_bone, character_up
)
character_forward = sign_correct_forward(
    character_forward, character_up, left_ankle_bone, right_ankle_bone
)

left_shin_length = (left_ankle_bone.head - left_knee_bone.head).length
right_shin_length = (right_ankle_bone.head - right_knee_bone.head).length
left_thigh_length = (left_knee_bone.head - left_hip_bone.head).length
right_thigh_length = (right_knee_bone.head - right_hip_bone.head).length

# Ankle controls: head sits exactly at the ankle bone's head, tail points backwards
# so the bone is clearly oriented backward from the character for easy selection.
left_ankle_head = left_ankle_bone.head.copy()
left_ankle_tail = left_ankle_head - character_forward * (
    left_shin_length * ANKLE_TAIL_LENGTH_FACTOR
)
right_ankle_head = right_ankle_bone.head.copy()
right_ankle_tail = right_ankle_head - character_forward * (
    right_shin_length * ANKLE_TAIL_LENGTH_FACTOR
)

create_unparented_bone(
    edit_bones, LEFT_ANKLE_CONTROL_BONE_NAME, left_ankle_head, left_ankle_tail
)
create_unparented_bone(
    edit_bones, RIGHT_ANKLE_CONTROL_BONE_NAME, right_ankle_head, right_ankle_tail
)

# Knee pole targets: head sits well in front of the knee's head (so even a fully
# flexed knee stays behind the pole), tail points further forward for visibility.
left_knee_pole_head = left_knee_bone.head + character_forward * (
    left_thigh_length * POLE_FORWARD_THIGH_FACTOR
)
left_knee_pole_tail = left_knee_pole_head + character_forward * (
    left_thigh_length * POLE_TAIL_LENGTH_FACTOR
)
right_knee_pole_head = right_knee_bone.head + character_forward * (
    right_thigh_length * POLE_FORWARD_THIGH_FACTOR
)
right_knee_pole_tail = right_knee_pole_head + character_forward * (
    right_thigh_length * POLE_TAIL_LENGTH_FACTOR
)

create_unparented_bone(
    edit_bones, LEFT_KNEE_CONTROL_BONE_NAME, left_knee_pole_head, left_knee_pole_tail
)
create_unparented_bone(
    edit_bones, RIGHT_KNEE_CONTROL_BONE_NAME, right_knee_pole_head, right_knee_pole_tail
)

# Bend offset (armature space) used during pole-angle calibration. About 35% of the
# full leg length along character-up is enough to force a clearly bent knee without
# pushing the IK target out of reach.
BEND_FRACTION = 0.35
left_bend_offset = character_up * (left_shin_length + left_thigh_length) * BEND_FRACTION
right_bend_offset = character_up * (right_shin_length + right_thigh_length) * BEND_FRACTION


# --- Pose Mode: install IK constraints, then calibrate pole_angle on a bent knee ---
bpy.ops.object.mode_set(mode='POSE')

left_knee_pbone = armature_object.pose.bones[LEFT_KNEE_BONE_NAME]
right_knee_pbone = armature_object.pose.bones[RIGHT_KNEE_BONE_NAME]
left_hip_pbone = armature_object.pose.bones[LEFT_HIP_BONE_NAME]
right_hip_pbone = armature_object.pose.bones[RIGHT_HIP_BONE_NAME]
left_ankle_pbone = armature_object.pose.bones[LEFT_ANKLE_BONE_NAME]
right_ankle_pbone = armature_object.pose.bones[RIGHT_ANKLE_BONE_NAME]
left_ankle_ctrl_pbone = armature_object.pose.bones[LEFT_ANKLE_CONTROL_BONE_NAME]
right_ankle_ctrl_pbone = armature_object.pose.bones[RIGHT_ANKLE_CONTROL_BONE_NAME]
left_knee_ctrl_pbone = armature_object.pose.bones[LEFT_KNEE_CONTROL_BONE_NAME]
right_knee_ctrl_pbone = armature_object.pose.bones[RIGHT_KNEE_CONTROL_BONE_NAME]


def configure_leg_ik(shin_pbone, ankle_control_name, knee_control_name):
    """Install/refresh the IK constraint with pole_angle left at 0.
    The correct pole_angle is determined afterwards by bent-knee calibration."""
    ik = get_or_create_ik_constraint(shin_pbone)
    ik.target = armature_object
    ik.subtarget = ankle_control_name
    ik.pole_target = armature_object
    ik.pole_subtarget = knee_control_name
    ik.chain_count = 2
    ik.use_rotation = False
    ik.use_stretch = False
    ik.pole_angle = 0.0
    return ik


left_ik = configure_leg_ik(
    left_knee_pbone, LEFT_ANKLE_CONTROL_BONE_NAME, LEFT_KNEE_CONTROL_BONE_NAME
)
right_ik = configure_leg_ik(
    right_knee_pbone, RIGHT_ANKLE_CONTROL_BONE_NAME, RIGHT_KNEE_CONTROL_BONE_NAME
)

calibrate_pole_angle_from_bent_pose(
    left_hip_pbone,
    left_knee_pbone,
    left_ankle_ctrl_pbone,
    left_knee_ctrl_pbone,
    left_ik,
    left_bend_offset,
    label="left",
)
calibrate_pole_angle_from_bent_pose(
    right_hip_pbone,
    right_knee_pbone,
    right_ankle_ctrl_pbone,
    right_knee_ctrl_pbone,
    right_ik,
    right_bend_offset,
    label="right",
)

## add the IK constraint for the feet but don´t fill them yet
left_foot_ik = get_or_create_ik_constraint(left_ankle_pbone)
right_foot_ik = get_or_create_ik_constraint(right_ankle_pbone)
left_foot_ik.target = armature_object
right_foot_ik.target = armature_object
left_foot_ik.chain_count = 1
right_foot_ik.chain_count = 1

bpy.ops.object.mode_set(mode='EDIT')

# names
LEFT_HAND_BONE_KEY = "LeftHand"
RIGHT_HAND_BONE_KEY = "RightHand"
LEFT_FOREARM_BONE_KEY = "LeftForeArm"
RIGHT_FOREARM_BONE_KEY = "RightForeArm"
LEFT_HAND_CTRL_BONE_KEY = "LeftHandControl"
RIGHT_HAND_CTRL_BONE_KEY = "RightHandControl"
LEFT_ELBOW_CTRL_BONE_KEY = "LeftElbowControl"
RIGHT_ELBOW_CTRL_BONE_KEY = "RightElbowControl"
left_hand_bone_name = resolve_bone_name(armature_object.data, LEFT_HAND_BONE_KEY)
right_hand_bone_name = resolve_bone_name(armature_object.data, RIGHT_HAND_BONE_KEY)
left_forearm_bone_name = resolve_bone_name(armature_object.data, LEFT_FOREARM_BONE_KEY)
right_forearm_bone_name = resolve_bone_name(armature_object.data, RIGHT_FOREARM_BONE_KEY)

left_hand_bone = armature_object.data.edit_bones[left_hand_bone_name]
right_hand_bone = armature_object.data.edit_bones[right_hand_bone_name]
left_forearm_bone = armature_object.data.edit_bones[left_forearm_bone_name]
right_forearm_bone = armature_object.data.edit_bones[right_forearm_bone_name]

# create unparented control bones for the hands
left_hand_ctrl_bone = create_unparented_bone(edit_bones, LEFT_HAND_CTRL_BONE_KEY, left_hand_bone.head, left_hand_bone.head-character_forward*left_hand_bone.length)
right_hand_ctrl_bone = create_unparented_bone(edit_bones, RIGHT_HAND_CTRL_BONE_KEY, right_hand_bone.head, right_hand_bone.head-character_forward*right_hand_bone.length)

# create the pole target bones for the hands
left_elbow_pole_target = create_unparented_bone(edit_bones, LEFT_ELBOW_CTRL_BONE_KEY, left_forearm_bone.head, left_forearm_bone.head-character_forward*left_forearm_bone.length)
right_elbow_pole_target = create_unparented_bone(edit_bones, RIGHT_ELBOW_CTRL_BONE_KEY, right_forearm_bone.head, right_forearm_bone.head-character_forward*right_forearm_bone.length)

bpy.ops.object.mode_set(mode='POSE')
# setup IK constraints for the hands
left_hand_pbone = armature_object.pose.bones[left_hand_bone_name]
right_hand_pbone = armature_object.pose.bones[right_hand_bone_name]
left_forearm_pbone = armature_object.pose.bones[left_forearm_bone_name]
right_forearm_pbone = armature_object.pose.bones[right_forearm_bone_name]
left_hand_ctrl_pbone = armature_object.pose.bones[LEFT_HAND_CTRL_BONE_KEY]
right_hand_ctrl_pbone = armature_object.pose.bones[RIGHT_HAND_CTRL_BONE_KEY]
left_elbow_ctrl_pbone = armature_object.pose.bones[LEFT_ELBOW_CTRL_BONE_KEY]
right_elbow_ctrl_pbone = armature_object.pose.bones[RIGHT_ELBOW_CTRL_BONE_KEY]

left_hand_ik = get_or_create_ik_constraint(left_forearm_pbone)
left_hand_ik.target = armature_object
left_hand_ik.subtarget = LEFT_HAND_CTRL_BONE_KEY
left_hand_ik.chain_count = 2
left_hand_ik.pole_target = armature_object
left_hand_ik.pole_subtarget = LEFT_ELBOW_CTRL_BONE_KEY
left_hand_ik.pole_angle = 0.0

right_hand_ik = get_or_create_ik_constraint(right_forearm_pbone)
right_hand_ik.target = armature_object
right_hand_ik.subtarget = RIGHT_HAND_CTRL_BONE_KEY
right_hand_ik.chain_count = 2
right_hand_ik.pole_target = armature_object
right_hand_ik.pole_subtarget = RIGHT_ELBOW_CTRL_BONE_KEY
right_hand_ik.pole_angle = 0.0

