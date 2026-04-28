# Haye Boonstra, 2026
# Thesis project: 3D pose estimation of bicycle kinematics using neural networks

# This script is used to rig a bicycle model in Blender and attach the keypoints desired for the 2D and 3D pose estimation.
# it can only be used with a certain naming scheme, although child objects of these parts can be arbitrary and will be animated alongside the parent.
# required parts of the bicycle can be found in the names list.

# To automatically align the steer bone along the steering axis, a marker object with the name "steer alignment marker" is required.
# Check the origin of the steer/fork object to make sure that it is situated on the steering axis.
# Place an empty object with the name "steer alignment marker" along the steering axis underneath the steer objects origin. If the marker is placed higher than the origin, the steering will be flipped.

# In general the forward direction is X, Y is sideways and Z is up.
# keep in mind that bones have a different axis system that normal objects. 
# bones have a Y axis that is always along the bone axis. The X and Z axes are defined based on the "roll" of the bone.
# If a bicycle is animated strangely, check the bone directions and see if the X and Z axes are pointing in the correct direction.
# Don´t look at the object axes, look at the bone axes. Don´t ask me how much time it cost to figure this out.
import bpy
from mathutils import Vector

## set the collection to bicycle
bicycle_collection = bpy.data.collections.get("Bicycle")
if bicycle_collection is None:
    bicycle_collection = bpy.data.collections.new("Bicycle")
if bicycle_collection.name not in bpy.context.scene.collection.children:
    bpy.context.scene.collection.children.link(bicycle_collection)


def find_layer_collection(layer_collection, target_name):
    if layer_collection.collection.name == target_name:
        return layer_collection
    for child in layer_collection.children:
        result = find_layer_collection(child, target_name)
        if result is not None:
            return result
    return None


# Mimic clicking the Bicycle collection in the Outliner.
bicycle_layer_collection = find_layer_collection(
    bpy.context.view_layer.layer_collection, bicycle_collection.name
)
if bicycle_layer_collection is not None:
    bpy.context.view_layer.active_layer_collection = bicycle_layer_collection
# part names
names = ["frame", "steer", "crank", "right_pedal", "left_pedal", "front_wheel", "rear_wheel"]
bone_directions = {
    "frame": "Y",
    "steer": "X", # the steer bone will be aligned along the steering axis later.
    "crank": "Y",
    "right_pedal": "Y",
    "left_pedal": "Y",
    "front_wheel": "Y",
    "rear_wheel": "Y",
}

# Assign vertex groups to the parts
for name in names:
    obj = bpy.data.objects[name]
    # Create new vertex group if it doesn't already exist
    if name not in obj.vertex_groups:
        obj.vertex_groups.new(name=name)
    vg = obj.vertex_groups[name]
    for vertex in obj.data.vertices:
        # In Blender Python 2.8+/3.x API, 'type' must be specified for add()
        # Set 'type' to 'REPLACE' (sometimes given as 'assign' in old scripts)
        vg.add([vertex.index], 1.0, 'REPLACE')

# Reuse armature if it already exists to avoid Bicycle_Armature.001.
armature_object = bpy.data.objects.get("Bicycle_Armature")
if armature_object is None:
    armature_data = bpy.data.armatures.new(name="Bicycle")
    armature_object = bpy.data.objects.new("Bicycle_Armature", armature_data)
    bicycle_collection.objects.link(armature_object)
else:
    armature_data = armature_object.data
    if armature_object.name not in bicycle_collection.objects:
        bicycle_collection.objects.link(armature_object)

# Create bones for the armature. All bones are part of the same armature as done in edit mode in the video
# bones have a specific orientation found in bone_directions

# Make the new armature active and enter Edit Mode (bones can only be edited there).
bpy.context.view_layer.objects.active = armature_object
armature_object.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')

edit_bones = armature_data.edit_bones
# Remove old bones so reruns stay deterministic.
while edit_bones:
    edit_bones.remove(edit_bones[0])

bone_length = 0.20


def bone_tail_from_axis(head, axis, length):
    if axis == "X":
        return head + Vector((length, 0.0, 0.0))
    if axis == "Y":
        return head + Vector((0.0, length, 0.0))
    return head + Vector((0.0, 0.0, length))


def bone_name_for_part(part_name):
    if part_name == "frame":
        return "b_root"
    return f"b_{part_name}"


# Create one template bone first, then duplicate/position it for each part.
template_part = names[0]
template_obj = bpy.data.objects[template_part]
template_axis = bone_directions.get(template_part, "Y")
template_head = armature_object.matrix_world.inverted() @ template_obj.matrix_world.translation

template_bone = edit_bones.new("b_root")
template_bone.head = template_head
template_bone.tail = bone_tail_from_axis(template_head, template_axis, bone_length)

for part_name in [name for name in names[1:] if name != "steer"]:
    part_obj = bpy.data.objects[part_name]
    axis = bone_directions.get(part_name, "Y")
    head = armature_object.matrix_world.inverted() @ part_obj.matrix_world.translation

    bone = edit_bones.new(f"b_{part_name}")
    bone.head = head
    bone.tail = bone_tail_from_axis(head, axis, bone_length)

# Create the steer bone
steer_obj = bpy.data.objects["steer"]
steer_axis = bone_directions.get("steer", "X")
steer_head = armature_object.matrix_world.inverted() @ steer_obj.matrix_world.translation
steer_bone = edit_bones.new("b_steer")
steer_bone.head = steer_head

# If the alignment marker exists, place steer tail exactly on it.
steer_tail_marker = bpy.data.objects.get("steer alignment marker")
if steer_tail_marker is not None:
    steer_tail = armature_object.matrix_world.inverted() @ steer_tail_marker.matrix_world.translation
    # Prevent zero-length bone if marker matches head location.
    if (steer_tail - steer_head).length > 1e-6:
        steer_bone.tail = steer_tail
    else:
        steer_bone.tail = bone_tail_from_axis(steer_head, steer_axis, bone_length)
else:
    steer_bone.tail = bone_tail_from_axis(steer_head, steer_axis, bone_length)

# Parent hierarchy must be assigned in Edit Mode.
parent_bones = {
    "b_root": None,
    "b_steer": "b_root",
    "b_crank": "b_root",
    "b_right_pedal": "b_crank",
    "b_left_pedal": "b_crank",
    "b_front_wheel": "b_steer",
    "b_rear_wheel": "b_root",
}

for part_name in names:
    bone_name = bone_name_for_part(part_name)
    parent_bone = parent_bones[bone_name]
    if parent_bone is not None:
        edit_bones[bone_name].parent = edit_bones[parent_bone]

bpy.ops.object.mode_set(mode='OBJECT')

# Bind each bicycle part object directly to its matching bone.
for part_name in names:
    part_obj = bpy.data.objects[part_name]
    bone_name = bone_name_for_part(part_name)

    world_matrix = part_obj.matrix_world.copy()
    part_obj.parent = armature_object
    part_obj.parent_type = 'BONE'
    part_obj.parent_bone = bone_name
    # Keep object at the same world-space transform after parenting.
    part_obj.matrix_world = world_matrix

## place the keypoints on the bicycle for pose estimation and foot alignment
# Each tuple follows the convention: (parent bone name, keypoint object name).
keypoints = [
    ("b_root", "bottom_bracket"),
    ("b_root", "seat_stay"),
    ("b_root", "rear_hub_left"),
    ("b_root", "rear_hub_right"),
    ("b_root", "rear_wheel_ground"),
    ("b_root", "upper_head_tube"),
    ("b_root", "lower_head_tube"),
    ("b_root", "saddle"),
    ("b_left_pedal", "left_pedal"),
    ("b_left_pedal", "left_pedal_tracker"),
    ("b_right_pedal", "right_pedal"),
    ("b_right_pedal", "right_pedal_tracker"),
    ("b_steer", "handle_bar_middle"),
    ("b_steer", "handle_bar_left"),
    ("b_steer", "handle_bar_right"),
    ("b_steer", "front_hub_left"),
    ("b_steer", "front_hub_right"),
    ("b_steer", "front_wheel_back"),
    ("b_steer", "front_wheel_front"),
    ("b_steer", "front_wheel_ground"),
]

keypoints_collection = bpy.data.collections.get("Keypoints")
if keypoints_collection is None:
    keypoints_collection = bpy.data.collections.new("Keypoints")
    bpy.context.scene.collection.children.link(keypoints_collection)

for parent_bone_name, keypoint_name in keypoints:
    keypoint_obj = bpy.data.objects.get(keypoint_name)
    if keypoint_obj is None:
        keypoint_obj = bpy.data.objects.new(keypoint_name, None)
        keypoint_obj.empty_display_type = 'PLAIN_AXES'
        keypoint_obj.empty_display_size = 0.02
        keypoints_collection.objects.link(keypoint_obj)
    elif keypoint_obj.type == 'EMPTY' and keypoint_name not in keypoints_collection.objects:
        for collection in list(keypoint_obj.users_collection):
            if collection.name != keypoints_collection.name:
                collection.objects.unlink(keypoint_obj)
        keypoints_collection.objects.link(keypoint_obj)

    world_matrix = keypoint_obj.matrix_world.copy()
    if keypoint_obj.type == 'EMPTY':
        keypoint_obj.empty_display_type = 'PLAIN_AXES'
        keypoint_obj.empty_display_size = 0.02
    keypoint_obj.parent = armature_object
    keypoint_obj.parent_type = 'BONE'
    keypoint_obj.parent_bone = parent_bone_name
    # Keep the keypoint in place if it already existed before parenting.
    keypoint_obj.matrix_world = world_matrix


## align the keypoints to an initial position closer to the actual position to make manual alignment easier.
# Since there is no real way of knowing the true positions of the keypoints we need to do manual adjustment.
# however, we can get quite close by using the bounding boxes!

keypoints_locations = [
    ("bottom_bracket", "crank"),
    ("seat_stay", "seat"),
    ("rear_hub_left", "rear_wheel"),
    ("rear_hub_right", "rear_wheel"),
    ("rear_wheel_ground", "rear_wheel"),
    ("upper_head_tube", "steer"),
    ("lower_head_tube", "frame"),
    ("saddle", "seat"),
    ("left_pedal", "left_pedal"),
    ("left_pedal_tracker", "left_pedal"),
    ("right_pedal", "right_pedal"),
    ("right_pedal_tracker", "right_pedal"),
    ("handle_bar_middle", "steer"),
    ("handle_bar_left", "steer"),
    ("handle_bar_right", "steer"),
    ("front_hub_left", "front_wheel"),
    ("front_hub_right", "front_wheel"),
    ("front_wheel_back", "front_wheel"),
    ("front_wheel_front", "front_wheel"),
    ("front_wheel_ground", "front_wheel"),
]

for keypoint_name, parent_object_name in keypoints_locations:
    keypoint_obj = bpy.data.objects.get(keypoint_name)
    parent_obj = bpy.data.objects.get(parent_object_name)

    if keypoint_obj is None or parent_obj is None:
        continue

    bbox_world = [parent_obj.matrix_world @ Vector(corner) for corner in parent_obj.bound_box]
    bbox_center = sum(bbox_world, Vector((0.0, 0.0, 0.0))) / len(bbox_world)

    x_axis = (parent_obj.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    y_axis = (parent_obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
    z_axis = (parent_obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()

    x_values = [corner.dot(x_axis) for corner in bbox_world]
    y_values = [corner.dot(y_axis) for corner in bbox_world]
    z_values = [corner.dot(z_axis) for corner in bbox_world]

    half_length = (max(x_values) - min(x_values)) * 0.5
    half_width = (max(y_values) - min(y_values)) * 0.5
    half_height = (max(z_values) - min(z_values)) * 0.5

    location = parent_obj.matrix_world.translation.copy()

    # nasty elif statements for the 
    if keypoint_name in {"rear_hub_left", "front_hub_left"}:
        location = bbox_center - y_axis * half_width
    elif keypoint_name in {"rear_hub_right", "front_hub_right"}:
        location = bbox_center + y_axis * half_width
    elif keypoint_name in {"rear_wheel_ground", "front_wheel_ground"}:
        location = bbox_center - z_axis * half_height
    elif keypoint_name == "front_wheel_back":
        location = bbox_center - x_axis * half_length
    elif keypoint_name == "front_wheel_front":
        location = bbox_center + x_axis * half_length
    elif keypoint_name == "upper_head_tube":
        location = bbox_center - z_axis * half_height
    elif keypoint_name == "lower_head_tube":
        location = bbox_center + x_axis * half_length
    elif keypoint_name == "handle_bar_left":
        location = bbox_center - y_axis * half_width
    elif keypoint_name == "handle_bar_right":
        location = bbox_center + y_axis * half_width
    elif keypoint_name == "handle_bar_middle":
        location = bbox_center + z_axis * min(half_height * 0.25, 0.05)
    elif keypoint_name in {"left_pedal_tracker", "right_pedal_tracker"}:
        location = parent_obj.matrix_world.translation + z_axis * min(max(half_height, 0.03), 0.08)

    keypoint_obj.matrix_world.translation = location