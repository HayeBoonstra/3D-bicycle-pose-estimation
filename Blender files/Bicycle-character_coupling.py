# Haye Boonstra, 2026
# Thesis project: 3D pose estimation of bicycle kinematics using neural networks
# A script to couple a bicycle rig with a character rig.
# It assumes 

def resolve_mixamo_bone_name(armature_or_data, target_bone_name: str) -> str:
    """Resolve a bone name independent of rig prefixes (e.g. mixamorig:)."""
    names = []

    # Accept both armature object (obj.type == 'ARMATURE') and armature data.
    if hasattr(armature_or_data, "bones"):
        names.extend([bone.name for bone in armature_or_data.bones])

    if hasattr(armature_or_data, "pose") and armature_or_data.pose:
        names.extend([bone.name for bone in armature_or_data.pose.bones])

    # De-duplicate while preserving order.
    names = list(dict.fromkeys(names))
    key_lower = target_bone_name.lower()

    exact_matches = [name for name in names if name.lower() == key_lower]
    if exact_matches:
        return exact_matches[0]

    suffix_matches = [name for name in names if name.lower().endswith(f":{key_lower}")]
    if suffix_matches:
        return sorted(suffix_matches, key=len)[0]

    generic_suffix_matches = [name for name in names if name.lower().endswith(key_lower)]
    if generic_suffix_matches:
        return sorted(generic_suffix_matches, key=len)[0]

    raise KeyError(
        f'Could not find a bone matching "{target_bone_name}" in armature "{armature_or_data.name}". '
        f'Available bones: {names[:20]}{"..." if len(names) > 20 else ""}'
    )

def move_armature_to_point(armature, current_point, target_point):
    """ move the armature to the point """
    translation_vector = target_point - current_point
    armature.location += translation_vector

def clear_parent_keep_world_transform(obj):
    """Clear parent while preserving world transform."""
    world_matrix = obj.matrix_world.copy()
    obj.parent = None
    obj.parent_type = 'OBJECT'
    obj.parent_bone = ""
    obj.matrix_world = world_matrix

def parent_to_bone_keep_world_transform(child_obj, parent_armature_obj, parent_bone_name: str):
    """Parent to a bone while preserving the child's world transform."""
    view_layer = bpy.context.view_layer
    previously_selected_objects = list(bpy.context.selected_objects)
    previously_active_object = view_layer.objects.active

    for obj in previously_selected_objects:
        obj.select_set(False)
     
    parent_armature_obj.select_set(True)
    child_obj.select_set(True)
    view_layer.objects.active = parent_armature_obj
    parent_armature_obj.data.bones.active = parent_armature_obj.data.bones[parent_bone_name]

    # Use Blender's native parenting op with keep_transform to avoid bone head/tail offsets.
    bpy.ops.object.parent_set(type='BONE', keep_transform=True)

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in previously_selected_objects:
        obj.select_set(True)
    view_layer.objects.active = previously_active_object

import bpy
import numpy as np

BICYCLE_COLLECTION_NAME = "Bicycle"
CHARACTER_COLLECTION_NAME = "Character"
KEYPOINTS_COLLECTION_NAME = "Keypoints"
BICYCLE_NAME = "Bicycle_Armature"
CHARACTER_NAME = "Character_Armature"

forward_tilt_angle = np.deg2rad(10)# degrees
hip_z_offset = 0.1 # let the character sit properly on the saddle

bicycle = bpy.data.collections[BICYCLE_COLLECTION_NAME].objects[BICYCLE_NAME]
character = bpy.data.collections[CHARACTER_COLLECTION_NAME].objects[CHARACTER_NAME]

# Avoid compounding offsets if script is executed repeatedly.
if character.parent is not None:
    clear_parent_keep_world_transform(character)

# get the character's hip.deg2rad(10.0)  bone as it serves at the root of the character rig
character_hip_bone_name = resolve_mixamo_bone_name(character, "Hips")
character_hip_bone = character.pose.bones[character_hip_bone_name]

# move the character's hip bone to the bicycle's saddle keypoint
saddle_point = bpy.data.collections[KEYPOINTS_COLLECTION_NAME].objects["saddle"].matrix_world.translation.copy()
saddle_point.z += hip_z_offset

character_hip_bone_world_position = (character.matrix_world @ character_hip_bone.matrix).translation
move_armature_to_point(character, character_hip_bone_world_position, saddle_point)

# tilt the character forward
character.rotation_euler.x += forward_tilt_angle

## parent the character to the bicycle
parent_to_bone_keep_world_transform(character, bicycle, resolve_mixamo_bone_name(bicycle, "b_root"))

## set the character's 




