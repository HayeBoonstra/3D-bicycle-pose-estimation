## this script is used to randomize the camera position in a sphere around the bicycle.

import os
import random

import bpy
import numpy as np
from mathutils import Vector

scene = bpy.context.scene
camera = scene.camera

if camera is None:
    raise RuntimeError("No active scene camera. Set scene.camera first.")

seed = os.environ.get("CAMERA_SEED")
if seed not in {None, ""}:
    random.seed(int(seed))


def resolve_camera_target():
    target_name = os.environ.get("CAMERA_TARGET", "k_handlebar_middle")
    candidates = [target_name]
    if target_name.startswith("k_"):
        candidates.append(target_name[2:])
    else:
        candidates.append(f"k_{target_name}")

    for name in candidates:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            return obj
    raise KeyError(f"Could not find camera target object. Tried: {candidates}")


def resolve_prop_objects():
    # Optional explicit list takes priority: CAMERA_PROP_NAMES="crate_01,barrier_A"
    prop_names = os.environ.get("CAMERA_PROP_NAMES", "").strip()
    if prop_names:
        objects = []
        for raw_name in prop_names.split(","):
            name = raw_name.strip()
            if not name:
                continue
            obj = bpy.data.objects.get(name)
            if obj is not None and obj.type == "MESH":
                objects.append(obj)
        if objects:
            return objects

    # Fallback to a collection name when explicit names are not provided.
    collection_name = os.environ.get("CAMERA_PROP_COLLECTION", "props").strip()
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return []
    return [obj for obj in collection.all_objects if obj.type == "MESH"]


def point_inside_mesh(obj_eval, point_world, max_hits=64):
    # Ray parity test in object-local space for watertight meshes.
    point_local = obj_eval.matrix_world.inverted() @ point_world
    direction_local = Vector((1.0, 0.0, 0.0))
    origin = point_local.copy()
    epsilon = 1e-5
    hit_count = 0

    for _ in range(max_hits):
        hit, location, _normal, _face_index = obj_eval.ray_cast(origin, direction_local)
        if not hit:
            break
        hit_count += 1
        origin = location + direction_local * epsilon

    return (hit_count % 2) == 1


def point_inside_world_bbox(obj_eval, point_world, padding=0.0):
    corners_world = [obj_eval.matrix_world @ Vector(corner) for corner in obj_eval.bound_box]
    min_corner = Vector(
        (
            min(corner.x for corner in corners_world) - padding,
            min(corner.y for corner in corners_world) - padding,
            min(corner.z for corner in corners_world) - padding,
        )
    )
    max_corner = Vector(
        (
            max(corner.x for corner in corners_world) + padding,
            max(corner.y for corner in corners_world) + padding,
            max(corner.z for corner in corners_world) + padding,
        )
    )
    return (
        min_corner.x <= point_world.x <= max_corner.x
        and min_corner.y <= point_world.y <= max_corner.y
        and min_corner.z <= point_world.z <= max_corner.z
    )


def camera_is_too_close_or_inside_props(
    camera_location,
    prop_objects,
    depsgraph,
    min_distance=0.8,
    bbox_padding=0.1,
):
    for prop in prop_objects:
        prop_eval = prop.evaluated_get(depsgraph)
        if point_inside_world_bbox(prop_eval, camera_location, padding=bbox_padding):
            return True
        is_on_surface, nearest, _normal, _index = prop_eval.closest_point_on_mesh(camera_location)
        if is_on_surface and (camera_location - nearest).length < min_distance:
            return True
        if point_inside_mesh(prop_eval, camera_location):
            return True
    return False


def object_is_target_or_parent(hit_obj, target_obj):
    obj = hit_obj
    while obj is not None:
        if obj == target_obj:
            return True
        obj = obj.parent
    return False


def object_is_in_name_set_or_parent(hit_obj, object_names):
    obj = hit_obj
    while obj is not None:
        if obj.name in object_names:
            return True
        obj = obj.parent
    return False


def line_of_sight_blocked(camera_location, target_location, target_obj, blocker_names, depsgraph):
    direction = target_location - camera_location
    distance = direction.length
    if distance <= 1e-6:
        return True
    direction.normalize()

    hit, _loc, _normal, _face, hit_obj, _matrix = scene.ray_cast(
        depsgraph,
        camera_location,
        direction,
        distance=distance - 1e-4,
    )
    if not hit or hit_obj is None:
        return False
    if object_is_target_or_parent(hit_obj, target_obj):
        return False
    return object_is_in_name_set_or_parent(hit_obj, blocker_names)


def poor_view_angle(camera_location, target_location, min_height_above_target, min_pitch_deg):
    # Reject very low/flat camera views that often produce "under-prop" images.
    if camera_location.z < target_location.z + min_height_above_target:
        return True

    direction = target_location - camera_location
    distance = direction.length
    if distance <= 1e-6:
        return True
    pitch_deg = np.rad2deg(np.arcsin(abs(direction.z) / distance))
    return pitch_deg < min_pitch_deg

## calculate the randomized parameters
frame_start = int(scene.frame_start)
frame_end = int(scene.frame_end)

# Sample the bicycle location at one random frame in its animation.
sample_frame = random.randint(frame_start, frame_end)
os.environ["CAMERA_SAMPLE_FRAME"] = str(sample_frame)
current_frame = scene.frame_current
scene.frame_set(sample_frame)
bpy.context.view_layer.update()

# Use evaluated world-space position at the sampled frame.
# `.location` is local transform and may appear constant if parented/constrained.
target_obj = resolve_camera_target()
depsgraph = bpy.context.evaluated_depsgraph_get()
target_eval = target_obj.evaluated_get(depsgraph)
bicycle_location = target_eval.matrix_world.translation.copy()
prop_objects = resolve_prop_objects()
blocker_names = {obj.name for obj in prop_objects}
max_tries = int(os.environ.get("CAMERA_MAX_TRIES", "200"))
min_height_above_target = float(os.environ.get("CAMERA_MIN_HEIGHT_ABOVE_TARGET", "-0.5"))
min_pitch_deg = float(os.environ.get("CAMERA_MIN_VIEW_PITCH_DEG", "-20.0"))
min_prop_clearance = float(os.environ.get("CAMERA_PROP_MIN_CLEARANCE", "0.8"))
prop_bbox_padding = float(os.environ.get("CAMERA_PROP_BBOX_PADDING", "0.1"))

camera_location = None
for _ in range(max_tries):
    camera_distance = random.uniform(4.0, 20.0)  # meters
    # generate the elevation and azimuth angles so that the camera never goes under the ground
    # gaussian distribution around the mean of 90 degrees with a standard deviation of 20 degrees
    elevation = random.gauss(np.deg2rad(90), np.deg2rad(20))  # radians
    azimuth = random.uniform(np.deg2rad(0), np.deg2rad(360))  # radians

    # calculate the camera location
    camera_offset = camera_distance * np.array(
        [
            np.sin(elevation) * np.cos(azimuth),
            np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )  # numpy array

    candidate_location = bicycle_location + Vector(camera_offset)
    # lift camera above the ground if it went under it. Ground is at Z = 0
    candidate_location[2] = max(candidate_location[2], 0.2)

    if poor_view_angle(
        candidate_location,
        bicycle_location,
        min_height_above_target=min_height_above_target,
        min_pitch_deg=min_pitch_deg,
    ):
        continue
    if camera_is_too_close_or_inside_props(
        candidate_location,
        prop_objects,
        depsgraph,
        min_distance=min_prop_clearance,
        bbox_padding=prop_bbox_padding,
    ):
        continue
    if line_of_sight_blocked(
        candidate_location,
        bicycle_location,
        target_obj,
        blocker_names,
        depsgraph,
    ):
        continue

    camera_location = candidate_location
    break

if camera_location is None:
    raise RuntimeError(
        f"Failed to sample valid camera pose after {max_tries} tries. "
        "Adjust camera ranges or prop setup."
    )

camera.location = camera_location
# set the camera rotation so it looks at the bicycle
camera_lookat = bicycle_location - camera_location
camera.rotation_euler = camera_lookat.to_track_quat('-Z', 'Y').to_euler()

# Remove any existing camera animation so previous scene keyframes cannot make
# the camera move during the rendered clip.
camera.animation_data_clear()

# Keep this exact camera pose fixed over the full animation range.
camera.keyframe_insert(data_path="location", frame=frame_start)
camera.keyframe_insert(data_path="rotation_euler", frame=frame_start)
camera.keyframe_insert(data_path="location", frame=frame_end)
camera.keyframe_insert(data_path="rotation_euler", frame=frame_end)

# scene.frame_set(current_frame)

