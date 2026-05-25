## Randomize camera around the bicycle with RTMPose-friendly framing constraints.

import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES

KEYPOINTS_COLLECTION = os.environ.get("CAMERA_KEYPOINTS_COLLECTION", "Keypoints")

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


def _render_size(scene_obj):
    render = scene_obj.render
    scale = render.resolution_percentage / 100.0
    return int(render.resolution_x * scale), int(render.resolution_y * scale)


def _focal_px(scene_obj, cam_obj, width, height):
    cam_data = cam_obj.data
    sensor_fit = cam_data.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= height else "VERTICAL"
    if sensor_fit == "VERTICAL":
        return cam_data.lens / cam_data.sensor_height * height
    return cam_data.lens / cam_data.sensor_width * width


def resolve_keypoint_objects():
    collection = bpy.data.collections.get(KEYPOINTS_COLLECTION)
    if collection is None:
        return {}
    objects = {}
    for name in BICYCLE_KEYPOINT_NAMES:
        obj = collection.objects.get(name)
        if obj is None and name.startswith("k_"):
            obj = collection.objects.get(name[2:])
        if obj is not None:
            objects[name] = obj
    return objects


def _keypoint_world_positions(keypoint_objects, depsgraph):
    positions = []
    for obj in keypoint_objects.values():
        obj_eval = obj.evaluated_get(depsgraph)
        positions.append(obj_eval.matrix_world.translation.copy())
    return positions


def _motion_radius(target_location, keypoint_positions):
    if not keypoint_positions:
        return 1.0
    return max((pos - target_location).length for pos in keypoint_positions)


def _required_min_distance(radius, focal_px, image_height, fit_margin):
    vertical_fov = 2.0 * np.arctan(float(image_height) / (2.0 * max(1e-6, float(focal_px))))
    return float(radius) / max(1e-6, np.tan(vertical_fov * 0.5)) * float(fit_margin)


def _apply_camera_pose(cam_obj, location, target_location):
    cam_obj.location = location
    look_at = target_location - location
    cam_obj.rotation_euler = look_at.to_track_quat("-Z", "Y").to_euler()


def _keypoint_screen_metrics(scene_obj, cam_obj, depsgraph, keypoint_objects, width, height):
    xs = []
    ys = []
    visible = 0
    for obj in keypoint_objects.values():
        world_co = obj.evaluated_get(depsgraph).matrix_world.translation
        co_ndc = world_to_camera_view(scene_obj, cam_obj, world_co)
        if co_ndc.z <= 0.0:
            continue
        px = float(co_ndc.x * width)
        py = float((1.0 - co_ndc.y) * height)
        if px < 0.0 or py < 0.0 or px > width or py > height:
            continue
        visible += 1
        xs.append(px)
        ys.append(py)
    if not xs:
        return 0.0, 0
    bbox_area = max(1.0, max(xs) - min(xs)) * max(1.0, max(ys) - min(ys))
    return bbox_area / max(1.0, float(width * height)), visible


def _visibility_check_frames(frame_start, frame_end, max_checks=48):
    total = max(1, frame_end - frame_start + 1)
    if total <= max_checks:
        return list(range(frame_start, frame_end + 1))
    step = max(1, total // max_checks)
    return list(range(frame_start, frame_end + 1, step))[:max_checks]


def _bbox_ok_across_frames(
    scene_obj,
    cam_obj,
    keypoint_objects,
    frame_start,
    frame_end,
    width,
    height,
    *,
    min_bbox_area_frac,
    max_bbox_area_frac,
    min_visible_keypoints,
    max_low_bbox_frame_frac,
):
    check_frames = _visibility_check_frames(frame_start, frame_end)
    low_bbox_frames = 0
    current_frame = scene_obj.frame_current
    for frame in check_frames:
        scene_obj.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bbox_area_frac, visible = _keypoint_screen_metrics(
            scene_obj, cam_obj, depsgraph, keypoint_objects, width, height
        )
        if visible < min_visible_keypoints:
            return False
        if bbox_area_frac < min_bbox_area_frac:
            low_bbox_frames += 1
        if bbox_area_frac > max_bbox_area_frac:
            return False
    scene_obj.frame_set(current_frame)
    bpy.context.view_layer.update()
    allowed_low = int(np.floor(max(0.0, max_low_bbox_frame_frac) * len(check_frames)))
    return low_bbox_frames <= allowed_low


def _visibility_ok_across_frames(
    scene_obj,
    cam_obj,
    keypoint_objects,
    frame_start,
    frame_end,
    *,
    min_visible_keypoints,
    min_visible_frame_ratio,
):
    if not keypoint_objects:
        return False
    width, height = _render_size(scene_obj)
    check_frames = _visibility_check_frames(frame_start, frame_end)
    visible_frames = 0
    current_frame = scene_obj.frame_current
    for frame in check_frames:
        scene_obj.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        _area_frac, visible = _keypoint_screen_metrics(
            scene_obj, cam_obj, depsgraph, keypoint_objects, width, height
        )
        if visible >= min_visible_keypoints:
            visible_frames += 1
    scene_obj.frame_set(current_frame)
    bpy.context.view_layer.update()
    return (visible_frames / max(1, len(check_frames))) >= min_visible_frame_ratio


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
min_camera_distance = float(os.environ.get("CAMERA_MIN_DISTANCE", "4.0"))
max_camera_distance = float(os.environ.get("CAMERA_MAX_DISTANCE", "12.0"))
min_bbox_area_frac = float(os.environ.get("CAMERA_MIN_BBOX_AREA_FRAC", "0.04"))
max_bbox_area_frac = float(os.environ.get("CAMERA_MAX_BBOX_AREA_FRAC", "0.80"))
min_visible_keypoints = int(os.environ.get("CAMERA_MIN_VISIBLE_KEYPOINTS", "14"))
min_visible_frame_ratio = float(os.environ.get("CAMERA_MIN_VISIBLE_FRAME_RATIO", "0.9"))
max_low_bbox_frame_frac = float(os.environ.get("CAMERA_MAX_LOW_BBOX_FRAME_FRAC", "0.15"))
camera_fit_margin = float(os.environ.get("CAMERA_FIT_MARGIN", "1.25"))
sync_window_size = int(os.environ.get("CAMERA_SYNC_WINDOW_SIZE", str(max(1, frame_end - frame_start + 1))))

width, height = _render_size(scene)
focal_px = _focal_px(scene, camera, width, height)
keypoint_objects = resolve_keypoint_objects()
keypoint_positions = _keypoint_world_positions(keypoint_objects, depsgraph)
motion_radius = _motion_radius(bicycle_location, keypoint_positions)
required_distance = _required_min_distance(motion_radius, focal_px, height, camera_fit_margin)
distance_low = max(min_camera_distance, required_distance)
distance_high = max(distance_low, max_camera_distance)
if distance_low > distance_high:
    raise RuntimeError(
        "Camera distance range is empty after RTMPose framing constraints. "
        f"required_distance={required_distance:.2f}m exceeds CAMERA_MAX_DISTANCE={max_camera_distance:.2f}m. "
        "Increase CAMERA_MAX_DISTANCE or CAMERA_FIT_MARGIN, or reduce CAMERA_MIN_BBOX_AREA_FRAC."
    )

half_window = max(0, sync_window_size // 2)
visibility_start = max(frame_start, sample_frame - half_window)
visibility_end = min(frame_end, sample_frame + half_window)
camera_mode = os.environ.get("CAMERA_MODE", "track").strip().lower()

camera_location = None
chosen_distance = None
chosen_elevation = None
chosen_azimuth = None
for _ in range(max_tries):
    camera_distance = random.uniform(distance_low, distance_high)
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

    _apply_camera_pose(camera, candidate_location, bicycle_location)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox_area_frac, visible_now = _keypoint_screen_metrics(
        scene, camera, depsgraph, keypoint_objects, width, height
    )
    if visible_now < min_visible_keypoints:
        continue
    if bbox_area_frac < min_bbox_area_frac or bbox_area_frac > max_bbox_area_frac:
        continue
    if camera_mode == "track":
        if not _bbox_ok_across_frames(
            scene,
            camera,
            keypoint_objects,
            visibility_start,
            visibility_end,
            width,
            height,
            min_bbox_area_frac=min_bbox_area_frac,
            max_bbox_area_frac=max_bbox_area_frac,
            min_visible_keypoints=min_visible_keypoints,
            max_low_bbox_frame_frac=max_low_bbox_frame_frac,
        ):
            continue
    elif not _visibility_ok_across_frames(
        scene,
        camera,
        keypoint_objects,
        visibility_start,
        visibility_end,
        min_visible_keypoints=min_visible_keypoints,
        min_visible_frame_ratio=min_visible_frame_ratio,
    ):
        continue

    camera_location = candidate_location
    chosen_distance = camera_distance
    chosen_elevation = elevation
    chosen_azimuth = azimuth
    break

if camera_location is None:
    raise RuntimeError(
        f"Failed to sample valid camera pose after {max_tries} tries. "
        "Adjust camera ranges, CAMERA_MIN_BBOX_AREA_FRAC, or prop setup. "
        f"distance_range=[{distance_low:.2f}, {distance_high:.2f}]m "
        f"bbox_area_frac=[{min_bbox_area_frac:.3f}, {max_bbox_area_frac:.3f}]"
    )

_apply_camera_pose(camera, camera_location, bicycle_location)

camera["camera_mode"] = camera_mode
if camera_mode == "track":
    track_offset = camera_location - bicycle_location
    camera["track_offset_x"] = float(track_offset.x)
    camera["track_offset_y"] = float(track_offset.y)
    camera["track_offset_z"] = float(track_offset.z)
    # Debug / legacy (not used by track_camera_to_target.py after offset fix).
    camera["track_distance"] = float(chosen_distance)
    camera["track_elevation"] = float(chosen_elevation)
    camera["track_azimuth"] = float(chosen_azimuth)

camera.animation_data_clear()

if camera_mode == "fixed":
    # World-fixed pose for the full timeline (legacy).
    camera.keyframe_insert(data_path="location", frame=frame_start)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_start)
    camera.keyframe_insert(data_path="location", frame=frame_end)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame_end)
# track mode: per-frame keyframes are written by track_camera_to_target.py after sync window.

scene.frame_set(current_frame)
bpy.context.view_layer.update()

