import math
import os
from pathlib import Path

import bpy
import csv
from mathutils import Quaternion, Vector
ARMATURE_NAME = "Bicycle_Armature"
ROOT_BONE_NAME = "b_root"
STEER_BONE_NAME = "b_steer"
REAR_WHEEL_BONE_NAME = "b_rear_wheel"
FRONT_WHEEL_BONE_NAME = "b_front_wheel"
CRANK_BONE_NAME = "b_crank"
LEFT_PEDAL_BONE_NAME = "b_left_pedal"
RIGHT_PEDAL_BONE_NAME = "b_right_pedal"
CAMERA_NAME = "Camera"


def _trajectory_csv_path() -> Path:
    requested = os.environ.get("TRAJECTORY_CSV", "").strip()
    if requested:
        return Path(requested).expanduser().resolve()
    directory = Path(os.environ.get("REPO_ROOT", "~/3D-bicycle-pose-estimation")).expanduser()
    return (directory / "mujoco_composite_00000_60hz.csv").resolve()


def _read_trajectory(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {key: float(value) for key, value in row.items() if value not in {None, ""}}
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Trajectory CSV is empty: {path}")
    return rows


def _row_value(row: dict[str, float], key: str, default: float = 0.0) -> float:
    return float(row.get(key, default))


trajectory_path = _trajectory_csv_path()
rows = _read_trajectory(trajectory_path)
arm = bpy.data.objects[ARMATURE_NAME]
root = arm.pose.bones[ROOT_BONE_NAME]
steer = arm.pose.bones[STEER_BONE_NAME]
rear_wheel = arm.pose.bones[REAR_WHEEL_BONE_NAME]
front_wheel = arm.pose.bones[FRONT_WHEEL_BONE_NAME]
crank = arm.pose.bones[CRANK_BONE_NAME]
left_pedal = arm.pose.bones[LEFT_PEDAL_BONE_NAME]
right_pedal = arm.pose.bones[RIGHT_PEDAL_BONE_NAME]

root.rotation_mode = "QUATERNION"
steer.rotation_mode = "QUATERNION"
rear_wheel.rotation_mode = "QUATERNION"
crank.rotation_mode = "QUATERNION"
front_wheel.rotation_mode = "QUATERNION"
left_pedal.rotation_mode = "QUATERNION"
right_pedal.rotation_mode = "QUATERNION"

scene = bpy.context.scene
fps = int(float(os.environ.get("TRAJECTORY_FPS", scene.render.fps or 60)))
scene.render.fps = fps
frame_offset = int(os.environ.get("TRAJECTORY_FRAME_START", str(scene.frame_start)))
if os.environ.get("TRAJECTORY_RESET_FRAME_RANGE", "1") == "1":
    scene.frame_start = frame_offset

# Freeze camera in world space using its CURRENT transform.
# Prefer the active scene camera; fallback to name for compatibility.
camera = scene.camera if scene.camera is not None else bpy.data.objects.get(CAMERA_NAME)
if camera is not None:
    camera_world_current = camera.matrix_world.copy()
    if camera.parent is not None:
        camera.parent = None
    camera.matrix_world = camera_world_current

# Clear old armature animation data so imported trajectory is deterministic.
arm.animation_data_clear()

for row in rows:
    time_sec = _row_value(row, "time", 0.0)
    frame = frame_offset + int(round(time_sec * fps))
    scene.frame_set(frame)
    ## root joint location and orientation setting
    p = Vector((_row_value(row, "tx"), _row_value(row, "ty"), _row_value(row, "tz")))
    root.location = p
    root.keyframe_insert(data_path="location")
    q = Quaternion(
        (
            _row_value(row, "rw", 1.0),
            _row_value(row, "rx"),
            _row_value(row, "ry"),
            _row_value(row, "rz"),
        )
    )
    q.normalize()
    root.rotation_quaternion = q
    root.keyframe_insert(data_path="rotation_quaternion")
    ## steer angle
    steer_q = Quaternion((0.0, 1.0, 0.0), -math.radians(_row_value(row, "steer_angle")))
    steer.rotation_quaternion = steer_q
    steer.keyframe_insert(data_path="rotation_quaternion")
    ## front wheel rotation
    front_wheel_q = Quaternion((0.0, 1.0, 0.0), math.radians(_row_value(row, "front_wheel_angle")))
    front_wheel.rotation_quaternion = front_wheel_q
    front_wheel.keyframe_insert(data_path="rotation_quaternion")
    ## rear wheel rotation
    rear_wheel_q = Quaternion((0.0, 1.0, 0.0), math.radians(_row_value(row, "rear_wheel_angle")))
    rear_wheel.rotation_quaternion = rear_wheel_q
    rear_wheel.keyframe_insert(data_path="rotation_quaternion")
    ## crank rotation
    crank_q = Quaternion((0.0, 1.0, 0.0), math.radians(_row_value(row, "crank_angle")))
    crank.rotation_quaternion = crank_q
    crank.keyframe_insert(data_path="rotation_quaternion")
    ## left pedal rotation
    left_pedal_q = Quaternion((0.0, 1.0, 0.0), math.radians(-_row_value(row, "left_pedal_angle")))
    left_pedal.rotation_quaternion = left_pedal_q
    left_pedal.keyframe_insert(data_path="rotation_quaternion")
    ## right pedal rotation
    right_pedal_q = Quaternion((0.0, 1.0, 0.0), math.radians(_row_value(row, "right_pedal_angle")))
    right_pedal.rotation_quaternion = right_pedal_q
    right_pedal.keyframe_insert(data_path="rotation_quaternion")

last_frame = frame_offset + int(round(_row_value(rows[-1], "time") * fps))
scene.frame_end = max(scene.frame_end, last_frame)

if camera is not None:
    scene.camera = camera
    camera.matrix_world = camera_world_current

    # Remove old camera transform keyframes to avoid restoring older poses.
    # Use keyframe_delete for compatibility across Blender animation APIs.
    try:
        delete_start = int(scene.frame_start)
        delete_end = int(max(scene.frame_end, last_frame))
        for frame in range(delete_start, delete_end + 1):
            camera.keyframe_delete(data_path="location", frame=frame)
            camera.keyframe_delete(data_path="rotation_quaternion", frame=frame)
            camera.keyframe_delete(data_path="rotation_euler", frame=frame)
    except:
        print("error in deleting camera transform keyframes or they don't exist")

    scene.frame_set(frame_offset)
    camera.keyframe_insert(data_path="location", frame=frame_offset)
    if camera.rotation_mode == "QUATERNION":
        camera.keyframe_insert(data_path="rotation_quaternion", frame=frame_offset)
    else:
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset)

    camera.keyframe_insert(data_path="location", frame=last_frame)
    if camera.rotation_mode == "QUATERNION":
        camera.keyframe_insert(data_path="rotation_quaternion", frame=last_frame)
    else:
        camera.keyframe_insert(data_path="rotation_euler", frame=last_frame)

    # Auto-bind camera marker at frame 0 and clear old camera marker bindings.
    for marker in list(scene.timeline_markers):
        if marker.camera is not None:
            scene.timeline_markers.remove(marker)
    marker = scene.timeline_markers.new("RenderCamera", frame=frame_offset)
    marker.camera = camera
# set the final frame to the length of the animation
scene.frame_end = last_frame

## set the frame in animation to the frame 0
scene.frame_set(scene.frame_start)

