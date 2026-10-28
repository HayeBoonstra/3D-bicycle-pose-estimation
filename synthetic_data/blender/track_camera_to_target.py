"""Keyframe the camera to follow the bicycle target (RTMPose-friendly framing).

Uses the world-space offset from the random pose sampled in randomize_camera.py
(camera position minus target at the sample frame). Each frame:

    camera.location = target(frame) + offset_world
    camera looks at target(frame)

So the rig keeps the same relative placement chosen at randomization while translating
with the bicycle. It does not re-interpret spherical angles (that caused top-down views).
"""

from __future__ import annotations

import os

import bpy
from mathutils import Vector

scene = bpy.context.scene
camera = scene.camera
if camera is None:
    raise RuntimeError("No active scene camera.")

if "track_offset_x" not in camera:
    raise RuntimeError(
        "Camera is missing track_offset_* properties. Run randomize_camera.py with CAMERA_MODE=track first."
    )

offset_world = Vector(
    (
        float(camera["track_offset_x"]),
        float(camera["track_offset_y"]),
        float(camera["track_offset_z"]),
    )
)
target_name = os.environ.get("CAMERA_TARGET", "k_handlebar_middle")


def _resolve_target():
    candidates = [target_name]
    if target_name.startswith("k_"):
        candidates.append(target_name[2:])
    else:
        candidates.append(f"k_{target_name}")
    for name in candidates:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            return obj
    raise KeyError(f"Camera target not found: {candidates}")


def _target_world_position(target_obj, frame: int) -> Vector:
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return target_obj.evaluated_get(depsgraph).matrix_world.translation.copy()


def _apply_pose(cam_obj, location: Vector, target: Vector) -> None:
    cam_obj.location = location
    look_at = target - location
    cam_obj.rotation_euler = look_at.to_track_quat("-Z", "Y").to_euler()


target_obj = _resolve_target()
frame_start = int(scene.frame_start)
frame_end = int(scene.frame_end)
current_frame = scene.frame_current

camera.animation_data_clear()

for frame in range(frame_start, frame_end + 1):
    target_pos = _target_world_position(target_obj, frame)
    location = target_pos + offset_world
    location.z = max(location.z, 0.2)
    _apply_pose(camera, location, target_pos)
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)

scene.frame_set(current_frame)
bpy.context.view_layer.update()
