## this script is used to randomize the camera position in a sphere around the bicycle.

import bpy
import random
import numpy as np
from mathutils import Vector
scene = bpy.context.scene
camera = scene.camera

if camera is None:
    raise RuntimeError("No active scene camera. Set scene.camera first.")

## calculate the randomized parameters
frame_start = int(scene.frame_start)
frame_end = int(scene.frame_end)

# Sample the bicycle location at one random frame in its animation.
sample_frame = random.randint(frame_start, frame_end)
current_frame = scene.frame_current
scene.frame_set(sample_frame)
bpy.context.view_layer.update()

# Use evaluated world-space position at the sampled frame.
# `.location` is local transform and may appear constant if parented/constrained.
target_obj = bpy.data.objects["k_handlebar_middle"]
depsgraph = bpy.context.evaluated_depsgraph_get()
target_eval = target_obj.evaluated_get(depsgraph)
bicycle_location = target_eval.matrix_world.translation.copy()
camera_distance = random.uniform(3.0, 6.0) # meters
# generate the elevation and azimuth angles so that the camera never goes under the ground
elevation = random.uniform(np.deg2rad(0), np.deg2rad(90-10)) # radians
azimuth = random.uniform(np.deg2rad(0), np.deg2rad(360)) # radians

# calculate the camera location
camera_offset = camera_distance * np.array([
    np.sin(elevation) * np.cos(azimuth),
    np.sin(elevation) * np.sin(azimuth),
    np.cos(elevation)
]) # numpy array

camera_location = bicycle_location + Vector(camera_offset)

# set the camera location
camera.location = camera_location
# set the camera rotation so it looks at the bicycle
camera_lookat = bicycle_location - camera_location
camera.rotation_euler = camera_lookat.to_track_quat('-Z', 'Y').to_euler()

# remove all keyframes from the camera
try:
    camera.keyframe_delete(data_path="location")
    camera.keyframe_delete(data_path="rotation_euler")
except:
    print("Could not delete camera keyframes, possibly no keyframes exist")
# Keep this exact camera pose fixed over the full animation range.
camera.keyframe_insert(data_path="location", frame=frame_start)
camera.keyframe_insert(data_path="rotation_euler", frame=frame_start)
camera.keyframe_insert(data_path="location", frame=frame_end)
camera.keyframe_insert(data_path="rotation_euler", frame=frame_end)

scene.frame_set(current_frame)

