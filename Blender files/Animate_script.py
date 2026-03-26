import math

import bpy
import csv
import os

import numpy as np
from mathutils import Euler, Quaternion, Vector

directory = os.path.expanduser("~/3D-bicycle-pose-estimation")
filename = "transform_data_60hz.csv"
path = os.path.join(directory, filename)

with open(path, newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)

data = np.loadtxt(path, delimiter=",", skiprows=1)
time = data[:, 0]
tx = data[:, 1]
ty = data[:, 2]
tz = data[:, 3]
rw = data[:, 4]
rx = data[:, 5]
ry = data[:, 6]
rz = data[:, 7]
rear_wheel_angle = data[:, 8]
steer_angle = data[:, 9]
front_wheel_angle = data[:, 10]

ARMATURE_NAME = "Armature"
ROOT_BONE_NAME = "b_root"

arm = bpy.data.objects[ARMATURE_NAME]
root = arm.pose.bones[ROOT_BONE_NAME]
root.rotation_mode = "QUATERNION"

scene = bpy.context.scene
fps = scene.render.fps

# set frame zero to the proper location and orientation

scene.frame_set(0)
p = Vector((float(tx[0]), float(ty[0]), float(tz[0])))
root.location = p
root.keyframe_insert(data_path="location")
q = Quaternion(
    (float(rw[0]), float(rx[0]), float(ry[0]), float(rz[0]))
)
q.normalize()
root.rotation_quaternion = q
root.keyframe_insert(data_path="rotation_quaternion")

for i in range(1, len(tx)):
    scene.frame_set(scene.frame_start + int(round(time[i] * fps)))
    p = Vector((float(tx[i]), float(ty[i]), float(tz[i])))
    root.location = p
    root.keyframe_insert(data_path="location")
    q = Quaternion(
        (float(rw[i]), float(rx[i]), float(ry[i]), float(rz[i]))
    )
    q.normalize()
    root.rotation_quaternion = q
    root.keyframe_insert(data_path="rotation_quaternion")

## set the frame in animation to the frame 0
scene.frame_set(0)