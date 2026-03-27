import math

import bpy
import csv
import os

import numpy as np
from mathutils import Quaternion, Vector

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
crank_angle = data[:, 11]
left_pedal_angle = data[:, 12]
right_pedal_angle = data[:, 13]
ARMATURE_NAME = "Armature"
ROOT_BONE_NAME = "b_root"
STEER_BONE_NAME = "b_steer"
REAR_WHEEL_BONE_NAME = "b_rear_wheel"
FRONT_WHEEL_BONE_NAME = "b_front_wheel"
CRANK_BONE_NAME = "b_crank"
LEFT_PEDAL_BONE_NAME = "b_left_pedal"
RIGHT_PEDAL_BONE_NAME = "b_right_pedal"

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
fps = scene.render.fps

# set frame zero to the proper location and orientation
for i in range(0, len(tx)):
    scene.frame_set(scene.frame_start + int(round(time[i] * fps)))
    ## root joint location and orientation setting
    p = Vector((float(tx[i]), float(ty[i]), float(tz[i])))
    root.location = p
    root.keyframe_insert(data_path="location")
    q = Quaternion(
        (float(rw[i]), float(rx[i]), float(ry[i]), float(rz[i]))
    )
    q.normalize()
    root.rotation_quaternion = q
    root.keyframe_insert(data_path="rotation_quaternion")
    ## steer angle 
    steer_q = Quaternion((0.0, 1.0, 0.0), -np.deg2rad(steer_angle[i]))
    steer.rotation_quaternion = steer_q
    steer.keyframe_insert(data_path="rotation_quaternion")
    ## front wheel rotation
    front_wheel_q = Quaternion((0.0, 1.0, 0.0), np.deg2rad(front_wheel_angle[i]))
    front_wheel.rotation_quaternion = front_wheel_q
    front_wheel.keyframe_insert(data_path="rotation_quaternion")
    ## rear wheel rotation
    rear_wheel_q = Quaternion((0.0, 1.0, 0.0), np.deg2rad(rear_wheel_angle[i]))
    rear_wheel.rotation_quaternion = rear_wheel_q
    rear_wheel.keyframe_insert(data_path="rotation_quaternion")
    ## crank rotation
    crank_q = Quaternion((0.0, 1.0, 0.0), np.deg2rad(crank_angle[i]))
    crank.rotation_quaternion = crank_q
    crank.keyframe_insert(data_path="rotation_quaternion")
    ## left pedal rotation
    left_pedal_q = Quaternion((0.0, 1.0, 0.0), np.deg2rad(left_pedal_angle[i]))
    left_pedal.rotation_quaternion = left_pedal_q
    left_pedal.keyframe_insert(data_path="rotation_quaternion")
    ## right pedal rotation
    right_pedal_q = Quaternion((0.0, 1.0, 0.0), np.deg2rad(right_pedal_angle[i]))
    right_pedal.rotation_quaternion = right_pedal_q
    right_pedal.keyframe_insert(data_path="rotation_quaternion")


## set the frame in animation to the frame 0
scene.frame_set(0)