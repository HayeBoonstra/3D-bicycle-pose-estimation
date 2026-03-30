import bpy
import json
import os
from bpy_extras.object_utils import world_to_camera_view

# ----------------------------
# Config
# ----------------------------
COLLECTION_NAME = "Keypoints"
OUTPUT_PATH = bpy.path.abspath("//../2D annotated videos/2D annotations")
RAW_IMAGES_PATH = bpy.path.abspath("//2D annotated videos/raw images")

scene = bpy.context.scene
cam = scene.camera

if cam is None:
    raise RuntimeError("No active scene camera. Set scene.camera first.")

kp_col = bpy.data.collections.get(COLLECTION_NAME)
if kp_col is None:
    raise RuntimeError(f'Collection "{COLLECTION_NAME}" not found.')

# Render size in pixels (respecting render percentage)
render = scene.render
scale = render.resolution_percentage / 100.0
width = int(render.resolution_x * scale)
height = int(render.resolution_y * scale)

os.makedirs(OUTPUT_PATH, exist_ok=True)

start_frame = int(scene.frame_start)
end_frame = int(scene.frame_end)
frame_count = end_frame - start_frame + 1

for frame in range(start_frame, end_frame + 1):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    annotations = {
        "frame": frame,
        "image_width": width,
        "image_height": height,
        "camera": cam.name,
        "image_file": os.path.join(RAW_IMAGES_PATH, f"frame{frame:04d}.png"),
        "keypoints": [],
    }

    for obj in kp_col.all_objects:
        if obj.type != 'EMPTY':
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        world_co = obj_eval.matrix_world.translation

        # Returns normalized camera coords:
        # x,y in [0,1] on image plane (can be outside range), z is depth in camera space
        co_ndc = world_to_camera_view(scene, cam, world_co)

        x_px = co_ndc.x * width
        y_px = (1.0 - co_ndc.y) * height  # flip Y: Blender NDC origin is bottom-left

        visible = (
            (co_ndc.z > 0.0)
            and (0.0 <= co_ndc.x <= 1.0)
            and (0.0 <= co_ndc.y <= 1.0)
        )

        annotations["keypoints"].append(
            {
                "name": obj.name,
                "x": float(x_px),
                "y": float(y_px),
                "z_cam": float(co_ndc.z),
                "visible_in_frame": bool(visible),
            }
        )

    # Sort for deterministic output
    annotations["keypoints"].sort(key=lambda k: k["name"])

    output_file = os.path.join(OUTPUT_PATH, f"keypoints_2d_frame_{frame:04d}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"[{frame}/{end_frame}] wrote {len(annotations['keypoints'])} keypoints -> {output_file}")

print(f"Done. Wrote {frame_count} annotation files to: {OUTPUT_PATH}")