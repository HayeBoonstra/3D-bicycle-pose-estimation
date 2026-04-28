import json
import os
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bicycle_keypoint_schema import (  # noqa: E402
    BICYCLE_KEYPOINT_NAMES,
    canonical_keypoint_name,
)

COLLECTION_NAME = "Keypoints"
DEFAULT_CLIP_ID = "interactive_clip"


def _matrix_to_list(matrix):
    return [[float(value) for value in row] for row in matrix]


def _vector_to_list(vector):
    return [float(value) for value in vector]


def _render_size(scene):
    render = scene.render
    scale = render.resolution_percentage / 100.0
    return int(render.resolution_x * scale), int(render.resolution_y * scale)


def _camera_intrinsics(scene, cam, width, height):
    """Approximate Blender camera settings as a pinhole intrinsic matrix."""
    cam_data = cam.data
    if cam_data.type != "PERSP":
        raise RuntimeError("Only perspective cameras are supported for keypoint export.")

    sensor_fit = cam_data.sensor_fit
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= height else "VERTICAL"

    if sensor_fit == "VERTICAL":
        focal_px = cam_data.lens / cam_data.sensor_height * height
    else:
        focal_px = cam_data.lens / cam_data.sensor_width * width

    pixel_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    fx = focal_px
    fy = focal_px * pixel_aspect
    cx = width * (0.5 - cam_data.shift_x)
    cy = height * (0.5 + cam_data.shift_y)

    return {
        "K": [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        "lens_mm": float(cam_data.lens),
        "sensor_width_mm": float(cam_data.sensor_width),
        "sensor_height_mm": float(cam_data.sensor_height),
        "sensor_fit": sensor_fit,
        "shift_x": float(cam_data.shift_x),
        "shift_y": float(cam_data.shift_y),
    }


def _camera_extrinsics(cam):
    world_to_camera = cam.matrix_world.inverted()
    return {
        "R": [[float(world_to_camera[row][col]) for col in range(3)] for row in range(3)],
        "t": [float(world_to_camera[row][3]) for row in range(3)],
        "camera_to_world": _matrix_to_list(cam.matrix_world),
        "world_to_camera": _matrix_to_list(world_to_camera),
    }


def _metadata_from_env(scene, clip_id, output_dir):
    return {
        "clip_id": clip_id,
        "scene_id": os.environ.get("SCENE_ID", ""),
        "blend_file": bpy.data.filepath,
        "bike": os.environ.get("BIKE_TAG", ""),
        "rider": os.environ.get("RIDER_TAG", ""),
        "fps": int(scene.render.fps),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "camera_seed": os.environ.get("CAMERA_SEED", ""),
        "camera_target": os.environ.get("CAMERA_TARGET", "k_handlebar_middle"),
        "output_dir": str(output_dir),
    }


def _keypoint_objects(collection):
    objects = {}
    for obj in collection.all_objects:
        if obj.type != "EMPTY":
            continue
        canonical_name = canonical_keypoint_name(obj.name)
        if canonical_name in BICYCLE_KEYPOINT_NAMES:
            objects[canonical_name] = obj
    return objects


def export_annotations():
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise RuntimeError("No active scene camera. Set scene.camera first.")

    kp_col = bpy.data.collections.get(COLLECTION_NAME)
    if kp_col is None:
        raise RuntimeError(f'Collection "{COLLECTION_NAME}" not found.')

    clip_id = os.environ.get("CLIP_ID", DEFAULT_CLIP_ID)
    out_root = Path(
        os.environ.get("CLIP_OUTPUT_DIR")
        or os.environ.get("RAW_RENDERS_DIR")
        or (REPO_ROOT / "raw_renders" / clip_id)
    )
    frames_dir = out_root / "frames"
    annotation_dir = out_root / "per_frame_annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    width, height = _render_size(scene)
    keypoint_objects = _keypoint_objects(kp_col)
    missing = sorted(set(BICYCLE_KEYPOINT_NAMES) - set(keypoint_objects))
    if missing:
        print(f"[annotation-export] Warning: missing keypoint empties: {missing}")

    start_frame = int(scene.frame_start)
    end_frame = int(scene.frame_end)
    frame_count = end_frame - start_frame + 1
    render_config = _metadata_from_env(scene, clip_id, out_root)
    render_config["missing_keypoints"] = missing

    camera_payload = {
        "camera": cam.name,
        "image_size": [width, height],
        "fps": int(scene.render.fps),
        **_camera_extrinsics(cam),
        **_camera_intrinsics(scene, cam, width, height),
    }

    with (out_root / "camera.json").open("w", encoding="utf-8") as f:
        json.dump(camera_payload, f, indent=2)
    with (out_root / "render_config.json").open("w", encoding="utf-8") as f:
        json.dump(render_config, f, indent=2)

    keypoints_3d_path = out_root / "keypoints_3d.jsonl"
    with keypoints_3d_path.open("w", encoding="utf-8") as jsonl:
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            annotations = {
                "clip_id": clip_id,
                "scene_id": render_config["scene_id"],
                "frame": frame,
                "frame_index": frame - start_frame,
                "image_width": width,
                "image_height": height,
                "camera": cam.name,
                "image_file": str(Path("frames") / f"frame_{frame:04d}.png"),
                "keypoints": [],
            }
            keypoints_3d = {
                "clip_id": clip_id,
                "scene_id": render_config["scene_id"],
                "frame": frame,
                "frame_index": frame - start_frame,
                "kps_world": [],
                "keypoints": [],
            }

            for keypoint_name in BICYCLE_KEYPOINT_NAMES:
                obj = keypoint_objects.get(keypoint_name)
                if obj is None:
                    annotations["keypoints"].append(
                        {
                            "name": keypoint_name,
                            "x": 0.0,
                            "y": 0.0,
                            "z_cam": 0.0,
                            "in_front_of_camera": False,
                            "visible_in_frame": False,
                            "missing": True,
                        }
                    )
                    keypoints_3d["keypoints"].append(
                        {"name": keypoint_name, "world": None, "missing": True}
                    )
                    keypoints_3d["kps_world"].append(None)
                    continue

                obj_eval = obj.evaluated_get(depsgraph)
                world_co = obj_eval.matrix_world.translation
                co_ndc = world_to_camera_view(scene, cam, world_co)
                x_px = co_ndc.x * width
                y_px = (1.0 - co_ndc.y) * height
                in_front = co_ndc.z > 0.0
                visible = in_front and (0.0 <= co_ndc.x <= 1.0) and (0.0 <= co_ndc.y <= 1.0)

                annotations["keypoints"].append(
                    {
                        "name": keypoint_name,
                        "x": float(x_px),
                        "y": float(y_px),
                        "z_cam": float(co_ndc.z),
                        "in_front_of_camera": bool(in_front),
                        "visible_in_frame": bool(visible),
                        "missing": False,
                    }
                )
                keypoints_3d["keypoints"].append(
                    {
                        "name": keypoint_name,
                        "world": _vector_to_list(world_co),
                        "missing": False,
                    }
                )
                keypoints_3d["kps_world"].append(_vector_to_list(world_co))

            output_file = annotation_dir / f"keypoints_2d_frame_{frame:04d}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2)
            jsonl.write(json.dumps(keypoints_3d) + "\n")
            print(
                f"[{frame}/{end_frame}] wrote {len(annotations['keypoints'])} "
                f"keypoints -> {output_file}"
            )

    print(f"Done. Wrote {frame_count} annotation files to: {annotation_dir}")


export_annotations()