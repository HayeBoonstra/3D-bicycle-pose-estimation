import json
import os
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_TOOLS_DIR = REPO_ROOT / "data_generation_pipeline_tools"
if str(PIPELINE_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_TOOLS_DIR))
from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES, canonical_keypoint_name
from data_generation_pipeline_tools.mujoco_trajectory_dynamics import (
    assert_trajectory_frame_count,
    dynamics_gt_payload,
)
from data_generation_pipeline_tools.pipeline_clip_registry import trajectory_file_sha256

COLLECTION_NAME = "Keypoints"
BICYCLE_MESH_COLLECTION = os.environ.get("BICYCLE_BBOX_COLLECTION", "Bicycle")
PROP_OCCLUDER_COLLECTION = os.environ.get("KEYPOINT_OCCLUDER_COLLECTION", "props")
DEFAULT_CLIP_ID = "interactive_clip"


def _env_flag(name, default=True):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


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


def _camera_payload_for_frame(scene, cam, width, height):
    return {
        "camera": cam.name,
        "image_size": [width, height],
        "fps": int(scene.render.fps),
        **_camera_extrinsics(cam),
        **_camera_intrinsics(scene, cam, width, height),
        "projection_model": "opencv_pinhole",
    }


def _world_to_camera_point(cam, world_point):
    world_to_camera = cam.matrix_world.inverted()
    camera_point = world_to_camera @ world_point
    return [float(camera_point.x), float(camera_point.y), float(camera_point.z)]


def _opencv_camera_point(cam, world_point):
    """OpenCV camera coords (Z>0 in front); Blender camera space uses Z<0 in front."""
    x, y, z = _world_to_camera_point(cam, world_point)
    return [x, y, -z]


def _world_background_strength(scene):
    world = scene.world
    if world is None or world.node_tree is None:
        return None
    nodes = world.node_tree.nodes
    for node in nodes:
        if node.bl_idname == "ShaderNodeBackground":
            strength = node.inputs.get("Strength")
            if strength is not None:
                return float(strength.default_value)
    return None


def _trajectory_csv_path() -> Path:
    requested = os.environ.get("TRAJECTORY_CSV", "").strip()
    if not requested:
        raise RuntimeError(
            "TRAJECTORY_CSV is required for bicycle annotation export (MuJoCo dynamics GT)."
        )
    path = Path(requested).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV not found: {path}")
    return path


def _metadata_from_env(scene, clip_id, output_dir, trajectory_csv: Path):
    lighting_strength = os.environ.get("LIGHTING_STRENGTH")
    if lighting_strength in {None, ""}:
        world_strength = _world_background_strength(scene)
        lighting_strength = "" if world_strength is None else f"{world_strength:.6f}"
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
        "lighting_seed": os.environ.get("LIGHTING_SEED", ""),
        "lighting_strength": lighting_strength,
        "output_dir": str(output_dir),
        "trajectory_csv": str(trajectory_csv),
        "dynamics_fields": ["steer_deg", "roll_deg"],
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


def _occluder_meshes(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return []
    return [obj for obj in collection.all_objects if obj.type == "MESH"]


def _occluded_by_props(scene, depsgraph, cam_location, world_point, occluders):
    if not occluders:
        return False

    direction = world_point - cam_location
    distance = direction.length
    if distance <= 1e-6:
        return False
    direction.normalize()

    hit, _loc, _normal, _face, hit_obj, _matrix = scene.ray_cast(
        depsgraph,
        cam_location,
        direction,
        distance=distance - 1e-4,
    )
    if not hit or hit_obj is None:
        return False
    occluder_names = {obj.name for obj in occluders}
    return hit_obj.name in occluder_names


def _gt_bbox_xywh_from_bicycle_meshes(scene, cam, depsgraph, width, height, collection_name):
    """Axis-aligned 2D bbox in pixel space from Bicycle collection mesh bound boxes.

    Uses world-space corners of each mesh's bound_box projected through the camera.
    Corners behind the camera (z <= 0 in normalized device coords) are skipped.
    Returns None if the collection is missing or no in-front corners were found.
    """
    col = bpy.data.collections.get(collection_name)
    if col is None:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for obj in col.all_objects:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        matrix_world = obj_eval.matrix_world
        for corner in obj_eval.bound_box:
            world_co = matrix_world @ Vector(corner)
            co_ndc = world_to_camera_view(scene, cam, world_co)
            if co_ndc.z <= 0.0:
                continue
            xs.append(float(co_ndc.x * width))
            ys.append(float((1.0 - co_ndc.y) * height))

    if not xs:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, max(1.0, x_max - x_min), max(1.0, y_max - y_min)]


def export_annotations():
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise RuntimeError("No active scene camera. Set scene.camera first.")

    quiet_mode = _env_flag("QUIET_MODE", default=True)

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
    occluder_objects = _occluder_meshes(PROP_OCCLUDER_COLLECTION)
    missing = sorted(set(BICYCLE_KEYPOINT_NAMES) - set(keypoint_objects))
    if missing and not quiet_mode:
        print(f"[annotation-export] Warning: missing keypoint empties: {missing}")

    start_frame = int(scene.frame_start)
    end_frame = int(scene.frame_end)
    frame_count = end_frame - start_frame + 1
    trajectory_csv = _trajectory_csv_path()
    trajectory_rows = assert_trajectory_frame_count(trajectory_csv, frame_count)
    render_config = _metadata_from_env(scene, clip_id, out_root, trajectory_csv)
    render_config["trajectory_sha256"] = trajectory_file_sha256(trajectory_csv)
    render_config["missing_keypoints"] = missing

    # Legacy single-file camera (first exported frame). Per-frame K/R/t live in keypoints_3d.jsonl.
    scene.frame_set(start_frame)
    bpy.context.view_layer.update()
    camera_payload = _camera_payload_for_frame(scene, cam, width, height)
    camera_payload["note"] = (
        "Static snapshot at clip frame_start; tracking cameras use per-frame K/R/t in keypoints_3d.jsonl."
    )

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
            frame_camera = _camera_payload_for_frame(scene, cam, width, height)
            keypoints_3d = {
                "clip_id": clip_id,
                "scene_id": render_config["scene_id"],
                "frame": frame,
                "frame_index": frame - start_frame,
                "timestamp_sec": float((frame - start_frame) / max(1, int(scene.render.fps))),
                "fps": int(scene.render.fps),
                "camera_name": cam.name,
                "K": frame_camera["K"],
                "R": frame_camera["R"],
                "t": frame_camera["t"],
                "coord_frames": {
                    "world": "blender_world_xyz_meters",
                    "camera": "opencv_camera_xyz_meters",
                },
                "joint_names": list(BICYCLE_KEYPOINT_NAMES),
                "kps_world": [],
                "kps_camera": [],
                "valid_3d": [],
                "missing_mask": [],
                "occluded_mask": [],
                "in_front_mask": [],
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
                            "occluded_by_prop": False,
                            "v": 0,
                            "missing": True,
                        }
                    )
                    keypoints_3d["keypoints"].append(
                        {"name": keypoint_name, "world": None, "camera": None, "missing": True}
                    )
                    keypoints_3d["kps_world"].append(None)
                    keypoints_3d["kps_camera"].append(None)
                    keypoints_3d["valid_3d"].append(0)
                    keypoints_3d["missing_mask"].append(1)
                    keypoints_3d["occluded_mask"].append(0)
                    keypoints_3d["in_front_mask"].append(0)
                    continue

                obj_eval = obj.evaluated_get(depsgraph)
                world_co = obj_eval.matrix_world.translation
                camera_xyz = _opencv_camera_point(cam, world_co)
                co_ndc = world_to_camera_view(scene, cam, world_co)
                x_px = co_ndc.x * width
                y_px = (1.0 - co_ndc.y) * height
                in_front = co_ndc.z > 0.0
                in_frame = in_front and (0.0 <= co_ndc.x <= 1.0) and (0.0 <= co_ndc.y <= 1.0)
                occluded_by_prop = in_frame and _occluded_by_props(
                    scene, depsgraph, cam.matrix_world.translation, world_co, occluder_objects
                )
                if not in_frame:
                    visibility = 0
                elif occluded_by_prop:
                    visibility = 1
                else:
                    visibility = 2

                annotations["keypoints"].append(
                    {
                        "name": keypoint_name,
                        "x": float(x_px),
                        "y": float(y_px),
                        "z_cam": float(co_ndc.z),
                        "in_front_of_camera": bool(in_front),
                        "visible_in_frame": bool(in_frame),
                        "occluded_by_prop": bool(occluded_by_prop),
                        "v": int(visibility),
                        "missing": False,
                    }
                )
                keypoints_3d["keypoints"].append(
                    {
                        "name": keypoint_name,
                        "world": _vector_to_list(world_co),
                        "camera": camera_xyz,
                        "missing": False,
                    }
                )
                keypoints_3d["kps_world"].append(_vector_to_list(world_co))
                keypoints_3d["kps_camera"].append(camera_xyz)
                keypoints_3d["valid_3d"].append(1)
                keypoints_3d["missing_mask"].append(0)
                keypoints_3d["occluded_mask"].append(1 if occluded_by_prop else 0)
                keypoints_3d["in_front_mask"].append(1 if in_front else 0)

            mesh_bbox = _gt_bbox_xywh_from_bicycle_meshes(
                scene, cam, depsgraph, width, height, BICYCLE_MESH_COLLECTION
            )
            if mesh_bbox is not None:
                annotations["gt_bbox_xywh"] = mesh_bbox
                annotations["gt_bbox_source"] = "bicycle_mesh"
                annotations["gt_bbox_collection"] = BICYCLE_MESH_COLLECTION

            output_file = annotation_dir / f"keypoints_2d_frame_{frame:04d}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2)
            if not (
                len(keypoints_3d["joint_names"])
                == len(keypoints_3d["kps_world"])
                == len(keypoints_3d["kps_camera"])
                == len(keypoints_3d["valid_3d"])
                == len(keypoints_3d["missing_mask"])
                == len(keypoints_3d["occluded_mask"])
                == len(keypoints_3d["in_front_mask"])
                == len(BICYCLE_KEYPOINT_NAMES)
            ):
                raise RuntimeError("3D keypoint export alignment mismatch.")
            frame_index = frame - start_frame
            keypoints_3d["dynamics_gt"] = dynamics_gt_payload(
                trajectory_csv,
                frame_index,
                trajectory_rows=trajectory_rows,
            )
            jsonl.write(json.dumps(keypoints_3d) + "\n")
            frame_idx = frame - start_frame + 1
            progress = (
                f"\r[annotation-export] {frame_idx}/{frame_count} "
                f"frames processed"
            )
            if not quiet_mode:
                print(progress, end="", flush=True)

    if not quiet_mode:
        print()
        print(f"Done. Wrote {frame_count} annotation files to: {annotation_dir}")


export_annotations()