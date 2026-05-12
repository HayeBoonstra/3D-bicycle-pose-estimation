"""Generate PoseMamba raw annotations directly from MuJoCo bicycle states.

The exporter mirrors the Blender raw dataset contract:

    raw_root/clip_id/
      camera.json
      keypoints_3d.jsonl
      per_frame_annotations/keypoints_2d_frame_XXXX.json

Those folders can be consumed by ``3d_keypoint_detector_training/build_sequences.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_pipeline_tools.bicycle_keypoint_schema import BICYCLE_KEYPOINT_NAMES, KEYPOINT_INDEX
from keypoint_detector_pipeline.world_transform import CameraModel, project_to_image, reprojection_rmse, world_to_camera


@dataclass(frozen=True)
class VirtualCamera:
    name: str
    width: int
    height: int
    position: np.ndarray
    target: np.ndarray
    model: CameraModel
    focal_length_px: float
    fov_deg: float


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def look_at_camera(
    position: np.ndarray,
    target: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float,
    name: str = "mujoco_virtual_camera",
) -> VirtualCamera:
    """Create OpenCV-style camera intrinsics/extrinsics looking at a target."""

    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = _unit(target - position)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    right = _unit(np.cross(forward, world_up))
    down = _unit(np.cross(forward, right))
    R = np.stack([right, down, forward], axis=0).astype(np.float32)
    t = (-R @ position).astype(np.float32)

    f = 0.5 * float(width) / math.tan(math.radians(float(fov_deg)) * 0.5)
    K = np.asarray(
        [
            [f, 0.0, width * 0.5],
            [0.0, f, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return VirtualCamera(
        name=name,
        width=int(width),
        height=int(height),
        position=position,
        target=target,
        model=CameraModel(K=K, R=R, t=t),
        focal_length_px=float(f),
        fov_deg=float(fov_deg),
    )


def sample_virtual_cameras(
    target: np.ndarray,
    *,
    points_world_by_frame: np.ndarray,
    count: int,
    seed: int,
    width: int,
    height: int,
    min_fov_deg: float,
    max_fov_deg: float,
    min_distance: float,
    max_distance: float,
    min_elevation_deg: float,
    max_elevation_deg: float,
    min_visible_keypoints: int,
    min_visible_frame_ratio: float,
    max_tries_per_camera: int,
    fit_margin: float,
) -> list[VirtualCamera]:
    rng = np.random.default_rng(seed)
    cameras: list[VirtualCamera] = []
    target = np.asarray(target, dtype=np.float32)
    points_world_by_frame = np.asarray(points_world_by_frame, dtype=np.float32)
    radius = float(np.max(np.linalg.norm(points_world_by_frame.reshape(-1, 3) - target[None, :], axis=1)))
    for idx in range(count):
        accepted: VirtualCamera | None = None
        for _ in range(max_tries_per_camera):
            distance = float(rng.uniform(min_distance, max_distance))
            azimuth = float(rng.uniform(0.0, math.tau))
            elevation = float(rng.uniform(math.radians(min_elevation_deg), math.radians(max_elevation_deg)))
            fov_deg = float(rng.uniform(min_fov_deg, max_fov_deg))
            focal_px = 0.5 * float(width) / math.tan(math.radians(fov_deg) * 0.5)
            vertical_fov = 2.0 * math.atan(float(height) / (2.0 * focal_px))
            required_distance = radius / max(1e-6, math.tan(vertical_fov * 0.5)) * fit_margin
            distance = max(distance, required_distance)
            offset = np.asarray(
                [
                    distance * math.cos(elevation) * math.cos(azimuth),
                    distance * math.cos(elevation) * math.sin(azimuth),
                    distance * math.sin(elevation),
                ],
                dtype=np.float32,
            )
            position = target + offset
            position[2] = max(position[2], 0.25)
            candidate = look_at_camera(
                position,
                target,
                width=width,
                height=height,
                fov_deg=fov_deg,
                name=f"mujoco_virtual_camera_{idx:02d}",
            )
            if _camera_visibility_ok(
                candidate,
                points_world_by_frame,
                min_visible_keypoints=min_visible_keypoints,
                min_visible_frame_ratio=min_visible_frame_ratio,
            ):
                accepted = candidate
                break
        if accepted is None:
            raise RuntimeError(
                "Failed to sample a valid camera for the full trajectory. "
                "Try increasing FOV, camera distance, or lowering visibility thresholds."
            )
        cameras.append(accepted)
    return cameras


def _camera_visibility_ok(
    camera: VirtualCamera,
    points_world_by_frame: np.ndarray,
    *,
    min_visible_keypoints: int,
    min_visible_frame_ratio: float,
) -> bool:
    visible_frame_count = 0
    for points_world in points_world_by_frame:
        points_cam = world_to_camera(points_world, camera.model)
        points_2d = project_to_image(points_cam, camera.model)
        in_front = points_cam[:, 2] > 1e-6
        in_frame = (
            in_front
            & (points_2d[:, 0] >= 0.0)
            & (points_2d[:, 0] <= camera.width)
            & (points_2d[:, 1] >= 0.0)
            & (points_2d[:, 1] <= camera.height)
        )
        if int(np.count_nonzero(in_frame)) >= min_visible_keypoints:
            visible_frame_count += 1
    return visible_frame_count / max(1, points_world_by_frame.shape[0]) >= min_visible_frame_ratio


def _camera_json(camera: VirtualCamera, fps: int) -> dict:
    return {
        "camera": camera.name,
        "image_size": [camera.width, camera.height],
        "fps": int(fps),
        "K": camera.model.K.tolist(),
        "R": camera.model.R.tolist(),
        "t": camera.model.t.tolist(),
        "camera_to_world": _camera_to_world_matrix(camera).tolist(),
        "world_to_camera": _world_to_camera_matrix(camera).tolist(),
        "focal_length_px": camera.focal_length_px,
        "fov_deg": camera.fov_deg,
        "projection_model": "opencv_pinhole",
    }


def _world_to_camera_matrix(camera: VirtualCamera) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = camera.model.R
    mat[:3, 3] = camera.model.t
    return mat


def _camera_to_world_matrix(camera: VirtualCamera) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = camera.model.R.T
    mat[:3, 3] = camera.position
    return mat


def _load_world_model():
    import mujoco

    generator_dir = Path(__file__).resolve().parent
    if str(generator_dir) not in sys.path:
        sys.path.insert(0, str(generator_dir))
    from world_contructor import World

    world = World()
    model = mujoco.MjModel.from_xml_string(world.create_world())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _rotation_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(1e-12, float(np.linalg.norm(quat)))
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _local_bicycle_keypoints(state: dict[str, float]) -> np.ndarray:
    generator_dir = Path(__file__).resolve().parent
    if str(generator_dir) not in sys.path:
        sys.path.insert(0, str(generator_dir))
    from bicycle_constructor import Bicycle

    bicycle = Bicycle()
    bicycle.create_bicycle_variables()
    radius = bicycle.wheel_size / 2.0
    fork_top = bicycle.front_hub + np.asarray([0.0, 0.0, bicycle.fork_length + bicycle.handlebar_height])
    crank = math.radians(float(state.get("crank_angle", 0.0)))
    left_pedal = np.asarray(
        [
            0.0,
            -bicycle.crank_width - bicycle.pedal_width / 2.0 - 4.0 * 0.016,
            -bicycle.crank_length,
        ],
        dtype=np.float32,
    )
    right_pedal = np.asarray(
        [
            0.0,
            bicycle.crank_width + bicycle.pedal_width / 2.0 + 4.0 * 0.016,
            bicycle.crank_length,
        ],
        dtype=np.float32,
    )
    crank_rot = np.asarray(
        [
            [math.cos(crank), 0.0, math.sin(crank)],
            [0.0, 1.0, 0.0],
            [-math.sin(crank), 0.0, math.cos(crank)],
        ],
        dtype=np.float32,
    )
    points_by_name = {
        "k_bottom_bracket": bicycle.bottom_bracket,
        "k_seat_stay": bicycle.seat_stay_attachment,
        "k_saddle": bicycle.seat_tube_post,
        "k_upper_head_tube": fork_top,
        "k_lower_head_tube": bicycle.head_tube,
        "k_handlebar_left": fork_top + np.asarray([0.0, -bicycle.handlebar_width / 2.0, 0.0]),
        "k_handlebar_middle": fork_top,
        "k_handlebar_right": fork_top + np.asarray([0.0, bicycle.handlebar_width / 2.0, 0.0]),
        "k_front_hub_left": bicycle.front_hub + np.asarray([0.0, -bicycle.wheel_clearance, 0.0]),
        "k_front_hub_right": bicycle.front_hub + np.asarray([0.0, bicycle.wheel_clearance, 0.0]),
        "k_front_wheel_back": bicycle.front_hub + np.asarray([-radius, 0.0, 0.0]),
        "k_front_wheel_front": bicycle.front_hub + np.asarray([radius, 0.0, 0.0]),
        "k_front_wheel_ground": bicycle.front_hub + np.asarray([0.0, 0.0, -radius]),
        "k_rear_hub_left": bicycle.rear_hub + np.asarray([0.0, -bicycle.wheel_clearance, 0.0]),
        "k_rear_hub_right": bicycle.rear_hub + np.asarray([0.0, bicycle.wheel_clearance, 0.0]),
        "k_rear_wheel_ground": bicycle.rear_hub + np.asarray([0.0, 0.0, -radius]),
        "k_left_pedal": crank_rot @ left_pedal,
        "k_right_pedal": crank_rot @ right_pedal,
    }
    points = np.zeros((len(BICYCLE_KEYPOINT_NAMES), 3), dtype=np.float32)
    for name, value in points_by_name.items():
        points[KEYPOINT_INDEX[name]] = np.asarray(value, dtype=np.float32)
    return points


def _fallback_keypoints_from_states(states: list[dict[str, float]]) -> np.ndarray:
    frames = []
    for state in states:
        local = _local_bicycle_keypoints(state)
        quat = np.asarray(
            [
                state.get("rw", 1.0),
                state.get("rx", 0.0),
                state.get("ry", 0.0),
                state.get("rz", 0.0),
            ],
            dtype=np.float32,
        )
        R = _rotation_from_quat_wxyz(quat)
        translation = np.asarray(
            [state.get("tx", 0.0), state.get("ty", 0.0), state.get("tz", 0.0) + 0.35],
            dtype=np.float32,
        )
        frames.append((R @ local.T).T + translation[None, :])
    return np.asarray(frames, dtype=np.float32)


def _site_ids(model, mujoco_module) -> dict[str, int]:
    ids: dict[str, int] = {}
    missing: list[str] = []
    for name in BICYCLE_KEYPOINT_NAMES:
        site_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            missing.append(name)
        else:
            ids[name] = int(site_id)
    if missing:
        raise RuntimeError(f"MuJoCo model is missing canonical keypoint sites: {missing}")
    return ids


def _extract_site_keypoints(data, site_ids: dict[str, int]) -> np.ndarray:
    points = np.zeros((len(BICYCLE_KEYPOINT_NAMES), 3), dtype=np.float32)
    for name, site_id in site_ids.items():
        points[KEYPOINT_INDEX[name]] = np.asarray(data.site_xpos[site_id], dtype=np.float32)
    return points


def _quat_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    half = float(yaw_rad) * 0.5
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _procedural_states(
    frame_count: int,
    fps: int,
    pattern: str,
    *,
    seed: int,
    min_speed_mps: float,
    max_speed_mps: float,
    min_crank_hz: float,
    max_crank_hz: float,
    min_turn_rate_deg: float,
    max_turn_rate_deg: float,
    min_sine_yaw_deg: float,
    max_sine_yaw_deg: float,
    min_sine_frequency_hz: float,
    max_sine_frequency_hz: float,
) -> list[dict[str, float]]:
    states: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    speed_mps = float(rng.uniform(min_speed_mps, max_speed_mps))
    crank_hz = float(rng.uniform(min_crank_hz, max_crank_hz))
    turn_rate_mag = math.radians(float(rng.uniform(min_turn_rate_deg, max_turn_rate_deg)))
    sine_yaw_amp = math.radians(float(rng.uniform(min_sine_yaw_deg, max_sine_yaw_deg)))
    sine_frequency_hz = float(rng.uniform(min_sine_frequency_hz, max_sine_frequency_hz))
    sine_phase = float(rng.uniform(0.0, math.tau))
    initial_yaw = float(rng.uniform(-math.pi, math.pi))
    turn_rate = {"straight": 0.0, "left": turn_rate_mag, "right": -turn_rate_mag, "sine": 0.0, "zigzag": 0.0}[pattern]
    x, y, yaw = 0.0, 0.0, initial_yaw
    for frame_idx in range(frame_count):
        t = frame_idx / float(fps)
        if pattern == "sine":
            yaw = initial_yaw + sine_yaw_amp * math.sin(math.tau * sine_frequency_hz * t + sine_phase)
        elif pattern == "zigzag":
            # Alternating smooth turns create sharper heading changes than the sine path.
            yaw = initial_yaw + sine_yaw_amp * math.asin(math.sin(math.tau * sine_frequency_hz * t + sine_phase)) * 2.0 / math.pi
        elif frame_idx > 0:
            yaw += turn_rate / float(fps)
        if frame_idx > 0:
            x += speed_mps * math.cos(yaw) / float(fps)
            y += speed_mps * math.sin(yaw) / float(fps)
        crank = t * math.tau * crank_hz
        states.append(
            {
                "tx": x,
                "ty": y,
                "tz": 0.0,
                "rw": _quat_wxyz_from_yaw(yaw)[0],
                "rx": 0.0,
                "ry": 0.0,
                "rz": _quat_wxyz_from_yaw(yaw)[3],
                "rear_wheel_angle": math.degrees(crank * 1.4),
                "steer_angle": math.degrees(0.18 * math.sin(t * 1.2) if pattern in {"sine", "zigzag"} else turn_rate),
                "front_wheel_angle": math.degrees(crank * 1.4),
                "crank_angle": math.degrees(crank),
                "left_pedal_angle": math.degrees(crank),
                "right_pedal_angle": -math.degrees(crank),
            }
        )
    return states


def _load_transform_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value not in {None, ""}})
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")
    return rows


def _apply_state(data, state: dict[str, float]) -> None:
    qpos = data.qpos
    qpos[0] = state.get("tx", qpos[0])
    qpos[1] = state.get("ty", qpos[1])
    qpos[2] = state.get("tz", qpos[2])
    if {"rw", "rx", "ry", "rz"}.issubset(state):
        qpos[3:7] = [state["rw"], state["rx"], state["ry"], state["rz"]]
    for idx, key in (
        (7, "rear_wheel_angle"),
        (8, "steer_angle"),
        (9, "front_wheel_angle"),
        (10, "crank_angle"),
        (11, "left_pedal_angle"),
        (12, "right_pedal_angle"),
    ):
        if idx < qpos.shape[0] and key in state:
            qpos[idx] = math.radians(float(state[key]))


def _bbox_from_points(points_2d: np.ndarray, visible: np.ndarray, width: int, height: int, margin: float) -> list[float]:
    if np.any(visible):
        selected = points_2d[visible]
        x_min, y_min = np.min(selected, axis=0)
        x_max, y_max = np.max(selected, axis=0)
    else:
        x_min, y_min, x_max, y_max = 0.0, 0.0, float(width), float(height)
    x_min = max(0.0, float(x_min) - margin)
    y_min = max(0.0, float(y_min) - margin)
    x_max = min(float(width), float(x_max) + margin)
    y_max = min(float(height), float(y_max) + margin)
    return [x_min, y_min, max(1.0, x_max - x_min), max(1.0, y_max - y_min)]


def _write_clip(
    *,
    clip_dir: Path,
    clip_id: str,
    scene_id: str,
    fps: int,
    camera: VirtualCamera,
    points_world_by_frame: np.ndarray,
) -> dict[str, float]:
    annotation_dir = clip_dir / "per_frame_annotations"
    frames_dir = clip_dir / "frames"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with (clip_dir / "camera.json").open("w", encoding="utf-8") as f:
        json.dump(_camera_json(camera, fps), f, indent=2)
    with (clip_dir / "render_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_id": clip_id,
                "scene_id": scene_id,
                "source": "mujoco_direct",
                "fps": int(fps),
                "frame_start": 0,
                "frame_end": int(points_world_by_frame.shape[0] - 1),
                "output_dir": str(clip_dir),
            },
            f,
            indent=2,
        )

    rmses: list[float] = []
    visible_counts: list[int] = []
    keypoints_3d_path = clip_dir / "keypoints_3d.jsonl"
    with keypoints_3d_path.open("w", encoding="utf-8") as jsonl:
        for frame_idx, points_world in enumerate(points_world_by_frame):
            points_cam = world_to_camera(points_world, camera.model)
            points_2d = project_to_image(points_cam, camera.model)
            in_front = points_cam[:, 2] > 1e-6
            in_frame = (
                in_front
                & (points_2d[:, 0] >= 0.0)
                & (points_2d[:, 0] <= camera.width)
                & (points_2d[:, 1] >= 0.0)
                & (points_2d[:, 1] <= camera.height)
            )
            bbox_xywh = _bbox_from_points(points_2d, in_frame, camera.width, camera.height, margin=12.0)
            visible_counts.append(int(np.count_nonzero(in_frame)))
            rmses.append(reprojection_rmse(points_cam, points_2d, camera.model))

            annotations = {
                "clip_id": clip_id,
                "scene_id": scene_id,
                "frame": frame_idx,
                "frame_index": frame_idx,
                "image_width": camera.width,
                "image_height": camera.height,
                "camera": camera.name,
                "image_file": str(Path("frames") / f"frame_{frame_idx:04d}.png"),
                "keypoints": [],
                "gt_bbox_xywh": bbox_xywh,
                "gt_bbox_source": "projected_mujoco_keypoints",
            }
            keypoints_3d = {
                "clip_id": clip_id,
                "scene_id": scene_id,
                "frame": frame_idx,
                "frame_index": frame_idx,
                "timestamp_sec": float(frame_idx / max(1, int(fps))),
                "fps": int(fps),
                "camera_name": camera.name,
                "coord_frames": {
                    "world": "mujoco_world_xyz_meters",
                    "camera": "opencv_camera_xyz_meters",
                },
                "joint_names": list(BICYCLE_KEYPOINT_NAMES),
                "kps_world": points_world.astype(float).tolist(),
                "kps_camera": points_cam.astype(float).tolist(),
                "valid_3d": [1] * len(BICYCLE_KEYPOINT_NAMES),
                "missing_mask": [0] * len(BICYCLE_KEYPOINT_NAMES),
                "occluded_mask": [0] * len(BICYCLE_KEYPOINT_NAMES),
                "in_front_mask": [1 if value else 0 for value in in_front.tolist()],
                "keypoints": [],
            }
            for joint_idx, name in enumerate(BICYCLE_KEYPOINT_NAMES):
                visibility = 2 if bool(in_frame[joint_idx]) else 0
                kp2d = {
                    "name": name,
                    "x": float(points_2d[joint_idx, 0]),
                    "y": float(points_2d[joint_idx, 1]),
                    "z_cam": float(points_cam[joint_idx, 2]),
                    "in_front_of_camera": bool(in_front[joint_idx]),
                    "visible_in_frame": bool(in_frame[joint_idx]),
                    "occluded_by_prop": False,
                    "v": int(visibility),
                    "missing": False,
                }
                annotations["keypoints"].append(kp2d)
                keypoints_3d["keypoints"].append(
                    {
                        "name": name,
                        "world": points_world[joint_idx].astype(float).tolist(),
                        "camera": points_cam[joint_idx].astype(float).tolist(),
                        "missing": False,
                    }
                )

            with (annotation_dir / f"keypoints_2d_frame_{frame_idx:04d}.json").open("w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2)
            jsonl.write(json.dumps(keypoints_3d) + "\n")

    return {
        "reprojection_rmse": float(max(rmses) if rmses else 0.0),
        "min_visible_keypoints": float(min(visible_counts) if visible_counts else 0),
        "mean_visible_keypoints": float(np.mean(visible_counts) if visible_counts else 0.0),
    }


def generate_annotations(args: argparse.Namespace) -> None:
    if args.trajectory_csv is not None:
        states = _load_transform_csv(args.trajectory_csv)
    else:
        states = _procedural_states(
            args.frames,
            args.fps,
            args.pattern,
            seed=args.trajectory_seed if args.trajectory_seed is not None else args.seed,
            min_speed_mps=args.min_speed_mps,
            max_speed_mps=args.max_speed_mps,
            min_crank_hz=args.min_crank_hz,
            max_crank_hz=args.max_crank_hz,
            min_turn_rate_deg=args.min_turn_rate_deg,
            max_turn_rate_deg=args.max_turn_rate_deg,
            min_sine_yaw_deg=args.min_sine_yaw_deg,
            max_sine_yaw_deg=args.max_sine_yaw_deg,
            min_sine_frequency_hz=args.min_sine_frequency_hz,
            max_sine_frequency_hz=args.max_sine_frequency_hz,
        )

    try:
        mujoco, model, data = _load_world_model()
    except ModuleNotFoundError as exc:
        if exc.name != "mujoco":
            raise
        points_world = _fallback_keypoints_from_states(states)
        source_backend = "geometry_fallback"
    else:
        site_ids = _site_ids(model, mujoco)
        points_world_by_frame = []
        for state in states:
            _apply_state(data, state)
            mujoco.mj_forward(model, data)
            points_world_by_frame.append(_extract_site_keypoints(data, site_ids))
        points_world = np.asarray(points_world_by_frame, dtype=np.float32)
        source_backend = "mujoco_site_xpos"

    target_idx = KEYPOINT_INDEX[args.camera_target]
    target = np.mean(points_world[:, target_idx, :], axis=0)
    if args.fit_camera_to_motion:
        target = np.mean(points_world.reshape(-1, 3), axis=0)
    cameras = sample_virtual_cameras(
        target,
        points_world_by_frame=points_world,
        count=args.num_cameras,
        seed=args.seed,
        width=args.width,
        height=args.height,
        min_fov_deg=args.min_fov_deg,
        max_fov_deg=args.max_fov_deg,
        min_distance=args.min_camera_distance,
        max_distance=args.max_camera_distance,
        min_elevation_deg=args.min_camera_elevation_deg,
        max_elevation_deg=args.max_camera_elevation_deg,
        min_visible_keypoints=args.min_visible_keypoints,
        min_visible_frame_ratio=args.min_visible_frame_ratio,
        max_tries_per_camera=args.camera_max_tries,
        fit_margin=args.camera_fit_margin,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cam_idx, camera in enumerate(cameras):
        clip_id = f"{args.clip_id}_cam{cam_idx:02d}" if args.num_cameras > 1 else args.clip_id
        clip_dir = args.out / clip_id
        stats = _write_clip(
            clip_dir=clip_dir,
            clip_id=clip_id,
            scene_id=args.scene_id,
            fps=args.fps,
            camera=camera,
            points_world_by_frame=points_world,
        )
        manifest.append({"clip_id": clip_id, "clip_dir": str(clip_dir), **stats})

    with (args.out / "mujoco_direct_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "mujoco_direct",
                "backend": source_backend,
                "joint_names": list(BICYCLE_KEYPOINT_NAMES),
                "fps": args.fps,
                "frames": int(points_world.shape[0]),
                "clips": manifest,
            },
            f,
            indent=2,
        )
    print(json.dumps({"generated_clips": len(manifest), "out": str(args.out), "clips": manifest}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MuJoCo-direct raw annotations for PoseMamba bicycle training.")
    parser.add_argument("--out", type=Path, default=Path("raw_mujoco_direct"))
    parser.add_argument("--clip-id", default="mujoco_direct_clip_000000")
    parser.add_argument("--scene-id", default="mujoco_direct")
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument("--pattern", choices=["straight", "left", "right", "sine", "zigzag"], default="sine")
    parser.add_argument("--trajectory-seed", type=int)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--min-speed-mps", type=float, default=2.0)
    parser.add_argument("--max-speed-mps", type=float, default=8.0)
    parser.add_argument("--min-crank-hz", type=float, default=0.8)
    parser.add_argument("--max-crank-hz", type=float, default=2.4)
    parser.add_argument("--min-turn-rate-deg", type=float, default=5.0)
    parser.add_argument("--max-turn-rate-deg", type=float, default=35.0)
    parser.add_argument("--min-sine-yaw-deg", type=float, default=10.0)
    parser.add_argument("--max-sine-yaw-deg", type=float, default=45.0)
    parser.add_argument("--min-sine-frequency-hz", type=float, default=0.05)
    parser.add_argument("--max-sine-frequency-hz", type=float, default=0.35)
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fov-deg",
        type=float,
        help="Deprecated shorthand: use a fixed FOV by setting both min/max FOV to this value.",
    )
    parser.add_argument("--min-fov-deg", type=float, default=35.0)
    parser.add_argument("--max-fov-deg", type=float, default=75.0)
    parser.add_argument("--min-camera-distance", type=float, default=4.0)
    parser.add_argument("--max-camera-distance", type=float, default=12.0)
    parser.add_argument("--min-camera-elevation-deg", type=float, default=8.0)
    parser.add_argument("--max-camera-elevation-deg", type=float, default=45.0)
    parser.add_argument("--fit-camera-to-motion", dest="fit_camera_to_motion", action="store_true", default=True)
    parser.add_argument("--no-fit-camera-to-motion", dest="fit_camera_to_motion", action="store_false")
    parser.add_argument("--camera-fit-margin", type=float, default=1.15)
    parser.add_argument("--min-visible-keypoints", type=int, default=len(BICYCLE_KEYPOINT_NAMES))
    parser.add_argument("--min-visible-frame-ratio", type=float, default=1.0)
    parser.add_argument("--camera-max-tries", type=int, default=200)
    parser.add_argument("--camera-target", choices=BICYCLE_KEYPOINT_NAMES, default="k_handlebar_middle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if args.num_cameras < 1:
        raise ValueError("--num-cameras must be >= 1")
    if args.fov_deg is not None:
        args.min_fov_deg = args.fov_deg
        args.max_fov_deg = args.fov_deg
    if args.min_fov_deg > args.max_fov_deg:
        raise ValueError("--min-fov-deg must be <= --max-fov-deg")
    if args.min_camera_distance > args.max_camera_distance:
        raise ValueError("--min-camera-distance must be <= --max-camera-distance")
    if args.min_camera_elevation_deg > args.max_camera_elevation_deg:
        raise ValueError("--min-camera-elevation-deg must be <= --max-camera-elevation-deg")
    if args.min_visible_keypoints < 1 or args.min_visible_keypoints > len(BICYCLE_KEYPOINT_NAMES):
        raise ValueError("--min-visible-keypoints must be between 1 and the number of keypoints")
    if not 0.0 < args.min_visible_frame_ratio <= 1.0:
        raise ValueError("--min-visible-frame-ratio must be in (0, 1]")
    if args.camera_max_tries < 1:
        raise ValueError("--camera-max-tries must be >= 1")
    if args.camera_fit_margin <= 0.0:
        raise ValueError("--camera-fit-margin must be > 0")
    if args.min_speed_mps > args.max_speed_mps:
        raise ValueError("--min-speed-mps must be <= --max-speed-mps")
    if args.min_crank_hz > args.max_crank_hz:
        raise ValueError("--min-crank-hz must be <= --max-crank-hz")
    if args.min_turn_rate_deg > args.max_turn_rate_deg:
        raise ValueError("--min-turn-rate-deg must be <= --max-turn-rate-deg")
    if args.min_sine_yaw_deg > args.max_sine_yaw_deg:
        raise ValueError("--min-sine-yaw-deg must be <= --max-sine-yaw-deg")
    if args.min_sine_frequency_hz > args.max_sine_frequency_hz:
        raise ValueError("--min-sine-frequency-hz must be <= --max-sine-frequency-hz")
    generate_annotations(args)


if __name__ == "__main__":
    main()
