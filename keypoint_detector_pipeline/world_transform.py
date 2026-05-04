"""Camera/world coordinate transforms and reprojection checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


def _as_np(data, shape):
    arr = np.asarray(data, dtype=np.float32)
    return arr.reshape(shape)


def camera_from_json(payload: dict) -> CameraModel:
    K = _as_np(payload.get("K", np.eye(3)), (3, 3))
    R = _as_np(payload.get("R", np.eye(3)), (3, 3))
    t = _as_np(payload.get("t", [0.0, 0.0, 0.0]), (3,))
    return CameraModel(K=K, R=R, t=t)


def world_to_camera(points_world: np.ndarray, camera: CameraModel) -> np.ndarray:
    pw = np.asarray(points_world, dtype=np.float32)
    return (camera.R @ pw[..., None]).squeeze(-1) + camera.t


def camera_to_world(points_camera: np.ndarray, camera: CameraModel) -> np.ndarray:
    pc = np.asarray(points_camera, dtype=np.float32)
    return (camera.R.T @ (pc - camera.t)[..., None]).squeeze(-1)


def project_to_image(points_camera: np.ndarray, camera: CameraModel) -> np.ndarray:
    pc = np.asarray(points_camera, dtype=np.float32)
    z = np.maximum(pc[..., 2:3], 1e-6)
    norm = pc[..., :2] / z
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    xy = np.zeros_like(norm)
    xy[..., 0] = fx * norm[..., 0] + cx
    xy[..., 1] = fy * norm[..., 1] + cy
    return xy


def reprojection_rmse(points_camera: np.ndarray, points_image: np.ndarray, camera: CameraModel) -> float:
    proj = project_to_image(points_camera, camera)
    err = np.linalg.norm(proj - points_image, axis=-1)
    return float(np.sqrt(np.mean(np.square(err))))

