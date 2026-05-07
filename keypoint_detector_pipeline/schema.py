"""Shared keypoint schema access for inference/training code."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "data_generation_pipeline_tools" / "bicycle_keypoint_schema.py"

_FALLBACK_KEYPOINTS = [
    "k_bottom_bracket",
    "k_seat_stay",
    "k_saddle",
    "k_upper_head_tube",
    "k_lower_head_tube",
    "k_handlebar_left",
    "k_handlebar_middle",
    "k_handlebar_right",
    "k_front_hub_left",
    "k_front_hub_right",
    "k_front_wheel_back",
    "k_front_wheel_front",
    "k_front_wheel_ground",
    "k_rear_hub_left",
    "k_rear_hub_right",
    "k_rear_wheel_ground",
    "k_left_pedal",
    "k_right_pedal",
]


def _load_schema_module():
    if not SCHEMA_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("bicycle_keypoint_schema", SCHEMA_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_schema_module = _load_schema_module()
BICYCLE_KEYPOINT_NAMES = (
    list(getattr(_schema_module, "BICYCLE_KEYPOINT_NAMES", _FALLBACK_KEYPOINTS))
    if _schema_module
    else _FALLBACK_KEYPOINTS
)
NUM_KEYPOINTS = len(BICYCLE_KEYPOINT_NAMES)

