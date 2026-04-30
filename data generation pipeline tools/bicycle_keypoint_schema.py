"""Shared bicycle keypoint schema for Blender export and COCO conversion."""

from __future__ import annotations

CATEGORY_ID = 1
CATEGORY_NAME = "bicycle"

BICYCLE_KEYPOINT_NAMES = [
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

BICYCLE_SKELETON_NAMES = [
    ("k_bottom_bracket", "k_seat_stay"),
    ("k_seat_stay", "k_saddle"),
    ("k_bottom_bracket", "k_lower_head_tube"),
    ("k_lower_head_tube", "k_upper_head_tube"),
    ("k_handlebar_left", "k_handlebar_middle"),
    ("k_handlebar_middle", "k_handlebar_right"),
    ("k_handlebar_middle", "k_upper_head_tube"),
    ("k_front_hub_left", "k_front_hub_right"),
    ("k_front_wheel_back", "k_front_wheel_front"),
    ("k_front_wheel_front", "k_front_wheel_ground"),
    ("k_front_wheel_ground", "k_front_wheel_back"),
    ("k_rear_hub_left", "k_rear_hub_right"),
    ("k_bottom_bracket", "k_rear_wheel_ground"),
    ("k_bottom_bracket", "k_left_pedal"),
    ("k_bottom_bracket", "k_right_pedal"),
]

KEYPOINT_INDEX = {name: index for index, name in enumerate(BICYCLE_KEYPOINT_NAMES)}

# COCO skeleton edges are 1-indexed keypoint indices.
BICYCLE_SKELETON = [
    [KEYPOINT_INDEX[start] + 1, KEYPOINT_INDEX[end] + 1]
    for start, end in BICYCLE_SKELETON_NAMES
]


def canonical_keypoint_name(name: str) -> str:
    """Return the schema keypoint name for a Blender object name.

    The current Blender scenes may contain either `bottom_bracket` or
    `k_bottom_bracket` style empties. The exported dataset always uses the
    `k_` prefix.
    """
    return name if name.startswith("k_") else f"k_{name}"


def coco_category() -> dict:
    """Return the COCO category definition for this keypoint schema."""
    return {
        "id": CATEGORY_ID,
        "name": CATEGORY_NAME,
        "supercategory": "vehicle",
        "keypoints": BICYCLE_KEYPOINT_NAMES,
        "skeleton": BICYCLE_SKELETON,
    }
