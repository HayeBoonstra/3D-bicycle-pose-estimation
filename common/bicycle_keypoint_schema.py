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
    ("k_lower_head_tube", "k_front_hub_left"),
    ("k_lower_head_tube", "k_front_hub_right"),
    ("k_handlebar_left", "k_handlebar_middle"),
    ("k_handlebar_middle", "k_handlebar_right"),
    ("k_handlebar_middle", "k_upper_head_tube"),
    ("k_bottom_bracket", "k_rear_hub_left"),
    ("k_bottom_bracket", "k_rear_hub_right"),
    ("k_front_hub_left", "k_front_hub_right"),
    ("k_front_hub_left", "k_front_wheel_back"),
    ("k_front_hub_right", "k_front_wheel_back"),
    ("k_front_hub_right", "k_front_wheel_front"),
    ("k_front_hub_left", "k_front_wheel_front"),
    ("k_front_hub_left", "k_front_wheel_ground"),
    ("k_front_hub_right", "k_front_wheel_ground"),
    ("k_rear_hub_left", "k_rear_hub_right"),
    ("k_rear_hub_left", "k_rear_wheel_ground"),
    ("k_rear_hub_right", "k_rear_wheel_ground"),
    ("k_bottom_bracket", "k_left_pedal"),
    ("k_bottom_bracket", "k_right_pedal"),
    ("k_rear_hub_left", "k_seat_stay"),
    ("k_rear_hub_right", "k_seat_stay"),
]

KEYPOINT_INDEX = {name: index for index, name in enumerate(BICYCLE_KEYPOINT_NAMES)}

# COCO skeleton edges are 1-indexed keypoint indices.
BICYCLE_SKELETON = [
    [KEYPOINT_INDEX[start] + 1, KEYPOINT_INDEX[end] + 1]
    for start, end in BICYCLE_SKELETON_NAMES
]

# Semantic edge groups for 3D visualization (each skeleton edge appears once).
SKELETON_PART_GROUPS: dict[str, list[tuple[str, str]]] = {
    "front_assembly": [
        ("k_bottom_bracket", "k_lower_head_tube"),
        ("k_lower_head_tube", "k_upper_head_tube"),
        ("k_lower_head_tube", "k_front_hub_left"),
        ("k_lower_head_tube", "k_front_hub_right"),
        ("k_handlebar_left", "k_handlebar_middle"),
        ("k_handlebar_middle", "k_handlebar_right"),
        ("k_handlebar_middle", "k_upper_head_tube"),
        ("k_front_hub_left", "k_front_hub_right"),
        ("k_front_hub_left", "k_front_wheel_back"),
        ("k_front_hub_right", "k_front_wheel_back"),
        ("k_front_hub_right", "k_front_wheel_front"),
        ("k_front_hub_left", "k_front_wheel_front"),
        ("k_front_hub_left", "k_front_wheel_ground"),
        ("k_front_hub_right", "k_front_wheel_ground"),
    ],
    "rear_assembly": [
        ("k_bottom_bracket", "k_seat_stay"),
        ("k_seat_stay", "k_saddle"),
        ("k_bottom_bracket", "k_rear_hub_left"),
        ("k_bottom_bracket", "k_rear_hub_right"),
        ("k_rear_hub_left", "k_rear_hub_right"),
        ("k_rear_hub_left", "k_rear_wheel_ground"),
        ("k_rear_hub_right", "k_rear_wheel_ground"),
        ("k_rear_hub_left", "k_seat_stay"),
        ("k_rear_hub_right", "k_seat_stay"),
    ],
    "drivetrain": [
        ("k_bottom_bracket", "k_left_pedal"),
        ("k_bottom_bracket", "k_right_pedal"),
    ],
}

PART_COLORS_PRED: dict[str, str] = {
    "front_assembly": "#00457E",
    "rear_assembly": "#2F8F5B",
    "drivetrain": "#C44E52",
}

PART_COLORS_GT: dict[str, str] = {
    "front_assembly": "#999999",
    "rear_assembly": "#BBBBBB",
    "drivetrain": "#888888",
}


def skeleton_edges_by_part() -> dict[str, list[tuple[int, int]]]:
    """Return skeleton edge indices grouped by semantic part."""
    grouped: dict[str, list[tuple[int, int]]] = {}
    for part, edges in SKELETON_PART_GROUPS.items():
        grouped[part] = [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b]) for a, b in edges]
    return grouped


def validate_skeleton_part_groups() -> None:
    """Ensure part groups partition BICYCLE_SKELETON_NAMES exactly once."""
    seen: set[tuple[str, str]] = set()
    for edges in SKELETON_PART_GROUPS.values():
        for edge in edges:
            if edge in seen:
                raise ValueError(f"Duplicate skeleton edge in part groups: {edge}")
            seen.add(edge)
    expected = set(BICYCLE_SKELETON_NAMES)
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise ValueError(
            f"Skeleton part groups mismatch: missing={missing!r}, extra={extra!r}"
        )


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
