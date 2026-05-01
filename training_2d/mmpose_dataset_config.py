"""Generate MMPose dataset config for bicycle keypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.schema import BICYCLE_KEYPOINT_NAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a minimal MMPose config snippet.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_config(dataset_root: Path) -> str:
    keypoints = ", ".join([f'"{name}"' for name in BICYCLE_KEYPOINT_NAMES])
    return f"""dataset_type = 'CocoDataset'
data_root = '{dataset_root.as_posix()}'

metainfo = dict(
    dataset_name='bicycle_pose',
    keypoint_info={{{', '.join([f'{i}: dict(name=\"{k}\")' for i, k in enumerate(BICYCLE_KEYPOINT_NAMES)])}}},
    skeleton_info={{}},
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=metainfo,
    )
)
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        metainfo=metainfo,
    )
)
custom_keypoint_names = [{keypoints}]
"""


def main() -> None:
    args = _parse_args()
    text = build_config(args.dataset_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote MMPose config snippet to {args.out}")


if __name__ == "__main__":
    main()

