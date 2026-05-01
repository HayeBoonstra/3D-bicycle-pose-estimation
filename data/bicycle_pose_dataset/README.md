# Bicycle Pose Dataset

Synthetic bicycle keypoint dataset generated from hand-authored Blender scenes.

The 2D annotations are standard COCO keypoints JSON files so they can be loaded by
common 2D pose frameworks. The 3D keypoints and camera metadata are stored next to
the COCO files for the later temporal 3D pose stage.

## Folder Layout

```text
data/bicycle_pose_dataset/
  annotations/
    train.json
    val.json
    test.json
  images/
    train/<clip_id>/frame_0000.png
    val/<clip_id>/frame_0000.png
    test/<clip_id>/frame_0000.png
  metadata/<clip_id>/
    camera.json
    keypoints_3d.jsonl
    render_config.json
  videos/
    <clip_id>.mp4
  splits.json
```

Raw Blender output is written to `raw_renders/<clip_id>/` first, then converted to
this dataset layout with `tools/convert_to_coco.py`.

## Scene Registry Workflow

Each renderable scene is a hand-built `.blend` file under `Blender files/Scenes/`.
The scene should already contain the bicycle, rider, keypoint empties, active
camera, and MuJoCo-derived animation keyframes.

Register scenes in `Blender files/Scenes/scenes.yaml`:

```yaml
scenes:
  - id: mountain_bike
    blend: "mountain bike.blend"
    bike: mountain
    rider: henka
    frame_range: null
    camera_target: k_handlebar_middle
    weight: 1.0
```

The batch renderer samples entries from this registry and randomizes only the
camera via `Blender files/randomize_camera.py`.

## Keypoint Schema

The canonical schema lives in `tools/bicycle_keypoint_schema.py`.

Keypoint order:

1. `k_bottom_bracket`
2. `k_seat_stay`
3. `k_saddle`
4. `k_upper_head_tube`
5. `k_lower_head_tube`
6. `k_handlebar_left`
7. `k_handlebar_middle`
8. `k_handlebar_right`
9. `k_front_hub_left`
10. `k_front_hub_right`
11. `k_front_wheel_back`
12. `k_front_wheel_front`
13. `k_front_wheel_ground`
14. `k_rear_hub_left`
15. `k_rear_hub_right`
16. `k_rear_wheel_ground`
17. `k_left_pedal`
18. `k_right_pedal`

COCO visibility flags:

- `2`: keypoint is in front of the camera and inside the rendered image.
- `1`: keypoint is in front of the camera but outside the rendered image when
  `convert_to_coco.py --outside-visibility occluded` is used.
- `0`: keypoint is missing, behind the camera, or outside the image when
  `--outside-visibility unlabeled` is used.

Bboxes are computed from visible keypoints with a 10 percent margin and clamped
to the image bounds.

## Commands

Validate the scene registry:

```bash
python tools/scene_registry.py
```

Render camera-randomized clips:

```bash
python tools/batch_render.py --num-clips 20 --seed 42 --encode-video
```

Create deterministic clip-level splits:

```bash
python tools/split_dataset.py --seed 42
```

Convert raw renders into COCO:

```bash
python tools/convert_to_coco.py --outside-visibility occluded
```

Visualize the produced COCO annotations:

```bash
python tools/visualize_coco.py \
  --coco data/bicycle_pose_dataset/annotations/train.json \
  --encode-video
```

## Detector Notes

MMPose is the recommended first training target because it supports custom
COCO-style keypoint datasets and arbitrary skeletons cleanly. Use
`data/bicycle_pose_dataset/annotations/train.json` and the matching
`images/train/` root in a custom dataset config. RTMPose and HRNet are sensible
starting points.

Ultralytics YOLO-pose can train on custom keypoint counts, but it expects YOLO
label text files rather than COCO JSON. Use this COCO dataset as the source of
truth and add a small COCO-to-YOLO-pose adapter if you choose YOLO.

Detectron2 Keypoint R-CNN can register the COCO file directly with a custom
category and keypoint metadata. It is a solid baseline, but usually slower to
iterate than MMPose for custom skeleton experiments.

## 3D Stage Notes

`metadata/<clip_id>/camera.json` stores camera intrinsics and extrinsics.
`metadata/<clip_id>/keypoints_3d.jsonl` stores one JSON object per frame with
world-space keypoints in the same order as the 2D COCO annotations. Keep
`splits.json` fixed between the 2D and 3D stages to avoid temporal leakage.
