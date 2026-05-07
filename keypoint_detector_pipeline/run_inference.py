"""Run the full detector -> 2D -> 3D -> world pipeline on an image sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keypoint_detector_pipeline.detect_rfdetr import run_detection
from keypoint_detector_pipeline.io_utils import dump_json, iter_jsonl, load_json, write_jsonl
from keypoint_detector_pipeline.lift3d_ssm import TemporalSSMLifter
from keypoint_detector_pipeline.pose2d_mmpose import MMPose2DInferencer
from keypoint_detector_pipeline.schema import NUM_KEYPOINTS
from keypoint_detector_pipeline.sequence_builder import build_temporal_windows, rows_to_arrays
from keypoint_detector_pipeline.world_transform import camera_from_json, camera_to_world, reprojection_rmse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cycling 3D keypoint inference pipeline.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--rfdetr-model", default="rfdetr-2xlarge")
    parser.add_argument("--det-confidence", type=float, default=0.5)
    parser.add_argument("--mmpose-model", default="rtmpose-l_8xb256-420e_coco-256x192")
    parser.add_argument("--mmpose-weights", default=None)
    parser.add_argument("--lifter-weights", type=Path, default=None)
    parser.add_argument("--lifter-config", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=27)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = args.output_dir / "detections.jsonl"
    keypoints2d_path = args.output_dir / "keypoints_2d.jsonl"
    keypoints3d_cam_path = args.output_dir / "keypoints_3d_camera.jsonl"
    keypoints3d_world_path = args.output_dir / "keypoints_3d_world.jsonl"

    run_detection(
        image_dir=args.frames_dir,
        output_path=detections_path,
        model_id=args.rfdetr_model,
        confidence=args.det_confidence,
    )

    infer2d = MMPose2DInferencer(
        pose2d_model=args.mmpose_model,
        pose2d_weights=args.mmpose_weights,
    )
    rows_2d = []
    for det in iter_jsonl(detections_path):
        kps, conf, bbox = infer2d.predict_global(Path(det["image_path"]), det["bbox_xyxy"])
        rows_2d.append(
            {
                "frame_id": int(det["frame_id"]),
                "image_path": det["image_path"],
                "bbox_xyxy": bbox if bbox is not None else [0.0, 0.0, 1.0, 1.0],
                "det_score": float(det["score"]),
                "keypoints_2d": kps.tolist(),
                "confidence": conf.tolist(),
            }
        )
    write_jsonl(keypoints2d_path, rows_2d)

    points_2d, conf_2d, bboxes = rows_to_arrays(rows_2d)
    windows, conf_windows = build_temporal_windows(points_2d, conf_2d, bboxes, window_size=args.window_size)

    lifter = TemporalSSMLifter(
        num_keypoints=NUM_KEYPOINTS,
        window_size=args.window_size,
        config_path=args.lifter_config,
    )
    lifter.load_weights(args.lifter_weights)
    kps3d_camera = lifter.infer(windows, conf_windows)

    rows_3d_cam = []
    for idx, row in enumerate(sorted(rows_2d, key=lambda r: r["frame_id"])):
        rows_3d_cam.append(
            {
                "frame_id": int(row["frame_id"]),
                "image_path": row["image_path"],
                "keypoints_3d_camera": kps3d_camera[idx].tolist(),
            }
        )
    write_jsonl(keypoints3d_cam_path, rows_3d_cam)

    if args.camera_json and args.camera_json.exists():
        cam = camera_from_json(load_json(args.camera_json))
        rows_world = []
        reproj_vals = []
        for idx, row in enumerate(rows_3d_cam):
            pc = np.asarray(row["keypoints_3d_camera"], dtype=np.float32)
            pw = camera_to_world(pc, cam)
            p2d = np.asarray(rows_2d[idx]["keypoints_2d"], dtype=np.float32)
            reproj_vals.append(reprojection_rmse(pc, p2d, cam))
            rows_world.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "image_path": row["image_path"],
                    "keypoints_3d_world": pw.tolist(),
                }
            )
        write_jsonl(keypoints3d_world_path, rows_world)
        dump_json(
            args.output_dir / "metrics.json",
            {"reprojection_rmse_mean_px": float(np.mean(reproj_vals)), "num_frames": len(rows_world)},
        )

    print(f"Wrote {detections_path}")
    print(f"Wrote {keypoints2d_path}")
    print(f"Wrote {keypoints3d_cam_path}")
    if (args.output_dir / "keypoints_3d_world.jsonl").exists():
        print(f"Wrote {keypoints3d_world_path}")


if __name__ == "__main__":
    main()

