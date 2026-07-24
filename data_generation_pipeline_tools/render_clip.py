"""Render one registered Blender scene into a raw clip dataset folder.

Run through Blender, for example:

blender --background "Blender files/Scenes/mountain bike.blend" \
  --python tools/render_clip.py -- \
  --clip-id clip_mountain_bike_00000017 \
  --scene-id mountain_bike \
  --camera-seed 17 \
  --out raw_renders/clip_mountain_bike_00000017
"""

from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[1]
BLENDER_FILES_DIR = REPO_ROOT / "Blender files"
RANDOMIZE_CAMERA_SCRIPT = BLENDER_FILES_DIR / "randomize_camera.py"
TRACK_CAMERA_SCRIPT = BLENDER_FILES_DIR / "track_camera_to_target.py"
RANDOMIZE_LIGHTING_SCRIPT = BLENDER_FILES_DIR / "randomize_lighting_strength.py"
RANDOMIZE_FOG_SCRIPT = BLENDER_FILES_DIR / "randomize_fog.py"
ANIMATE_SCRIPT = BLENDER_FILES_DIR / "Animate_script.py"
EXPORT_SCRIPT = BLENDER_FILES_DIR / "extract_2D_annotation.py"


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Blender clip and export labels.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--camera-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bike", default="")
    parser.add_argument("--rider", default="")
    parser.add_argument("--camera-target", default="k_handlebar_middle")
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        help="Optional MuJoCo transform CSV to import onto the bicycle armature before rendering.",
    )
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument(
        "--sync-window-size",
        type=int,
        default=80,
        help="Rendered window size (frames) centered on sampled camera frame.",
    )
    parser.add_argument(
        "--no-sync-camera-window",
        action="store_true",
        help="Disable camera-sampled frame window synchronization.",
    )
    parser.add_argument("--encode-video", action="store_true")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--render-format", choices=("PNG", "JPEG"), default=os.environ.get("RENDER_FORMAT", "PNG"))
    parser.add_argument(
        "--resolution-percentage",
        type=int,
        default=int(os.environ.get("BLENDER_RESOLUTION_PERCENTAGE", "100")),
    )
    parser.add_argument(
        "--cycles-samples",
        type=int,
        default=int(os.environ.get("BLENDER_CYCLES_SAMPLES", "0")),
        help="If >0 and the scene uses Cycles, cap render samples for faster scale runs.",
    )
    parser.add_argument(
        "--quiet-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress non-essential render logs (use --no-quiet-mode to enable).",
    )
    parser.add_argument(
        "--camera-min-distance",
        type=float,
        default=float(os.environ.get("CAMERA_MIN_DISTANCE", "4.0")),
        help="Minimum camera distance (m); also enforces motion-aware fit for RTMPose.",
    )
    parser.add_argument(
        "--camera-max-distance",
        type=float,
        default=float(os.environ.get("CAMERA_MAX_DISTANCE", "12.0")),
        help="Maximum camera distance (m). Keep moderate so the bike is not tiny in frame.",
    )
    parser.add_argument(
        "--camera-min-bbox-area-frac",
        type=float,
        default=float(os.environ.get("CAMERA_MIN_BBOX_AREA_FRAC", "0.04")),
        help="Reject poses where projected keypoint bbox area is below this image fraction.",
    )
    parser.add_argument(
        "--camera-max-bbox-area-frac",
        type=float,
        default=float(os.environ.get("CAMERA_MAX_BBOX_AREA_FRAC", "0.80")),
        help="Reject poses where the bike fills more than this fraction (too zoomed in).",
    )
    parser.add_argument(
        "--camera-min-visible-keypoints",
        type=int,
        default=int(os.environ.get("CAMERA_MIN_VISIBLE_KEYPOINTS", "14")),
    )
    parser.add_argument(
        "--camera-min-visible-frame-ratio",
        type=float,
        default=float(os.environ.get("CAMERA_MIN_VISIBLE_FRAME_RATIO", "0.9")),
    )
    parser.add_argument(
        "--camera-fit-margin",
        type=float,
        default=float(os.environ.get("CAMERA_FIT_MARGIN", "1.25")),
        help="Scale motion radius when computing minimum camera distance.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("track", "fixed"),
        default=os.environ.get("CAMERA_MODE", "track"),
        help="track: camera follows bicycle each frame; fixed: legacy world-fixed camera.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help=(
            "Re-apply trajectory and camera setup, then re-export labels without rendering PNG frames. "
            "Requires an existing clip output directory and --trajectory-csv."
        ),
    )
    return parser.parse_args(_argv_after_double_dash())


def _set_env(args: argparse.Namespace, output_dir: Path) -> None:
    os.environ["CLIP_ID"] = args.clip_id
    os.environ["SCENE_ID"] = args.scene_id
    os.environ["CAMERA_SEED"] = str(args.camera_seed)
    os.environ["CAMERA_TARGET"] = args.camera_target
    os.environ["CLIP_OUTPUT_DIR"] = str(output_dir)
    os.environ["RAW_RENDERS_DIR"] = str(output_dir)
    os.environ["BIKE_TAG"] = args.bike
    os.environ["RIDER_TAG"] = args.rider
    os.environ["REPO_ROOT"] = str(REPO_ROOT)
    if args.trajectory_csv is not None:
        os.environ["TRAJECTORY_CSV"] = str(args.trajectory_csv.resolve())
    else:
        os.environ.pop("TRAJECTORY_CSV", None)
    os.environ["CAMERA_SYNC_WINDOW_SIZE"] = str(args.sync_window_size)
    os.environ["CAMERA_MIN_DISTANCE"] = str(args.camera_min_distance)
    os.environ["CAMERA_MAX_DISTANCE"] = str(args.camera_max_distance)
    os.environ["CAMERA_MIN_BBOX_AREA_FRAC"] = str(args.camera_min_bbox_area_frac)
    os.environ["CAMERA_MAX_BBOX_AREA_FRAC"] = str(args.camera_max_bbox_area_frac)
    os.environ["CAMERA_MIN_VISIBLE_KEYPOINTS"] = str(args.camera_min_visible_keypoints)
    os.environ["CAMERA_MIN_VISIBLE_FRAME_RATIO"] = str(args.camera_min_visible_frame_ratio)
    os.environ["CAMERA_FIT_MARGIN"] = str(args.camera_fit_margin)
    os.environ["CAMERA_MODE"] = str(args.camera_mode)


def _configure_scene(args: argparse.Namespace, output_dir: Path) -> None:
    scene = bpy.context.scene
    if args.frame_start is not None:
        scene.frame_start = args.frame_start
    if args.frame_end is not None:
        scene.frame_end = args.frame_end
    if args.fps is not None:
        scene.render.fps = args.fps

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    render_format = str(args.render_format).upper()
    # Some scenes are saved with movie output (FFMPEG), which can lock image_settings
    # enums. Force image-sequence mode first, then assign requested still format.
    try:
        scene.render.file_format = render_format
    except Exception:
        pass
    try:
        scene.render.image_settings.file_format = render_format
    except TypeError:
        # Fallback: force PNG sequence if the requested format is unavailable.
        scene.render.file_format = "PNG"
        scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_percentage = max(1, min(100, int(args.resolution_percentage)))
    if args.cycles_samples > 0 and hasattr(scene, "cycles"):
        scene.cycles.samples = int(args.cycles_samples)
    scene.render.filepath = str(frames_dir / "frame_")


def _run_blender_script(path: Path, quiet_mode: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not quiet_mode:
        print(f"[render_clip] running {path}")
    runpy.run_path(str(path), run_name="__main__")


@contextlib.contextmanager
def _silence_stdout_stderr():
    """Temporarily silence noisy Blender C-level render logs."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield saved_stdout
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


@contextlib.contextmanager
def _render_progress(scene, output_fd: int, quiet_mode: bool) -> None:
    """Print one in-place progress line while Blender renders animation frames."""
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    total_frames = max(1, frame_end - frame_start + 1)
    state = {"last_frame": None, "printed": False}

    def _write(message: str) -> None:
        try:
            os.write(output_fd, message.encode("utf-8", errors="replace"))
        except OSError:
            # If the output stream is gone, avoid crashing the render.
            pass

    def _on_render_write(_scene, _depsgraph):
        current_frame = int(_scene.frame_current)
        if state["last_frame"] == current_frame:
            return
        state["last_frame"] = current_frame
        done = max(0, min(total_frames, current_frame - frame_start + 1))
        if not quiet_mode:
            _write(f"\r[render_clip] rendering frame {done}/{total_frames} (scene frame {current_frame})")
        state["printed"] = True

    bpy.app.handlers.render_write.append(_on_render_write)
    try:
        yield
    finally:
        if _on_render_write in bpy.app.handlers.render_write:
            bpy.app.handlers.render_write.remove(_on_render_write)
        if state["printed"] and not quiet_mode:
            _write("\n")


def _synchronize_frame_window(args: argparse.Namespace) -> None:
    if args.no_sync_camera_window:
        return
    if args.sync_window_size < 1:
        raise ValueError("--sync-window-size must be >= 1")

    scene = bpy.context.scene
    min_frame = int(args.frame_start) if args.frame_start is not None else int(scene.frame_start)
    max_frame = int(args.frame_end) if args.frame_end is not None else int(scene.frame_end)
    if max_frame < min_frame:
        raise ValueError("frame range is invalid: frame_end must be >= frame_start")

    sample_frame_raw = os.environ.get("CAMERA_SAMPLE_FRAME")
    if sample_frame_raw in {None, ""}:
        print("[render_clip] camera sample frame not set; using original frame range")
        return

    sample_frame = int(sample_frame_raw)
    window = min(int(args.sync_window_size), max_frame - min_frame + 1)
    half = window // 2
    start = sample_frame - half
    end = start + window - 1

    if start < min_frame:
        start = min_frame
        end = start + window - 1
    if end > max_frame:
        end = max_frame
        start = end - window + 1

    scene.frame_start = int(start)
    scene.frame_end = int(end)
    if not args.quiet_mode:
        print(
            "[render_clip] synchronized frame window around sampled frame "
            f"{sample_frame}: {scene.frame_start}-{scene.frame_end} (n={window})"
        )


def _encode_mp4(output_dir: Path, clip_id: str, fps: int, frame_start: int, quiet_mode: bool) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.")

    output_video = output_dir / f"{clip_id}.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(frame_start),
        "-i",
        str(output_dir / "frames" / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)
    if not quiet_mode:
        print(f"[render_clip] wrote {output_video}")


def _load_existing_render_config(output_dir: Path) -> dict:
    path = output_dir / "render_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"--annotations-only requires existing render_config.json in {output_dir}"
        )
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def render_one_clip(args: argparse.Namespace) -> None:
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.trajectory_csv is None:
        raise ValueError("--trajectory-csv is required for bicycle clip export")

    if args.annotations_only:
        existing = _load_existing_render_config(output_dir)
        scene = bpy.context.scene
        scene.frame_start = int(existing.get("frame_start", scene.frame_start))
        scene.frame_end = int(existing.get("frame_end", scene.frame_end))
        if args.fps is not None:
            scene.render.fps = args.fps
        elif existing.get("fps"):
            scene.render.fps = int(existing["fps"])

    _set_env(args, output_dir)
    _configure_scene(args, output_dir)

    scene = bpy.context.scene
    if not args.quiet_mode:
        print(f"[render_clip] importing trajectory: {args.trajectory_csv}")
    _run_blender_script(ANIMATE_SCRIPT, args.quiet_mode)
    if args.frame_start is not None:
        scene.frame_start = int(args.frame_start)
    if args.frame_end is not None:
        scene.frame_end = int(args.frame_end)
    if args.fps is not None:
        scene.render.fps = args.fps
    if not args.quiet_mode:
        print(
            f"[render_clip] clip={args.clip_id} scene={args.scene_id} "
            f"frames={scene.frame_start}-{scene.frame_end} camera_seed={args.camera_seed}"
        )

    if not args.quiet_mode:
        print("[render_clip] randomizing camera...")
    _run_blender_script(RANDOMIZE_CAMERA_SCRIPT, args.quiet_mode)
    if not args.quiet_mode:
        print("[render_clip] randomizing lighting...")
    _run_blender_script(RANDOMIZE_LIGHTING_SCRIPT, args.quiet_mode)
    if not args.quiet_mode:
        print("[render_clip] randomizing fog...")
    _run_blender_script(RANDOMIZE_FOG_SCRIPT, args.quiet_mode)
    _synchronize_frame_window(args)
    if args.camera_mode == "track":
        if not args.quiet_mode:
            print("[render_clip] keyframing follow camera...")
        _run_blender_script(TRACK_CAMERA_SCRIPT, args.quiet_mode)
    if args.annotations_only:
        if not args.quiet_mode:
            print("[render_clip] annotations-only: skipping PNG render")
    else:
        if not args.quiet_mode:
            print("[render_clip] rendering frames...")
        if args.quiet_mode:
            with _silence_stdout_stderr() as terminal_fd:
                with _render_progress(scene, terminal_fd, args.quiet_mode):
                    bpy.ops.render.render(animation=True)
        else:
            terminal_fd = sys.stdout.fileno()
            with _render_progress(scene, terminal_fd, args.quiet_mode):
                bpy.ops.render.render(animation=True)
        if not args.quiet_mode:
            print("[render_clip] rendering complete")
    _run_blender_script(EXPORT_SCRIPT, args.quiet_mode)

    if args.encode_video:
        _encode_mp4(
            output_dir,
            args.clip_id,
            int(scene.render.fps),
            int(scene.frame_start),
            args.quiet_mode,
        )


def main() -> None:
    render_one_clip(_parse_args())


if __name__ == "__main__":
    main()
