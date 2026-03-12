#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Convert an iPhone split-screen recording (LiDAR on left, RGB on right)
into a COLMAP dataset compatible with ``frgs reconstruct``.

The script crops the right half of each frame (the RGB camera feed) and
discards the left half (the LiDAR depth visualization, which is not usable
as raw depth data).

Pipeline:
  Phase 1 -- Extract frames from the video at a target FPS, crop to the
             RGB portion, with optional blur filtering.
  Phase 2 -- Run COLMAP feature extraction (SIFT) on the cropped frames.
  Phase 3 -- Run COLMAP feature matching, then GLOMAP (or COLMAP) mapper
             for sparse reconstruction.
  Phase 4 -- Arrange the output into the COLMAP layout expected by frgs.

Requires:
  - ffmpeg (on PATH or via --ffmpeg-path)
  - COLMAP (on PATH or via --colmap-path)
  - GLOMAP (on PATH or via --glomap-path) -- or pass --mapper colmap
  - opencv-python, numpy
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _colmap_video_common import (
    arrange_colmap_output,
    detect_split_screen_boundary,
    extract_frames,
    parse_crop_rect,
    run_colmap_feature_extractor,
    run_colmap_matcher,
    run_mapper,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert an iPhone split-screen recording (LiDAR + RGB) to a "
            "COLMAP dataset for frgs reconstruct. The RGB portion (right half) "
            "is extracted; the LiDAR visualization is discarded."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", type=Path, help="Path to the input screen recording (MP4/MOV).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the frgs-compatible COLMAP dataset.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediate files. Defaults to <output-dir>/_work.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target frame extraction rate (frames per second).",
    )
    parser.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Image format for extracted frames.",
    )
    parser.add_argument(
        "--crop",
        choices=["right_half", "left_half", "auto", "manual"],
        default="right_half",
        help=(
            "Which portion of the split-screen to keep. "
            "'right_half' uses the exact right half, "
            "'auto' attempts to detect the split boundary, "
            "'manual' requires --crop-rect."
        ),
    )
    parser.add_argument(
        "--crop-rect",
        type=str,
        default=None,
        help="Manual crop rectangle as x,y,w,h (e.g. '960,0,960,1920'). Required when --crop manual.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=None,
        help="Skip frames with Laplacian variance below this value. Set to 0 to disable.",
    )
    parser.add_argument(
        "--colmap-camera-model",
        default="PINHOLE",
        help="COLMAP camera model for feature extraction (PINHOLE works well for iPhone).",
    )
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive", "vocab_tree"],
        default="sequential",
        help="COLMAP matching strategy.",
    )
    parser.add_argument(
        "--mapper",
        choices=["glomap", "colmap"],
        default="glomap",
        help="Mapper for sparse reconstruction. GLOMAP is faster; COLMAP is the incremental fallback.",
    )
    parser.add_argument("--colmap-path", type=str, default=None, help="Explicit path to the colmap binary.")
    parser.add_argument("--glomap-path", type=str, default=None, help="Explicit path to the glomap binary.")
    parser.add_argument("--ffmpeg-path", type=str, default=None, help="Explicit path to the ffmpeg binary.")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Stop after frame extraction (Phase 1). Images are written to <output-dir>/images_raw/.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    video_path = args.video_path.resolve()
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve() if args.work_dir else output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "images_raw"
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"

    # Determine crop rectangle
    crop_rect = _resolve_crop_rect(args, video_path)
    logger.info("Crop rectangle (x, y, w, h): %s", crop_rect)

    # Phase 1: Frame extraction + cropping
    logger.info("=== Phase 1: Extracting and cropping frames from video ===")
    frames = extract_frames(
        video_path,
        image_dir,
        fps=args.fps,
        blur_threshold=args.blur_threshold,
        crop_rect=crop_rect,
        image_format=args.image_format,
        ffmpeg_path=args.ffmpeg_path,
    )
    if not frames:
        logger.error("No frames extracted. Check the video file and settings.")
        sys.exit(1)
    logger.info("Extracted %d cropped frames to %s", len(frames), image_dir)

    if args.extract_only:
        logger.info("--extract-only set; stopping after frame extraction.")
        logger.info("Raw images at: %s", image_dir)
        return

    # Phase 2: COLMAP feature extraction
    logger.info("=== Phase 2: COLMAP feature extraction ===")
    run_colmap_feature_extractor(
        image_dir,
        db_path,
        camera_model=args.colmap_camera_model,
        single_camera=True,
        colmap_path=args.colmap_path,
    )

    # Phase 3: Matching + Mapping
    logger.info("=== Phase 3: Matching + Mapping ===")
    run_colmap_matcher(db_path, matcher_type=args.matcher, colmap_path=args.colmap_path)
    model_dir = run_mapper(
        db_path,
        image_dir,
        sparse_dir,
        mapper=args.mapper,
        colmap_path=args.colmap_path,
        glomap_path=args.glomap_path,
    )

    # Phase 4: Arrange output
    logger.info("=== Phase 4: Arranging COLMAP output ===")
    arrange_colmap_output(image_dir, model_dir, output_dir)

    logger.info("Done. Run: frgs reconstruct %s -o <output.ply>", output_dir)


def _resolve_crop_rect(
    args: argparse.Namespace, video_path: Path
) -> tuple[int, int, int, int]:
    """Determine the crop rectangle from CLI arguments."""
    import cv2

    if args.crop == "manual":
        if args.crop_rect is None:
            logger.error("--crop manual requires --crop-rect x,y,w,h")
            sys.exit(1)
        return parse_crop_rect(args.crop_rect)

    if args.crop == "auto":
        return detect_split_screen_boundary(video_path, ffmpeg_path=args.ffmpeg_path)

    # right_half or left_half: probe video dimensions
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video to determine dimensions: %s", video_path)
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    half_w = w // 2
    if args.crop == "right_half":
        rect = (half_w, 0, w - half_w, h)
    else:  # left_half
        rect = (0, 0, half_w, h)

    logger.info("Video dimensions: %dx%d, using %s crop: %s", w, h, args.crop, rect)
    return rect


if __name__ == "__main__":
    main()
