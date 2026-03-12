#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Convert a Ray-Ban Meta glasses video recording into a COLMAP dataset
compatible with ``frgs reconstruct``.

Pipeline:
  Phase 1 -- Extract frames from the video at a target FPS, with optional
             blur filtering.
  Phase 2 -- Run COLMAP feature extraction (SIFT) on the frames.
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
    extract_frames,
    run_colmap_feature_extractor,
    run_colmap_matcher,
    run_mapper,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Ray-Ban Meta glasses video recording to a COLMAP dataset "
            "for frgs reconstruct."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", type=Path, help="Path to the input video file (MP4/MOV).")
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
        "--blur-threshold",
        type=float,
        default=None,
        help="Skip frames with Laplacian variance below this value. Set to 0 to disable.",
    )
    parser.add_argument(
        "--colmap-camera-model",
        default="OPENCV",
        help="COLMAP camera model for feature extraction (OPENCV recommended for wide-angle Meta glasses lens).",
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

    # Phase 1: Frame extraction
    logger.info("=== Phase 1: Extracting frames from video ===")
    frames = extract_frames(
        video_path,
        image_dir,
        fps=args.fps,
        blur_threshold=args.blur_threshold,
        image_format=args.image_format,
        ffmpeg_path=args.ffmpeg_path,
    )
    if not frames:
        logger.error("No frames extracted. Check the video file and settings.")
        sys.exit(1)
    logger.info("Extracted %d frames to %s", len(frames), image_dir)

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


if __name__ == "__main__":
    main()
