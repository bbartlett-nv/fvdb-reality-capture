# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Shared utilities for converting video recordings into COLMAP datasets
compatible with ``frgs reconstruct``.

Provides frame extraction (via ffmpeg), blur detection, COLMAP/GLOMAP
invocation wrappers, and output directory arrangement.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _find_executable(name: str, explicit_path: Optional[str] = None) -> str:
    """Return the path to *name*, preferring *explicit_path* if given."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            logger.error("%s not found at explicit path: %s", name, explicit_path)
            sys.exit(1)
        return str(p)
    found = shutil.which(name)
    if found is None:
        logger.error("%s is not on PATH. Install it or pass --%s-path.", name, name)
        sys.exit(1)
    return found


def compute_blur_score(image: np.ndarray) -> float:
    """Return the variance of the Laplacian (higher = sharper)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_blur_histogram(values: list[float], threshold: Optional[float], path: Path) -> None:
    """Save a histogram of per-frame sharpness values to *path*."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=40, edgecolor="black", alpha=0.7)
    if threshold and threshold > 0:
        ax.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.1f}")
        ax.legend()
    ax.set_xlabel("Laplacian Variance (sharpness)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Blur Metric Distribution")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    blur_threshold: Optional[float] = None,
    crop_rect: Optional[tuple[int, int, int, int]] = None,
    image_format: str = "jpeg",
    ffmpeg_path: Optional[str] = None,
) -> list[Path]:
    """
    Extract frames from a video file into *output_dir*.

    Parameters
    ----------
    video_path : Path
        Input video file (MP4, MOV, etc.).
    output_dir : Path
        Directory to write extracted frames into.
    fps : float
        Target extraction rate in frames per second.
    blur_threshold : float, optional
        Discard frames with Laplacian variance below this value.
    crop_rect : (x, y, w, h), optional
        Crop each frame to this rectangle before saving.
    image_format : str
        ``"jpeg"`` or ``"png"``.
    ffmpeg_path : str, optional
        Explicit path to the ffmpeg binary.

    Returns
    -------
    list[Path]
        Paths to the extracted frame images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_executable("ffmpeg", ffmpeg_path)

    ext = ".jpg" if image_format == "jpeg" else ".png"

    # Build ffmpeg filter chain
    vf_filters = [f"fps={fps}"]
    if crop_rect is not None:
        x, y, w, h = crop_rect
        vf_filters.append(f"crop={w}:{h}:{x}:{y}")

    vf = ",".join(vf_filters)

    tmp_dir = output_dir / "_ffmpeg_raw"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-qscale:v",
        "2" if image_format == "jpeg" else "0",
        str(tmp_dir / f"frame_%06d{ext}"),
        "-y",
        "-loglevel",
        "warning",
    ]
    logger.info("Running ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("ffmpeg exited with code %d", result.returncode)
        sys.exit(result.returncode)

    raw_frames = sorted(tmp_dir.glob(f"frame_*{ext}"))
    logger.info("ffmpeg extracted %d frames.", len(raw_frames))

    if not raw_frames:
        logger.error("No frames extracted from %s. Check the video file and fps setting.", video_path)
        sys.exit(1)

    kept_frames: list[Path] = []
    sharpness_values: list[float] = []
    skipped_blur = 0

    for raw_path in raw_frames:
        img = cv2.imread(str(raw_path))
        if img is None:
            logger.warning("Could not read frame %s, skipping.", raw_path)
            continue

        if blur_threshold is not None:
            score = compute_blur_score(img)
            sharpness_values.append(score)
            if blur_threshold > 0 and score < blur_threshold:
                skipped_blur += 1
                continue

        dst = output_dir / raw_path.name
        shutil.move(str(raw_path), str(dst))
        kept_frames.append(dst)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if skipped_blur > 0:
        logger.warning("Skipped %d blurry frames (Laplacian variance < %.1f).", skipped_blur, blur_threshold)
    if sharpness_values:
        hist_path = output_dir.parent / "blur_histogram.png"
        save_blur_histogram(sharpness_values, blur_threshold, hist_path)
        logger.info("Saved blur histogram to %s", hist_path)

    logger.info("Kept %d frames after filtering.", len(kept_frames))
    return kept_frames


# ---------------------------------------------------------------------------
# COLMAP / GLOMAP invocation
# ---------------------------------------------------------------------------


def run_colmap_feature_extractor(
    image_dir: Path,
    db_path: Path,
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    colmap_path: Optional[str] = None,
) -> None:
    """Run ``colmap feature_extractor``."""
    colmap = _find_executable("colmap", colmap_path)
    cmd = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("colmap feature_extractor exited with code %d", result.returncode)
        sys.exit(result.returncode)


def run_colmap_matcher(
    db_path: Path,
    matcher_type: str = "sequential",
    colmap_path: Optional[str] = None,
) -> None:
    """Run ``colmap sequential_matcher`` or ``colmap exhaustive_matcher``."""
    colmap = _find_executable("colmap", colmap_path)
    if matcher_type == "sequential":
        subcmd = "sequential_matcher"
    elif matcher_type == "exhaustive":
        subcmd = "exhaustive_matcher"
    elif matcher_type == "vocab_tree":
        subcmd = "vocab_tree_matcher"
    else:
        logger.error("Unknown matcher type: %s (expected sequential, exhaustive, or vocab_tree)", matcher_type)
        sys.exit(1)

    cmd = [colmap, subcmd, "--database_path", str(db_path)]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("colmap %s exited with code %d", subcmd, result.returncode)
        sys.exit(result.returncode)


def run_glomap_mapper(
    db_path: Path,
    image_dir: Path,
    sparse_dir: Path,
    glomap_path: Optional[str] = None,
) -> Path:
    """
    Run ``glomap mapper`` for global SfM. Returns the path to the sparse
    reconstruction (typically *sparse_dir*/0/).
    """
    glomap = _find_executable("glomap", glomap_path)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        glomap,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("glomap mapper exited with code %d", result.returncode)
        sys.exit(result.returncode)

    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        candidates = sorted(sparse_dir.iterdir())
        if candidates:
            model_dir = candidates[0]
        else:
            logger.error("No reconstruction output found in %s", sparse_dir)
            sys.exit(1)

    logger.info("GLOMAP reconstruction at %s", model_dir)
    return model_dir


def run_colmap_mapper(
    db_path: Path,
    image_dir: Path,
    sparse_dir: Path,
    colmap_path: Optional[str] = None,
) -> Path:
    """
    Run ``colmap mapper`` (incremental SfM). Returns the path to the sparse
    reconstruction (typically *sparse_dir*/0/).
    """
    colmap = _find_executable("colmap", colmap_path)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("colmap mapper exited with code %d", result.returncode)
        sys.exit(result.returncode)

    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        candidates = sorted(sparse_dir.iterdir())
        if candidates:
            model_dir = candidates[0]
        else:
            logger.error("No reconstruction output found in %s", sparse_dir)
            sys.exit(1)

    logger.info("COLMAP reconstruction at %s", model_dir)
    return model_dir


def run_mapper(
    db_path: Path,
    image_dir: Path,
    sparse_dir: Path,
    mapper: str = "glomap",
    colmap_path: Optional[str] = None,
    glomap_path: Optional[str] = None,
) -> Path:
    """
    Dispatch to glomap or colmap mapper based on *mapper* argument.
    Returns the path to the sparse model directory.
    """
    if mapper == "glomap":
        return run_glomap_mapper(db_path, image_dir, sparse_dir, glomap_path=glomap_path)
    elif mapper == "colmap":
        return run_colmap_mapper(db_path, image_dir, sparse_dir, colmap_path=colmap_path)
    else:
        logger.error("Unknown mapper: %s (expected glomap or colmap)", mapper)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Output arrangement
# ---------------------------------------------------------------------------


def arrange_colmap_output(image_dir: Path, model_dir: Path, output_dir: Path) -> None:
    """
    Create the directory layout expected by ``frgs reconstruct``::

        output_dir/
            images/          -- frame images
            sparse/0/        -- cameras.bin, images.bin, points3D.bin

    *image_dir* is the directory containing extracted frames.
    *model_dir* is the COLMAP/GLOMAP sparse model directory (containing .bin files).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dst = output_dir / "images"
    if images_dst.exists():
        if images_dst.is_symlink():
            images_dst.unlink()
        else:
            shutil.rmtree(images_dst)
    shutil.copytree(str(image_dir), str(images_dst))

    sparse_dst = output_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)

    for item in model_dir.iterdir():
        if item.is_file():
            dst_file = sparse_dst / item.name
            if dst_file.exists():
                dst_file.unlink()
            shutil.copy2(str(item), str(dst_file))

    logger.info("COLMAP dataset ready at %s", output_dir)
    logger.info("  images/ -> %d files", len(list(images_dst.rglob("*"))))
    logger.info("  sparse/0/ -> %d files", len(list(sparse_dst.iterdir())))


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------


def detect_split_screen_boundary(video_path: Path, ffmpeg_path: Optional[str] = None) -> tuple[int, int, int, int]:
    """
    Sample a frame from the video and attempt to detect the vertical split
    boundary for a side-by-side layout. Returns (x, y, w, h) for the right
    half (RGB portion).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Could not read a sample frame from %s", video_path)
        sys.exit(1)

    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    col_var = np.var(np.diff(gray.astype(np.float32), axis=1), axis=0)

    center_region = col_var[w // 4 : 3 * w // 4]
    split_col = int(np.argmax(center_region) + w // 4)

    if abs(split_col - w // 2) > w // 10:
        logger.warning(
            "Detected split at x=%d, but expected near x=%d. Falling back to exact midpoint.",
            split_col,
            w // 2,
        )
        split_col = w // 2

    logger.info("Detected split-screen boundary at x=%d (frame size %dx%d).", split_col, w, h)
    return (split_col, 0, w - split_col, h)


def parse_crop_rect(crop_str: str) -> tuple[int, int, int, int]:
    """Parse a ``x,y,w,h`` string into a crop rectangle tuple."""
    parts = crop_str.split(",")
    if len(parts) != 4:
        raise ValueError(f"Expected x,y,w,h but got: {crop_str!r}")
    return tuple(int(p.strip()) for p in parts)  # type: ignore[return-value]
