# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# undistort.py — Image conversion, masking, and optional undistortion for
# the Planet SuperDove L1A → COLMAP pipeline.
#
# This module provides:
#   - ``convert_and_mask_images()`` — Convert raw 16-bit TIFFs to 8-bit
#     JPEGs and generate binary nodata masks (always called).
#   - ``undistort_images()`` — Undistort images and masks to a pinhole
#     camera model (called when ``--undistort`` is passed).
#   - ``remap_observations()`` — Remap 2D pixel observations from the
#     distorted to undistorted image plane.
#
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from parse_metadata import FramePose
from paths import IMG_NODATA, PIXEL_PITCH_UM, TARGET_MAX, TARGET_MIN, psblue_camera_models, SAT_ID

logger = logging.getLogger(__name__)

# "rem" coefficients describe the distortion to REMOVE from the raw image.
# We scale them from pixel domain to normalised coordinates the same way
# as the "add" coefficients, using  k_norm = k_px * f^(2n).
_REM_K1_PX = psblue_camera_models[SAT_ID]["rem"]["k1"]
_REM_K2_PX = psblue_camera_models[SAT_ID]["rem"]["k2"]


def _rem_dist_coeffs_normalised(focal_length_px: float) -> np.ndarray:
    """
    Return the "rem" distortion coefficients in normalised units
    suitable for ``cv2.undistort()``.

    OpenCV's ``undistort()`` *removes* distortion, which is exactly what
    the "rem" polynomial describes.
    """
    f = focal_length_px
    return np.array([
        _REM_K1_PX * f**2,   # k1
        _REM_K2_PX * f**4,   # k2
        0.0,                  # p1
        0.0,                  # p2
        0.0,                  # k3
    ], dtype=np.float64)


def _scale_to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Scale a 16-bit (or higher) image to 8-bit [TARGET_MIN, TARGET_MAX]
    using the provided value range.

    Parameters
    ----------
    vmin, vmax : float
        The low/high bounds for the linear stretch (typically from
        percentile-based statistics computed across all images).
    """
    if img.dtype == np.uint8:
        return img
    if vmax <= vmin:
        return np.full_like(img, TARGET_MIN, dtype=np.uint8)

    scaled = (img.astype(np.float32) - vmin) / (vmax - vmin)
    scaled = scaled * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
    return np.clip(scaled, TARGET_MIN, TARGET_MAX).astype(np.uint8)


def _compute_percentile_bounds(
    poses: list[FramePose],
    percentile: float = 2.0,
) -> tuple[float, float]:
    """
    Compute the average (low, high) percentile bounds across all images,
    using only valid (non-zero) pixels.

    ``percentile=2.0`` means the 2nd percentile (dark) and 98th percentile
    (bright) are used, clipping the darkest and brightest 2% of pixels.
    The per-image percentiles are averaged to produce a single shared
    ``(vmin, vmax)`` for consistent brightness across the strip.
    """
    lows: list[float] = []
    highs: list[float] = []

    for pose in poses:
        img = cv2.imread(str(pose.image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Extract valid (non-nodata) pixel values
        if img.ndim == 3:
            valid_mask = np.any(img != IMG_NODATA, axis=2)
            valid_vals = img[valid_mask]
        else:
            valid_vals = img[img != IMG_NODATA]
        if len(valid_vals) == 0:
            continue
        lows.append(float(np.percentile(valid_vals, percentile)))
        highs.append(float(np.percentile(valid_vals, 100.0 - percentile)))

    vmin = float(np.mean(lows)) if lows else 0.0
    vmax = float(np.mean(highs)) if highs else 1.0
    return vmin, vmax


def convert_and_mask_images(
    poses: list[FramePose],
    output_dir: Path,
    percentile: float = 2.0,
) -> list[Path]:
    """
    Convert raw 16-bit L1A TIFFs to 8-bit JPEGs and generate binary
    nodata masks.

    Uses a two-pass approach for consistent brightness across the strip:

    **Pass 1**: Compute percentile-based scaling bounds averaged across
    all images.

    **Pass 2**: Apply the shared bounds to scale each image, write the
    8-bit JPEG and binary mask.

    Parameters
    ----------
    poses : list[FramePose]
    output_dir : Path
    percentile : float
        Clip percentile for the linear stretch (default 2.0 means the
        2nd and 98th percentiles are used as vmin/vmax).

    Returns
    -------
    image_paths : list[Path]
        Paths to the output 8-bit JPEGs (same order as ``poses``).
    """
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: compute shared percentile bounds across all images
    vmin, vmax = _compute_percentile_bounds(poses, percentile)
    logger.info(
        "Scaling bounds (p%.0f/p%.0f avg): vmin=%.1f  vmax=%.1f",
        percentile, 100 - percentile, vmin, vmax,
    )

    # Pass 2: convert each image using the shared bounds
    image_paths: list[Path] = []

    for pose in poses:
        src_path = pose.image_path
        dst_name = src_path.stem + ".jpg"
        dst_path = images_dir / dst_name
        mask_path = masks_dir / (src_path.stem + ".jpg")

        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read %s — skipping.", src_path)
            image_paths.append(src_path)
            continue

        # Build binary mask: valid (non-zero) pixels → 255
        if img.ndim == 3:
            valid = np.any(img != IMG_NODATA, axis=2)
        else:
            valid = img != IMG_NODATA
        mask = np.where(valid, np.uint8(255), np.uint8(0))

        # Scale 16-bit to 8-bit using shared percentile bounds
        scaled = _scale_to_uint8(img, vmin, vmax)

        cv2.imwrite(str(dst_path), scaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
        image_paths.append(dst_path)

    logger.info(
        "Converted %d images to 8-bit JPEG in %s, masks in %s",
        len(image_paths), images_dir, masks_dir,
    )
    return image_paths


def undistort_images(
    poses: list[FramePose],
    output_dir: Path,
) -> tuple[np.ndarray, int, int, list[Path]]:
    """
    Undistort all raw L1A TIFFs and write clean pinhole images.

    Parameters
    ----------
    poses : list[FramePose]
        Frame metadata (used for intrinsics and image paths).
    output_dir : Path
        Root output directory.  Images are written to ``output_dir/images/``.

    Returns
    -------
    new_K : np.ndarray (3, 3)
        The pinhole intrinsic matrix for the undistorted images.
    new_w : int
        Width of the undistorted (cropped) images.
    new_h : int
        Height of the undistorted (cropped) images.
    undistorted_paths : list[Path]
        Output file paths (one per frame, same order as ``poses``).
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Use the first frame's intrinsics (they're essentially identical)
    ref = poses[0]
    K = ref.intrinsic_matrix()
    dist = _rem_dist_coeffs_normalised(ref.focal_length_px)

    # Compute the optimal new camera matrix.
    # alpha=0 crops all black borders for the cleanest result.
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, dist, (ref.img_width, ref.img_height), alpha=0,
    )
    rx, ry, rw, rh = roi
    logger.info(
        "Undistort: new K  fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f  "
        "crop ROI=(%d,%d,%d,%d)",
        new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2],
        rx, ry, rw, rh,
    )

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute the undistort maps for nearest-neighbor mask remapping
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (ref.img_width, ref.img_height), cv2.CV_32FC1,
    )

    undistorted_paths: list[Path] = []

    for pose in poses:
        src_path = pose.image_path  # already an 8-bit JPEG from convert_and_mask_images
        dst_path = images_dir / src_path.name

        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("Could not read %s — skipping undistort.", src_path)
            undistorted_paths.append(src_path)
            continue

        # Undistort the image (bilinear, default)
        undistorted = cv2.undistort(img, K, dist, None, new_K)

        # Crop to the valid ROI
        if rw > 0 and rh > 0:
            undistorted = undistorted[ry : ry + rh, rx : rx + rw]

        cv2.imwrite(str(dst_path), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        undistorted_paths.append(dst_path)

        # Undistort the corresponding mask with nearest-neighbor interpolation
        mask_name = src_path.stem + ".jpg"
        mask_src = masks_dir / mask_name
        if mask_src.exists():
            mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Use remap with nearest-neighbor to keep binary values sharp
                mask_undist = cv2.remap(
                    mask, map1, map2, cv2.INTER_NEAREST,
                )
                if rw > 0 and rh > 0:
                    mask_undist = mask_undist[ry : ry + rh, rx : rx + rw]
                # Re-binarize to ensure strictly 0/255
                _, mask_undist = cv2.threshold(mask_undist, 128, 255, cv2.THRESH_BINARY)
                # Overwrite the distorted mask with the undistorted one
                mask_dst = masks_dir / (dst_path.stem + ".jpg")
                cv2.imwrite(str(mask_dst), mask_undist)
                # Remove the old distorted mask if the name changed
                if mask_dst != mask_src and mask_src.exists():
                    mask_src.unlink()

    new_w = rw if rw > 0 else ref.img_width
    new_h = rh if rh > 0 else ref.img_height

    # Adjust the principal point in new_K for the crop offset
    adjusted_K = new_K.copy()
    if rw > 0 and rh > 0:
        adjusted_K[0, 2] -= rx
        adjusted_K[1, 2] -= ry

    logger.info(
        "Undistorted %d images to %s  (%d×%d)",
        len(undistorted_paths), images_dir, new_w, new_h,
    )
    return adjusted_K, new_w, new_h, undistorted_paths


def remap_observations(
    observations: list[list[tuple[float, float, int]]],
    K_old: np.ndarray,
    dist_coeffs: np.ndarray,
    K_new: np.ndarray,
    crop_roi: tuple[int, int, int, int] | None = None,
) -> list[list[tuple[float, float, int]]]:
    """
    Remap 2D observations from the distorted image plane to the undistorted
    (pinhole) image plane.

    Parameters
    ----------
    observations : per-frame list of (px_x, px_y, point3d_id)
    K_old : 3×3 original (distorted) intrinsic matrix
    dist_coeffs : distortion coefficients in normalised units ("rem")
    K_new : 3×3 new pinhole intrinsic matrix (before crop adjustment)
    crop_roi : (x, y, w, h) ROI from getOptimalNewCameraMatrix, or None

    Returns
    -------
    remapped : same structure as ``observations`` with updated pixel coords
    """
    remapped: list[list[tuple[float, float, int]]] = []

    for frame_obs in observations:
        if not frame_obs:
            remapped.append([])
            continue

        # Gather all pixel coords for this frame
        pts = np.array(
            [[px_x, px_y] for px_x, px_y, _ in frame_obs],
            dtype=np.float64,
        ).reshape(-1, 1, 2)

        # cv2.undistortPoints maps distorted pixels → undistorted normalised,
        # then P (new camera matrix) maps back to pixel coords.
        undist_pts = cv2.undistortPoints(pts, K_old, dist_coeffs, P=K_new)
        undist_pts = undist_pts.reshape(-1, 2)

        # Apply crop offset if ROI was used
        if crop_roi is not None:
            rx, ry, _, _ = crop_roi
            undist_pts[:, 0] -= rx
            undist_pts[:, 1] -= ry

        new_obs = []
        for (ux, uy), (_, _, pt3d_id) in zip(undist_pts, frame_obs):
            new_obs.append((float(ux), float(uy), pt3d_id))
        remapped.append(new_obs)

    return remapped
