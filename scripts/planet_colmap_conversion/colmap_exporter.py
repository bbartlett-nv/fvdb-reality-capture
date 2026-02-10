# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# colmap_exporter.py — Write COLMAP sparse-model text files
# (cameras.txt, images.txt, points3D.txt).
#
# Section 4.3 of the work plan (Phase 3: File Formatting for COLMAP).
# Follows the COLMAP output format specification (ref [15]).
#
# Key design decisions:
#   - Camera model: OPENCV  (id=4) → fx, fy, cx, cy, k1, k2, p1, p2
#   - All coordinates are in the Local Tangent Plane (ENU) scaled by
#     ``SCALE_FACTOR`` (default 1/1000 → km) to keep float32 numerics
#     stable for Gaussian Splatting (Section 7.1).
#   - A single shared camera is written (CAMERA_ID=1) with parameters
#     averaged across all frames, since the satellite has one physical
#     camera.  All images reference this single camera.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from coord_transform import rotation_to_quaternion
from parse_metadata import FramePose
from paths import SCALE_FACTOR

logger = logging.getLogger(__name__)


# ═══════════════════ cameras.txt ══════════════════════════════════════════ #

def write_cameras_txt(
    poses: Sequence[FramePose],
    output_path: Path,
    pinhole: bool = False,
    pinhole_K: np.ndarray | None = None,
    pinhole_width: int | None = None,
    pinhole_height: int | None = None,
) -> None:
    """
    Write ``cameras.txt`` in COLMAP text format.

    A single camera entry (CAMERA_ID=1) is written.

    Parameters
    ----------
    pinhole : bool
        If True, write a ``PINHOLE`` model (4 params: fx, fy, cx, cy)
        using the provided ``pinhole_K`` matrix.  Otherwise write the
        default ``OPENCV`` model (8 params including distortion).
    pinhole_K : np.ndarray (3, 3) or None
        Intrinsic matrix for the undistorted pinhole camera.  Required
        when ``pinhole=True``.
    pinhole_width, pinhole_height : int or None
        Dimensions of the undistorted images.  Required when ``pinhole=True``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        if pinhole and pinhole_K is not None:
            # PINHOLE model: fx fy cx cy
            fx = pinhole_K[0, 0]
            fy = pinhole_K[1, 1]
            cx = pinhole_K[0, 2]
            cy = pinhole_K[1, 2]
            w = pinhole_width or poses[0].img_width
            h = pinhole_height or poses[0].img_height
            f.write(
                f"1 PINHOLE {w} {h} "
                f"{fx:.10g} {fy:.10g} {cx:.10g} {cy:.10g}\n"
            )
            logger.info("Wrote %s  (1 shared PINHOLE camera)", output_path)
        else:
            # OPENCV model: fx fy cx cy k1 k2 p1 p2
            all_params = []
            for pose in poses:
                dist = pose.normalized_dist_coeffs()
                all_params.append([
                    pose.focal_length_px,   # fx
                    pose.focal_length_px,   # fy  (square pixels)
                    pose.cx,
                    pose.cy,
                    dist[0],   # k1  (normalised)
                    dist[1],   # k2  (normalised)
                    dist[2],   # p1
                    dist[3],   # p2
                ])
            avg_params = np.mean(all_params, axis=0)
            width = poses[0].img_width
            height = poses[0].img_height
            params_str = " ".join(f"{p:.10g}" for p in avg_params)
            f.write(f"1 OPENCV {width} {height} {params_str}\n")
            logger.info("Wrote %s  (1 shared OPENCV camera)", output_path)


# ═══════════════════ images.txt ══════════════════════════════════════════ #

def write_images_txt(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    observations: list[list[tuple[float, float, int]]],
    output_path: Path,
    scale: float = SCALE_FACTOR,
) -> None:
    """
    Write ``images.txt`` in COLMAP text format.

    Format (two lines per image):
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] as (X, Y, POINT3D_ID) ...

    The translation vector ``t`` is scaled by ``scale`` to match
    the coordinate normalisation applied to ``points3D.txt``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for pose, R_wc, t_wc, obs in zip(poses, rotations, translations, observations):
            # Quaternion [qw, qx, qy, qz]
            q = rotation_to_quaternion(R_wc)

            # Scale translation
            t_scaled = t_wc * scale

            # Line 1: pose (all images reference the single shared CAMERA_ID=1)
            f.write(
                f"{pose.image_id} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"{t_scaled[0]:.8f} {t_scaled[1]:.8f} {t_scaled[2]:.8f} "
                f"1 {pose.image_path.name}\n"
            )

            # Line 2: 2D observations
            parts = []
            for px_x, px_y, pt3d_id in obs:
                parts.append(f"{px_x:.2f} {px_y:.2f} {pt3d_id}")
            f.write(" ".join(parts) + "\n")

    logger.info("Wrote %s  (%d images)", output_path, len(poses))


# ═══════════════════ points3D.txt ════════════════════════════════════════ #

def write_points3d_txt(
    points: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    output_path: Path,
    scale: float = SCALE_FACTOR,
) -> None:
    """
    Write ``points3D.txt`` in COLMAP text format.

    Format per line:
        POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX) ...

    Parameters
    ----------
    points : (P, 6)  [X, Y, Z, R, G, B]  in ENU (metres)
    observations : per-frame observation lists (used to build tracks)
    scale : coordinate scale factor
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build reverse index:  point3d_id → list of (image_id, point2d_idx)
    track_map: dict[int, list[tuple[int, int]]] = {}
    for frame_idx, obs_list in enumerate(observations):
        image_id = frame_idx + 1  # 1-based
        for pt2d_idx, (_, _, pt3d_id) in enumerate(obs_list):
            track_map.setdefault(pt3d_id, []).append((image_id, pt2d_idx))

    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")

        for i, row in enumerate(points):
            pt_id = i + 1  # 1-based
            x, y, z = row[0] * scale, row[1] * scale, row[2] * scale
            r, g, b = int(row[3]), int(row[4]), int(row[5])
            error = 0.0  # placeholder reprojection error

            track = track_map.get(pt_id, [])
            track_str = " ".join(f"{im} {idx}" for im, idx in track)

            f.write(f"{pt_id} {x:.8f} {y:.8f} {z:.8f} {r} {g} {b} "
                    f"{error:.4f} {track_str}\n")

    logger.info("Wrote %s  (%d points)", output_path, len(points))


# ═══════════════════ Convenience wrapper ═════════════════════════════════ #

def export_colmap_model(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    points: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    output_dir: Path,
    scale: float = SCALE_FACTOR,
    pinhole: bool = False,
    pinhole_K: np.ndarray | None = None,
    pinhole_width: int | None = None,
    pinhole_height: int | None = None,
) -> None:
    """Write the full COLMAP sparse model (cameras + images + points3D)."""
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    write_cameras_txt(
        poses, sparse_dir / "cameras.txt",
        pinhole=pinhole,
        pinhole_K=pinhole_K,
        pinhole_width=pinhole_width,
        pinhole_height=pinhole_height,
    )
    write_images_txt(poses, rotations, translations, observations, sparse_dir / "images.txt", scale)
    write_points3d_txt(points, observations, sparse_dir / "points3D.txt", scale)

    logger.info("COLMAP model exported to %s", sparse_dir)
