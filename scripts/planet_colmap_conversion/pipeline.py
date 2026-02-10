#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# pipeline.py — End-to-end orchestrator for converting Planet SuperDove L1A
# imagery into a COLMAP-compatible sparse model for 3D Gaussian Splatting.
#
# Usage:
#   python pipeline.py                            # use defaults from paths.py
#   python pipeline.py --source-dir /path/to/l1a  # override source
#   python pipeline.py --skip-rpc-check           # speed up dev iterations
#
# This script implements the full processing pipeline described in the work
# plan PDF (Sections 1–8):
#
#   Phase 1  — Metadata Parsing          (parse_metadata.py)
#   Phase 2  — Coordinate Transforms     (coord_transform.py)
#   Phase 3  — Ray-Casting → points3D    (ray_caster.py)
#   Phase 4  — RPC Validation            (rpc_check.py)
#   Phase 5  — COLMAP Export             (colmap_exporter.py)
#   Phase 6  — Diagnostics              (plot_errors.py)
#
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Pipeline modules (all live in the same package directory)
from colmap_exporter import export_colmap_model
from coord_transform import (
    LTPReference,
    compute_nadir_world_to_camera,
    compute_world_to_camera,
    ecef_to_enu,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from parse_metadata import FramePose, parse_all_frames
from paths import (
    DEM_DIR,
    DOWNSAMPLE_STRIDE,
    DST_DIR,
    SCALE_FACTOR,
    SOURCE_DIR,
    TARGET_RPC_TYPE,
)
from plot_errors import generate_all_diagnostics
from ray_caster import DEMSampler, cast_rays_all_frames
from rpc_check import RPCCheckResult, check_rpc_vs_pinhole

# ──────────────────────────── Logging ──────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ──────────────────────────── CLI ──────────────────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Planet SuperDove L1A imagery to a COLMAP sparse model.",
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory containing L1A TIFFs + *_metadata.json files.",
    )
    p.add_argument(
        "--dem-dir",
        type=Path,
        default=DEM_DIR,
        help="Directory containing GeoTIFF DEMs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DST_DIR,
        help="Root output directory for the COLMAP model.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=DOWNSAMPLE_STRIDE,
        help="Pixel stride for DEM ray-casting grid.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=SCALE_FACTOR,
        help="Coordinate scale factor (default: 1/1000 → km).",
    )
    p.add_argument(
        "--rpc-type",
        type=str,
        default=TARGET_RPC_TYPE,
        help="Which RPC block to use from metadata.",
    )
    p.add_argument(
        "--skip-rpc-check",
        action="store_true",
        help="Skip the RPC validation step.",
    )
    p.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip generating diagnostic plots.",
    )
    p.add_argument(
        "--use-nadir",
        action="store_true",
        help="Use nadir-pointing camera pose instead of the metadata "
        "quaternion.  Useful when the quaternion frame convention "
        "is unknown (e.g. ECI→Body vs ECEF→Body).",
    )
    p.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort raw L1A images to a pinhole camera model. "
        "Writes clean TIFFs to {output}/images/ and exports the "
        "COLMAP model with PINHOLE instead of OPENCV.",
    )
    return p.parse_args()


# ──────────────────────────── Pipeline Steps ──────────────────────────────── #


def step_parse_metadata(source_dir: Path, rpc_type: str) -> list[FramePose]:
    """Phase 1: Parse all L1A metadata files."""
    logger.info("═══ Phase 1: Parsing Metadata ═══")
    poses = parse_all_frames(source_dir, rpc_type=rpc_type)
    if not poses:
        logger.error("No frames found — aborting.")
        sys.exit(1)
    return poses


def step_compute_poses(
    poses: list[FramePose],
    use_nadir: bool = False,
) -> tuple[LTPReference, list[np.ndarray], list[np.ndarray]]:
    """
    Phase 2: Establish the Local Tangent Plane and compute world-to-camera
    transforms for every frame.
    """
    logger.info("═══ Phase 2: Computing Camera Poses ═══")
    if use_nadir:
        logger.info("  (Using nadir-pointing poses — quaternion bypassed)")

    # Choose the LTP origin as the sub-satellite ground point (altitude = 0).
    # The work plan (§2.1) specifies the LTP should be "centered at an
    # arbitrary reference point P_ref within the scene" — i.e. on the ground,
    # NOT at satellite altitude.  This ensures:
    #   - Cameras appear at ~498 km height in ENU (sanity check)
    #   - The DEM surface is near z ≈ 0 in ENU
    #   - float32 precision is preserved for ground-level coordinates
    all_ecef = np.array([p.C_sat for p in poses])
    centroid_ecef = all_ecef.mean(axis=0)
    lat_c, lon_c, _ = ecef_to_geodetic(centroid_ecef)
    ground_ecef = geodetic_to_ecef(lat_c, lon_c, 0.0)
    ltp = LTPReference.from_ecef(ground_ecef)
    logger.info(
        "LTP origin (ground): lat=%.6f°  lon=%.6f°  alt=%.1f m",
        ltp.lat_deg,
        ltp.lon_deg,
        ltp.alt_m,
    )

    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []

    for i, pose in enumerate(poses):
        if use_nadir:
            # Use position-only nadir-pointing pose
            if i + 1 < len(poses):
                next_ecef = poses[i + 1].C_sat
            elif i > 0:
                next_ecef = pose.C_sat + (pose.C_sat - poses[i - 1].C_sat)
            else:
                next_ecef = None
            R_wc, t_wc = compute_nadir_world_to_camera(
                pose.C_sat,
                ltp,
                C_sat_ecef_next=next_ecef,
            )
        else:
            # Pass acquisition_epoch so the quaternion is converted
            # from ECI→Body to ECEF→Body via Astropy GCRS→ITRS.
            # For velocity estimation, use forward difference where
            # possible, falling back to backward difference for the
            # last frame to avoid the arbitrary-perpendicular fallback
            # which would cause a ~90° roll discontinuity.
            if i + 1 < len(poses):
                next_ecef = poses[i + 1].C_sat
            elif i > 0:
                # Last frame: use backward difference (current - previous)
                # to keep velocity direction consistent with the strip.
                next_ecef = pose.C_sat + (pose.C_sat - poses[i - 1].C_sat)
            else:
                next_ecef = None
            R_wc, t_wc = compute_world_to_camera(
                pose.q_sat,
                pose.C_sat,
                ltp,
                epoch_unix=pose.acquisition_epoch,
                C_sat_ecef_next=next_ecef,
            )
        rotations.append(R_wc)
        translations.append(t_wc)

        # Log camera height above LTP for sanity (should be ~475 km)
        cam_enu = ecef_to_enu(pose.C_sat, ltp.origin_ecef, ltp.R_e2l)
        logger.info(
            "  Frame %d — camera height above LTP: %.1f km",
            pose.image_id,
            cam_enu[2] / 1000.0,
        )

    return ltp, rotations, translations


def step_raycast(
    poses: list[FramePose],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    ltp: LTPReference,
    dem_dir: Path,
    stride: int,
) -> tuple[np.ndarray, list[list[tuple[float, float, int]]]]:
    """Phase 3: Ray-cast through DEM to create the sparse point cloud."""
    logger.info("═══ Phase 3: Ray-Casting for Sparse Points ═══")

    # Load ALL DEM tiles in the directory (they may cover different regions)
    dem_files = sorted(dem_dir.glob("*.tif")) + sorted(dem_dir.glob("*.tiff"))
    if not dem_files:
        logger.error("No DEM files found in %s", dem_dir)
        sys.exit(1)

    dem = DEMSampler(dem_files)

    # Log coverage diagnostics
    logger.info(
        "DEM mosaic coverage: lon=[%.2f, %.2f]  lat=[%.2f, %.2f]",
        *dem.bounds(),
    )
    logger.info(
        "LTP ground point:    lon=%.4f  lat=%.4f",
        ltp.lon_deg,
        ltp.lat_deg,
    )

    try:
        points, observations = cast_rays_all_frames(
            poses,
            rotations,
            translations,
            ltp,
            dem,
            stride=stride,
        )
    finally:
        dem.close()

    return points, observations


def step_rpc_validation(
    poses: list[FramePose],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    ltp: LTPReference,
) -> list[RPCCheckResult | None]:
    """Phase 4: Validate poses against RPC ground truth."""
    logger.info("═══ Phase 4: RPC Validation ═══")
    results: list[RPCCheckResult | None] = []
    for pose, R, t in zip(poses, rotations, translations):
        r = check_rpc_vs_pinhole(pose, R, t, ltp)
        results.append(r)
    return results


def step_export(
    poses: list[FramePose],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    points: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    output_dir: Path,
    scale: float,
    pinhole: bool = False,
    pinhole_K: np.ndarray | None = None,
    pinhole_width: int | None = None,
    pinhole_height: int | None = None,
) -> None:
    """Phase 5: Write COLMAP text files."""
    logger.info("═══ Phase 5: Exporting COLMAP Model ═══")
    export_colmap_model(
        poses,
        rotations,
        translations,
        points,
        observations,
        output_dir,
        scale=scale,
        pinhole=pinhole,
        pinhole_K=pinhole_K,
        pinhole_width=pinhole_width,
        pinhole_height=pinhole_height,
    )


def step_diagnostics(
    poses: list[FramePose],
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    rpc_results: list[RPCCheckResult | None],
    points: np.ndarray | None,
    output_dir: Path,
    scale: float,
) -> None:
    """Phase 6: Generate diagnostic visualisations."""
    logger.info("═══ Phase 6: Generating Diagnostics ═══")
    generate_all_diagnostics(
        poses,
        rotations,
        translations,
        rpc_results,
        points,
        output_dir,
        scale=scale,
    )


# ──────────────────────────── Main ─────────────────────────────────────────── #


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    logger.info("Planet SuperDove L1A → COLMAP Pipeline")
    logger.info("  Source:  %s", args.source_dir)
    logger.info("  DEM:     %s", args.dem_dir)
    logger.info("  Output:  %s", args.output_dir)
    logger.info("  Scale:   %g", args.scale)
    logger.info("  Stride:  %d", args.stride)

    # Phase 1 — Metadata
    poses = step_parse_metadata(args.source_dir, args.rpc_type)

    # Phase 2 — Coordinate transforms
    ltp, rotations, translations = step_compute_poses(poses, use_nadir=args.use_nadir)

    # Phase 3 — Ray-casting
    points, observations = step_raycast(
        poses,
        rotations,
        translations,
        ltp,
        args.dem_dir,
        args.stride,
    )

    # Phase 4 — RPC validation (optional)
    rpc_results: list[RPCCheckResult | None] = []
    if not args.skip_rpc_check:
        rpc_results = step_rpc_validation(poses, rotations, translations, ltp)
    else:
        logger.info("═══ Phase 4: RPC Validation — SKIPPED ═══")

    # Phase 4.25 — Convert images to 8-bit JPEG and generate nodata masks
    from undistort import convert_and_mask_images

    logger.info("═══ Phase 4.25: Converting Images & Generating Masks ═══")
    image_paths = convert_and_mask_images(poses, args.output_dir)
    for pose, ipath in zip(poses, image_paths):
        pose.image_path = ipath

    # Phase 4.5 — Undistort images and masks (optional)
    pinhole = False
    pinhole_K = None
    pinhole_w = None
    pinhole_h = None

    if args.undistort:
        from undistort import remap_observations, undistort_images

        logger.info("═══ Phase 4.5: Undistorting Images & Masks ═══")

        # Undistort 8-bit JPEGs → pinhole images (also undistorts masks)
        pinhole_K, pinhole_w, pinhole_h, undist_paths = undistort_images(
            poses,
            args.output_dir,
        )

        # Remap 2D observations to the undistorted image plane
        ref = poses[0]
        K_old = ref.intrinsic_matrix()
        # Use the "rem" coefficients (normalised) for the remap — they
        # describe the distortion that was removed.
        from undistort import _rem_dist_coeffs_normalised

        dist_rem = _rem_dist_coeffs_normalised(ref.focal_length_px)

        # Get the un-cropped new K and ROI for observation remapping
        import cv2

        new_K_full, roi = cv2.getOptimalNewCameraMatrix(
            K_old,
            dist_rem,
            (ref.img_width, ref.img_height),
            alpha=0,
        )
        observations = remap_observations(
            observations,
            K_old,
            dist_rem,
            new_K_full,
            crop_roi=roi,
        )

        # Update pose image paths to point to undistorted files
        for pose, upath in zip(poses, undist_paths):
            pose.image_path = upath

        pinhole = True

    # Phase 5 — Export COLMAP model
    step_export(
        poses,
        rotations,
        translations,
        points,
        observations,
        args.output_dir,
        args.scale,
        pinhole=pinhole,
        pinhole_K=pinhole_K,
        pinhole_width=pinhole_w,
        pinhole_height=pinhole_h,
    )

    # Phase 6 — Diagnostics (optional)
    if not args.skip_diagnostics:
        step_diagnostics(
            poses,
            rotations,
            translations,
            rpc_results,
            points,
            args.output_dir,
            args.scale,
        )
    else:
        logger.info("═══ Phase 6: Diagnostics — SKIPPED ═══")

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline completed in %.1f s", elapsed)


if __name__ == "__main__":
    main()
