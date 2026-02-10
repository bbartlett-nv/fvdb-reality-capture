# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# rpc_check.py — Validate derived pinhole camera poses against Rational
# Polynomial Coefficient (RPC) projections.
#
# Section 5 of the work plan (Verification Strategy: The RPC Check).
#
# For a set of ground-truth 3-D points the RPC model provides a
# reference pixel projection.  We compare that to the projection through
# our derived pinhole + distortion model and report per-pixel residuals.
#
# Acceptance criterion (from the work plan):
#   mean residual  < 1.0 pixel   → excellent
#   mean residual  < 2.0 pixels  → acceptable
#   mean residual  > 5.0 pixels  → investigation required
#
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from coord_transform import (
    LTPReference,
    ecef_to_geodetic,
    enu_to_ecef,
)
from parse_metadata import FramePose

logger = logging.getLogger(__name__)


# ═══════════════════ RPC Projection (reference) ═══════════════════════════ #

def _rpc_project(
    rpc: dict,
    lon: float,
    lat: float,
    alt: float,
) -> tuple[float, float]:
    """
    Project a geodetic point through the RPC model to get (sample, line)
    pixel coordinates.

    The standard RPC model normalises inputs, evaluates a ratio of cubic
    polynomials, and then de-normalises:
        P = NUM / DEN   (for both line and sample)

    Reference: OGC RPC specification / Planet documentation.
    """
    # Normalise
    lon_off = rpc["longitude_offset"]
    lat_off = rpc["latitude_offset"]
    alt_off = rpc["height_offset"]
    lon_scale = rpc["longitude_scale"]
    lat_scale = rpc["latitude_scale"]
    alt_scale = rpc["height_scale"]

    P = (lon - lon_off) / lon_scale
    L = (lat - lat_off) / lat_scale
    H = (alt - alt_off) / alt_scale

    # 20-term cubic polynomial vector
    terms = _rpc_poly_terms(P, L, H)

    # Line (row)
    line_num = np.dot(rpc["line_numerator_coefficients"], terms)
    line_den = np.dot(rpc["line_denominator_coefficients"], terms)
    line_norm = line_num / line_den

    # Sample (column)
    samp_num = np.dot(rpc["sample_numerator_coefficients"], terms)
    samp_den = np.dot(rpc["sample_denominator_coefficients"], terms)
    samp_norm = samp_num / samp_den

    # De-normalise
    line = line_norm * rpc["line_scale"] + rpc["line_offset"]
    samp = samp_norm * rpc["sample_scale"] + rpc["sample_offset"]

    return samp, line  # (x, y) in pixel convention


def _rpc_poly_terms(P: float, L: float, H: float) -> np.ndarray:
    """20-term cubic polynomial basis for RPC."""
    return np.array([
        1, L, P, H,
        L * P, L * H, P * H,
        L ** 2, P ** 2, H ** 2,
        P * L * H,
        L ** 3, L * P ** 2, L * H ** 2, L ** 2 * P,
        P ** 3, P * H ** 2, L ** 2 * H, P ** 2 * H,
        H ** 3,
    ], dtype=np.float64)


# ═══════════════════ Pinhole Projection ═══════════════════════════════════ #

def _pinhole_project(
    point_enu: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: np.ndarray,
    dist_coeffs: list[float] | None = None,
) -> tuple[float, float] | None:
    """
    Project a 3-D ENU point through the pinhole + optional radial distortion.

    Parameters
    ----------
    point_enu : (3,) point in ENU / world
    R_wc, t_wc : world-to-camera rotation & translation
    K : 3×3 intrinsic matrix
    dist_coeffs : [k1, k2, p1, p2, k3]  in normalised units

    Returns
    -------
    (u, v) pixel coordinates, or None if behind camera.
    """
    # Transform to camera frame
    p_cam = R_wc @ point_enu + t_wc
    if p_cam[2] <= 0:
        return None

    # Normalised image coordinates
    x = p_cam[0] / p_cam[2]
    y = p_cam[1] / p_cam[2]

    # Apply radial/tangential distortion (OpenCV model)
    if dist_coeffs is not None and len(dist_coeffs) >= 2:
        k1, k2 = dist_coeffs[0], dist_coeffs[1]
        p1 = dist_coeffs[2] if len(dist_coeffs) > 2 else 0.0
        p2 = dist_coeffs[3] if len(dist_coeffs) > 3 else 0.0
        k3 = dist_coeffs[4] if len(dist_coeffs) > 4 else 0.0

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = x_d, y_d

    # Pixel coordinates via intrinsic matrix
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return u, v


# ═══════════════════ Validation Logic ═════════════════════════════════════ #

@dataclass
class RPCResidual:
    """Residual for a single test point."""
    point_enu: np.ndarray      # (3,)
    rpc_px: tuple[float, float]
    pinhole_px: tuple[float, float]
    error_px: float            # Euclidean pixel error


@dataclass
class RPCCheckResult:
    """Aggregated check results for one frame."""
    image_id: int
    image_name: str
    n_points: int
    mean_error: float
    max_error: float
    median_error: float
    residuals: list[RPCResidual]

    def summary(self) -> str:
        status = "✓ PASS" if self.mean_error < 2.0 else "✗ FAIL"
        return (
            f"[{status}] Frame {self.image_id} ({self.image_name}):  "
            f"mean={self.mean_error:.3f} px   max={self.max_error:.3f} px   "
            f"median={self.median_error:.3f} px   (n={self.n_points})"
        )


def check_rpc_vs_pinhole(
    pose: FramePose,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    ltp: LTPReference,
    test_points_enu: np.ndarray | None = None,
    n_grid: int = 20,
) -> RPCCheckResult | None:
    """
    Compare RPC and pinhole projections for a frame.

    If ``test_points_enu`` is not supplied, a grid of points at the mean
    scene elevation is generated automatically.

    Parameters
    ----------
    pose : FramePose
    R_wc, t_wc : world-to-camera transform
    ltp : LTPReference
    test_points_enu : (N, 3) optional explicit test points
    n_grid : int  number of grid points per axis (if generating)

    Returns
    -------
    RPCCheckResult or None if the frame has no RPC data.
    """
    if pose.rpc is None:
        logger.info("Frame %d has no RPC — skipping check.", pose.image_id)
        return None

    K = pose.intrinsic_matrix()
    dist_norm = pose.normalized_dist_coeffs()

    # Generate test points if not provided
    if test_points_enu is None:
        test_points_enu = _generate_test_grid(pose, R_wc, t_wc, K, ltp, n_grid)

    residuals: list[RPCResidual] = []

    for pt_enu in test_points_enu:
        # Pinhole projection
        ph = _pinhole_project(pt_enu, R_wc, t_wc, K, dist_norm)
        if ph is None:
            continue

        # Convert ENU → ECEF → geodetic for RPC
        pt_ecef = enu_to_ecef(pt_enu, ltp.origin_ecef, ltp.R_e2l)
        lat, lon, alt = ecef_to_geodetic(pt_ecef)

        try:
            rpc_px = _rpc_project(pose.rpc, lon, lat, alt)
        except Exception:
            continue

        err = np.sqrt((ph[0] - rpc_px[0]) ** 2 + (ph[1] - rpc_px[1]) ** 2)
        residuals.append(RPCResidual(
            point_enu=pt_enu,
            rpc_px=rpc_px,
            pinhole_px=ph,
            error_px=err,
        ))

    if not residuals:
        logger.warning("No valid residuals for frame %d", pose.image_id)
        return None

    errors = np.array([r.error_px for r in residuals])
    result = RPCCheckResult(
        image_id=pose.image_id,
        image_name=pose.image_path.name,
        n_points=len(residuals),
        mean_error=float(np.mean(errors)),
        max_error=float(np.max(errors)),
        median_error=float(np.median(errors)),
        residuals=residuals,
    )
    logger.info(result.summary())
    return result


def _generate_test_grid(
    pose: FramePose,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: np.ndarray,
    ltp: LTPReference,
    n: int,
) -> np.ndarray:
    """
    Create a grid of 3-D points in ENU at elevation = 0 (LTP surface),
    roughly covering the camera's field of view.
    """
    # Camera centre in ENU
    R_cw = R_wc.T
    cam_enu = -R_cw @ t_wc

    # Project image corners to get rough footprint at z ≈ 0
    # Use a flat plane at z = 0 in ENU
    corners_px = np.array([
        [0, 0],
        [pose.img_width, 0],
        [pose.img_width, pose.img_height],
        [0, pose.img_height],
    ], dtype=np.float64)

    footprint = []
    K_inv = np.linalg.inv(K)
    for px in corners_px:
        d_cam = K_inv @ np.array([px[0], px[1], 1.0])
        d_world = R_cw @ d_cam
        # Intersect with z = 0 plane
        if abs(d_world[2]) < 1e-10:
            continue
        t_val = -cam_enu[2] / d_world[2]
        if t_val < 0:
            continue
        footprint.append(cam_enu + t_val * d_world)

    if len(footprint) < 2:
        # Fallback: small grid around nadir
        x = np.linspace(-1000, 1000, n)
        y = np.linspace(-1000, 1000, n)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])

    fp = np.array(footprint)
    x = np.linspace(fp[:, 0].min(), fp[:, 0].max(), n)
    y = np.linspace(fp[:, 1].min(), fp[:, 1].max(), n)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n * n)])
