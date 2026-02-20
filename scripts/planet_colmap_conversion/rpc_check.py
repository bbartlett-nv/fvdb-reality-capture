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
from coord_transform import LTPReference, ecef_to_geodetic, enu_to_ecef
from parse_metadata import FramePose

logger = logging.getLogger(__name__)


# ═══════════════════ Vectorized NumPy RPC Projection ══════════════════════ #


def _parse_rpc_coeffs(rpc: dict) -> tuple[
    float, float, float, float, float, float,
    float, float, float, float,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Extract scalar offsets/scales and 20-element coefficient arrays from an RPC dict."""
    lon_off = float(rpc["LONG_OFF"])
    lat_off = float(rpc["LAT_OFF"])
    h_off = float(rpc["HEIGHT_OFF"])
    lon_sc = float(rpc["LONG_SCALE"])
    lat_sc = float(rpc["LAT_SCALE"])
    h_sc = float(rpc["HEIGHT_SCALE"])
    line_off = float(rpc["LINE_OFF"])
    line_sc = float(rpc["LINE_SCALE"])
    samp_off = float(rpc["SAMP_OFF"])
    samp_sc = float(rpc["SAMP_SCALE"])

    def _c(key: str) -> np.ndarray:
        v = rpc[key]
        if isinstance(v, str):
            return np.array(v.split(), dtype=np.float64)
        return np.asarray(v, dtype=np.float64)

    ln = _c("LINE_NUM_COEFF")
    ld = _c("LINE_DEN_COEFF")
    sn = _c("SAMP_NUM_COEFF")
    sd = _c("SAMP_DEN_COEFF")
    return (lon_off, lat_off, h_off, lon_sc, lat_sc, h_sc,
            line_off, line_sc, samp_off, samp_sc, ln, ld, sn, sd)


def _rpc_monomials(P: np.ndarray, L: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Build the (N, 20) RPC-00B monomial matrix for normalised (P, L, H).

    Term order follows the OGC RPC-00B / NITF standard:
      1, L, P, H, LP, LH, PH, L², P², H²,
      PLH, L³, LP², LH², L²P, P³, PH², L²H, P²H, H³
    """
    ones = np.ones_like(P)
    return np.column_stack([
        ones,          # 0
        L,             # 1
        P,             # 2
        H,             # 3
        L * P,         # 4
        L * H,         # 5
        P * H,         # 6
        L * L,         # 7
        P * P,         # 8
        H * H,         # 9
        P * L * H,     # 10
        L * L * L,     # 11
        L * P * P,     # 12
        L * H * H,     # 13
        L * L * P,     # 14
        P * P * P,     # 15
        P * H * H,     # 16
        L * L * H,     # 17
        P * P * H,     # 18
        H * H * H,     # 19
    ])


def rpc_forward_batch(
    rpc: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    alts: np.ndarray,
) -> np.ndarray:
    """Vectorized RPC forward projection: geodetic (lon, lat, alt) -> pixel (col, row).

    Parameters
    ----------
    rpc : dict
        RPC coefficient dictionary (Planet uppercase keys, string values).
    lons, lats, alts : (N,) arrays
        Geodetic coordinates in degrees / metres.

    Returns
    -------
    (N, 2) array of [col, row] pixel coordinates.
    """
    (lon_off, lat_off, h_off, lon_sc, lat_sc, h_sc,
     line_off, line_sc, samp_off, samp_sc, ln, ld, sn, sd) = _parse_rpc_coeffs(rpc)

    L = (np.asarray(lons, dtype=np.float64) - lon_off) / lon_sc
    P = (np.asarray(lats, dtype=np.float64) - lat_off) / lat_sc
    H = (np.asarray(alts, dtype=np.float64) - h_off) / h_sc

    M = _rpc_monomials(P, L, H)  # (N, 20)

    row = M @ ln / (M @ ld) * line_sc + line_off
    col = M @ sn / (M @ sd) * samp_sc + samp_off

    # GDAL uses pixel-is-area convention: (0,0) = top-left corner of the
    # first pixel, whereas the RPC standard uses pixel-center.  GDAL adds
    # +0.5 internally when converting from RPC to its pixel coordinates.
    return np.column_stack([col + 0.5, row + 0.5])


def rpc_forward(
    rpc: dict, lon: float, lat: float, alt: float,
) -> tuple[float, float]:
    """Scalar wrapper: returns (col, row)."""
    px = rpc_forward_batch(
        rpc,
        np.array([lon]),
        np.array([lat]),
        np.array([alt]),
    )
    return float(px[0, 0]), float(px[0, 1])


def rpc_inverse_batch(
    rpc: dict,
    cols: np.ndarray,
    rows: np.ndarray,
    alt_guess: float = 0.0,
    n_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RPC inverse projection: pixel (col, row) -> geodetic (lon, lat, alt).

    Uses Newton-Raphson iteration on the forward model with height fixed
    at *alt_guess* (same as GDAL's default behaviour).

    Returns
    -------
    (lons, lats, alts) each (N,).
    """
    (lon_off, lat_off, h_off, lon_sc, lat_sc, h_sc,
     line_off, line_sc, samp_off, samp_sc, ln, ld, sn, sd) = _parse_rpc_coeffs(rpc)

    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    N = len(cols)

    lons = np.full(N, lon_off)
    lats = np.full(N, lat_off)
    alts = np.full(N, alt_guess)

    eps = 1e-6  # for finite-difference Jacobian (degrees)

    for _ in range(n_iter):
        px = rpc_forward_batch(rpc, lons, lats, alts)  # (N, 2) — includes +0.5
        dc = cols - px[:, 0]
        dr = rows - px[:, 1]

        px_dlon = rpc_forward_batch(rpc, lons + eps, lats, alts)
        px_dlat = rpc_forward_batch(rpc, lons, lats + eps, alts)

        dc_dlon = (px_dlon[:, 0] - px[:, 0]) / eps
        dr_dlon = (px_dlon[:, 1] - px[:, 1]) / eps
        dc_dlat = (px_dlat[:, 0] - px[:, 0]) / eps
        dr_dlat = (px_dlat[:, 1] - px[:, 1]) / eps

        det = dc_dlon * dr_dlat - dc_dlat * dr_dlon
        safe = np.abs(det) > 1e-20
        det_safe = np.where(safe, det, 1.0)

        d_lon = (dc * dr_dlat - dr * dc_dlat) / det_safe
        d_lat = (dr * dc_dlon - dc * dr_dlon) / det_safe

        lons = np.where(safe, lons + d_lon, lons)
        lats = np.where(safe, lats + d_lat, lats)

    return lons, lats, alts


def rpc_inverse(
    rpc: dict, col: float, row: float, alt_guess: float = 0.0,
) -> tuple[float, float, float]:
    """Scalar wrapper: returns (lon, lat, alt)."""
    lo, la, al = rpc_inverse_batch(
        rpc,
        np.array([col]),
        np.array([row]),
        alt_guess=alt_guess,
    )
    return float(lo[0]), float(la[0]), float(al[0])


# ═══════════════════ GDAL Cross-Check (one-time) ═════════════════════════ #

_gdal_verified = False


def verify_rpc_against_gdal(poses: list[FramePose]) -> None:
    """Compare NumPy RPC projection against GDAL on non-outlier frames.

    Runs once per process.  For each frame whose TIFF-embedded RPCs match
    the JSON sidecar ``estimated_rpc`` (i.e. HEIGHT_OFF agrees), project
    a grid of test points through both implementations and log the max
    discrepancy.  Expected: < 0.01 px.
    """
    global _gdal_verified
    if _gdal_verified:
        return
    _gdal_verified = True

    try:
        from osgeo import gdal
        gdal.UseExceptions()
    except ImportError:
        logger.info("GDAL not available — skipping RPC cross-check.")
        return

    n_tested = 0
    max_fwd_err = 0.0
    max_inv_lon_err = 0.0
    max_inv_lat_err = 0.0

    for pose in poses:
        if pose.rpc is None:
            continue

        ds = gdal.Open(str(pose.image_path))
        if ds is None:
            continue
        tiff_rpc = ds.GetMetadata("RPC")
        tiff_h_off = tiff_rpc.get("HEIGHT_OFF", "")

        if tiff_h_off != str(pose.rpc.get("HEIGHT_OFF", "")):
            ds = None
            continue

        tr = gdal.Transformer(ds, None, ["METHOD=RPC"])

        rpc = pose.rpc
        h = float(rpc["HEIGHT_OFF"])
        lon0 = float(rpc["LONG_OFF"])
        lat0 = float(rpc["LAT_OFF"])
        lons = np.linspace(lon0 - 0.05, lon0 + 0.05, 3)
        lats = np.linspace(lat0 - 0.05, lat0 + 0.05, 3)
        lo, la = np.meshgrid(lons, lats)
        lo, la = lo.ravel(), la.ravel()
        alts = np.full(len(lo), h)

        np_px = rpc_forward_batch(rpc, lo, la, alts)
        for j in range(len(lo)):
            s, (gc, gr, _) = tr.TransformPoint(1, float(lo[j]), float(la[j]), float(alts[j]))
            if not s:
                continue
            fwd_err = max(abs(np_px[j, 0] - gc), abs(np_px[j, 1] - gr))
            max_fwd_err = max(max_fwd_err, fwd_err)

        test_c = np.array([float(rpc["SAMP_OFF"]), 1000.0])
        test_r = np.array([float(rpc["LINE_OFF"]), 2000.0])
        np_lo, np_la, _ = rpc_inverse_batch(rpc, test_c, test_r, alt_guess=0.0)
        for j in range(len(test_c)):
            s, (glon, glat, _) = tr.TransformPoint(0, float(test_c[j]), float(test_r[j]), 0)
            if not s:
                continue
            max_inv_lon_err = max(max_inv_lon_err, abs(np_lo[j] - glon))
            max_inv_lat_err = max(max_inv_lat_err, abs(np_la[j] - glat))

        ds = None
        n_tested += 1
        if n_tested >= 3:
            break

    if n_tested == 0:
        logger.info("RPC cross-check: no matching frames found.")
        return

    logger.info(
        "RPC cross-check (%d frames): forward max err=%.6f px, "
        "inverse max err=(%.8f° lon, %.8f° lat)",
        n_tested, max_fwd_err, max_inv_lon_err, max_inv_lat_err,
    )
    if max_fwd_err > 0.1:
        logger.warning("RPC cross-check: forward discrepancy %.3f px exceeds 0.1 px!", max_fwd_err)


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

    point_enu: np.ndarray  # (3,)
    rpc_px: tuple[float, float]
    pinhole_px: tuple[float, float]
    error_px: float  # Euclidean pixel error


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
    raw_image_path: str | None = None,  # kept for API compat, unused
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

    if test_points_enu is None:
        test_points_enu = _generate_test_grid(pose, R_wc, t_wc, K, ltp, n_grid)

    residuals: list[RPCResidual] = []

    for pt_enu in test_points_enu:
        ph = _pinhole_project(pt_enu, R_wc, t_wc, K, dist_norm)
        if ph is None:
            continue

        pt_ecef = enu_to_ecef(pt_enu, ltp.origin_ecef, ltp.R_e2l)
        lat, lon, alt = ecef_to_geodetic(pt_ecef)

        try:
            rpc_px = rpc_forward(pose.rpc, lon, lat, alt)
        except Exception:
            continue

        err = np.sqrt((ph[0] - rpc_px[0]) ** 2 + (ph[1] - rpc_px[1]) ** 2)
        residuals.append(
            RPCResidual(
                point_enu=pt_enu,
                rpc_px=rpc_px,
                pinhole_px=ph,
                error_px=err,
            )
        )

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
    corners_px = np.array(
        [
            [0, 0],
            [pose.img_width, 0],
            [pose.img_width, pose.img_height],
            [0, pose.img_height],
        ],
        dtype=np.float64,
    )

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


# ═══════════════════ Post-Export Consistency Check ════════════════════════ #


def verify_colmap_points_against_rpc(
    poses: list[FramePose],
    points: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    ltp: LTPReference,
    error_threshold: float = 5.0,
    raw_image_paths: list[str] | None = None,  # kept for API compat, unused
) -> dict:
    """
    Verify that all exported 3D/2D point pairs are consistent with the
    RPC ground truth.

    For each frame, iterates over its 2D observations, converts the
    corresponding 3D point from ENU to geodetic, projects through the
    RPC model, and compares with the stored 2D pixel coordinate.

    Parameters
    ----------
    poses : list[FramePose]
    points : (P, 6) array [X, Y, Z, R, G, B] in ENU
    observations : per-frame observation lists
    ltp : LTPReference
    error_threshold : float
        Points with residuals above this are flagged.

    Returns
    -------
    dict with keys:
        total_checked, total_passed, total_failed,
        mean_error, max_error, per_frame (list of per-frame stats)
    """
    all_errors: list[float] = []
    n_failed = 0
    per_frame: list[dict] = []

    for i, pose in enumerate(poses):
        if pose.rpc is None:
            per_frame.append({"image_id": pose.image_id, "n_checked": 0, "skipped": True})
            continue

        frame_obs = observations[i] if i < len(observations) else []
        frame_errors: list[float] = []

        for px_x, px_y, pt3d_id in frame_obs:
            idx = pt3d_id - 1  # 1-based → 0-based
            if idx < 0 or idx >= len(points):
                continue

            pt_enu = points[idx, :3]
            pt_ecef = enu_to_ecef(pt_enu, ltp.origin_ecef, ltp.R_e2l)
            lat, lon, alt = ecef_to_geodetic(pt_ecef)

            try:
                rpc_px = rpc_forward(pose.rpc, lon, lat, alt)
            except Exception:
                continue

            err = np.sqrt((px_x - rpc_px[0]) ** 2 + (px_y - rpc_px[1]) ** 2)
            frame_errors.append(err)
            all_errors.append(err)
            if err > error_threshold:
                n_failed += 1

        frame_info = {
            "image_id": pose.image_id,
            "image_name": pose.image_path.name,
            "n_checked": len(frame_errors),
        }
        if frame_errors:
            fe = np.array(frame_errors)
            frame_info["mean_error"] = float(np.mean(fe))
            frame_info["max_error"] = float(np.max(fe))
            frame_info["median_error"] = float(np.median(fe))
        per_frame.append(frame_info)

    result = {
        "total_checked": len(all_errors),
        "total_passed": len(all_errors) - n_failed,
        "total_failed": n_failed,
        "error_threshold": error_threshold,
        "per_frame": per_frame,
    }
    if all_errors:
        ae = np.array(all_errors)
        result["mean_error"] = float(np.mean(ae))
        result["max_error"] = float(np.max(ae))
        result["median_error"] = float(np.median(ae))
    else:
        result["mean_error"] = 0.0
        result["max_error"] = 0.0
        result["median_error"] = 0.0

    return result
