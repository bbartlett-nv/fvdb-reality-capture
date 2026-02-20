# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# ray_caster.py — Generate a sparse point cloud by intersecting camera rays
# with a DEM surface.
#
# Section 4.2 of the work plan (Phase 2: Ray-Casting for Sparse Point Generation).
#
# Strategy:
#   1. Build a regular grid of pixel coordinates at ``DOWNSAMPLE_STRIDE``.
#   2. Un-project each pixel into a 3-D ray in the LTP/ENU frame.
#   3. Intersect each ray with the DEM (loaded as a rasterio dataset).
#   4. Collect the intersection points into ``points3D.txt`` format, keeping
#      the 2-D ↔ 3-D correspondences for ``images.txt``.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import rasterio
    from rasterio.transform import rowcol
except ImportError:
    rasterio = None  # type: ignore[assignment]

from coord_transform import (
    LTPReference,
    batch_ecef_to_geodetic,
    batch_geodetic_to_ecef,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_to_ecef,
    geodetic_to_ecef,
)
from parse_metadata import FramePose
from paths import DEM_NODATA, DOWNSAMPLE_STRIDE

logger = logging.getLogger(__name__)


# ──────────────────────────── DEM wrapper ──────────────────────────────────── #


class _DEMTile:
    """A single GeoTIFF DEM tile loaded into memory."""

    def __init__(self, dem_path: Path):
        ds = rasterio.open(dem_path)
        self.data = ds.read(1)
        self.transform = ds.transform
        self.bnd = ds.bounds  # BoundingBox(left, bottom, right, top)
        self.name = dem_path.name
        ds.close()

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.bnd.left <= lon <= self.bnd.right
            and self.bnd.bottom <= lat <= self.bnd.top
        )

    def sample(self, lon: float, lat: float) -> float | None:
        row, col = rowcol(self.transform, lon, lat)
        if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
            val = float(self.data[row, col])
            if val == DEM_NODATA or np.isnan(val):
                return None
            return val
        return None


class DEMSampler:
    """
    Wraps one or more GeoTIFF DEM tiles so we can query elevation at
    arbitrary geodetic / ECEF locations.  Multiple tiles are treated as a
    virtual mosaic — the first tile that covers the query point is used.
    """

    def __init__(self, dem_paths: Path | list[Path]):
        if rasterio is None:
            raise ImportError("rasterio is required for DEM sampling.")
        if isinstance(dem_paths, Path):
            dem_paths = [dem_paths]
        self.tiles: list[_DEMTile] = []
        for p in dem_paths:
            tile = _DEMTile(p)
            self.tiles.append(tile)
            logger.info(
                "Loaded DEM %s  (%d×%d, bounds=%.2f..%.2f lon, %.2f..%.2f lat)",
                tile.name,
                tile.data.shape[1],
                tile.data.shape[0],
                tile.bnd.left,
                tile.bnd.right,
                tile.bnd.bottom,
                tile.bnd.top,
            )
        if not self.tiles:
            raise FileNotFoundError("No DEM tiles loaded.")

    def bounds(self) -> tuple[float, float, float, float]:
        """Return (lon_min, lon_max, lat_min, lat_max) across all tiles."""
        lon_min = min(t.bnd.left for t in self.tiles)
        lon_max = max(t.bnd.right for t in self.tiles)
        lat_min = min(t.bnd.bottom for t in self.tiles)
        lat_max = max(t.bnd.top for t in self.tiles)
        return lon_min, lon_max, lat_min, lat_max

    def sample_lonlat(self, lon: float, lat: float) -> float | None:
        """Return elevation (m) at WGS-84 lon/lat, or None if outside all tiles."""
        for tile in self.tiles:
            if tile.contains(lon, lat):
                return tile.sample(lon, lat)
        return None

    def sample_lonlat_batch(
        self, lons: np.ndarray, lats: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized DEM lookup for N points.

        Returns (N,) float64 array with elevations; NaN for misses/nodata.
        """
        N = len(lons)
        result = np.full(N, np.nan, dtype=np.float64)

        for tile in self.tiles:
            # Mask of points within this tile's bounds
            mask = (
                (lons >= tile.bnd.left) & (lons <= tile.bnd.right)
                & (lats >= tile.bnd.bottom) & (lats <= tile.bnd.top)
                & np.isnan(result)  # skip already-resolved
            )
            if not mask.any():
                continue

            # Compute row/col using the inverse affine transform
            inv = ~tile.transform  # inverse affine
            col_f = inv.a * lons[mask] + inv.b * lats[mask] + inv.c
            row_f = inv.d * lons[mask] + inv.e * lats[mask] + inv.f
            cols = np.floor(col_f).astype(np.intp)
            rows = np.floor(row_f).astype(np.intp)

            h, w = tile.data.shape
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

            vals = np.full(mask.sum(), np.nan, dtype=np.float64)
            if valid.any():
                sampled = tile.data[rows[valid], cols[valid]].astype(np.float64)
                bad = (sampled == DEM_NODATA) | np.isnan(sampled)
                sampled[bad] = np.nan
                vals[valid] = sampled

            result[mask] = vals

        return result

    def sample_ecef(self, ecef: np.ndarray) -> float | None:
        """Return elevation at an ECEF point (converted internally to lon/lat)."""
        lat, lon, _ = ecef_to_geodetic(ecef)
        return self.sample_lonlat(lon, lat)

    def close(self):
        pass  # tiles were already closed after reading data into memory


# ──────────────────── Ray generation & intersection ───────────────────────── #


def _pixel_grid(width: int, height: int, stride: int) -> np.ndarray:
    """
    Generate an N×2 array of (u, v) pixel coordinates sampled at ``stride``.
    """
    us = np.arange(0, width, stride, dtype=np.float64)
    vs = np.arange(0, height, stride, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    return np.column_stack([uu.ravel(), vv.ravel()])


def _unproject_pixels(
    pixels: np.ndarray,
    K_inv: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project pixel coordinates into rays in the world (ENU) frame.

    Each pixel (u, v) maps to a normalised camera-frame direction:
        d_cam = K^{-1} · [u, v, 1]^T

    In world coordinates:
        origin  =  −R^T · t        (camera centre)
        direction = R^T · d_cam

    Parameters
    ----------
    pixels : (N, 2)
    K_inv  : (3, 3)
    R_wc   : (3, 3)  world-to-camera rotation
    t_wc   : (3,)    world-to-camera translation

    Returns
    -------
    origins    : (N, 3)  all identical (camera centre in world)
    directions : (N, 3)  unit ray directions in world
    """
    N = pixels.shape[0]
    # Homogeneous pixel coords
    ones = np.ones((N, 1), dtype=np.float64)
    p_hom = np.hstack([pixels, ones])  # (N, 3)

    # Camera-frame directions
    d_cam = (K_inv @ p_hom.T).T  # (N, 3)

    # Camera centre in world
    R_cw = R_wc.T  # camera-to-world rotation
    cam_centre = -R_cw @ t_wc  # (3,)

    # World-frame directions
    d_world = (R_cw @ d_cam.T).T  # (N, 3)
    norms = np.linalg.norm(d_world, axis=1, keepdims=True)
    d_world /= np.maximum(norms, 1e-12)

    origins = np.tile(cam_centre, (N, 1))
    return origins, d_world


def _intersect_ray_dem(
    origin: np.ndarray,
    direction: np.ndarray,
    ltp: LTPReference,
    dem: DEMSampler,
    cam_alt_m: float,
    n_steps: int = 64,
) -> np.ndarray | None:
    """
    Intersect a single ray with the DEM via coarse-to-fine stepping.

    With the LTP origin on the ground, cameras are at ~498 km in ENU-Z.
    We march ``t`` along the ray searching for the sign-change where the
    ray crosses the DEM surface.

    Parameters
    ----------
    cam_alt_m : float
        Approximate camera altitude above the LTP ground (metres).
        Used to set the search range:  [0.9 * alt, 1.1 * alt].

    Returns the ENU intersection point or ``None`` on miss.
    """
    # Search range: the ground is roughly cam_alt_m metres along the ray
    t_min = max(0.0, cam_alt_m * 0.85)
    t_max = cam_alt_m * 1.15

    ts = np.linspace(t_min, t_max, n_steps)
    prev_above: bool | None = None
    prev_t = ts[0]

    for t in ts:
        pt_enu = origin + t * direction
        # Convert ENU → ECEF → geodetic to query DEM
        pt_ecef = enu_to_ecef(pt_enu, ltp.origin_ecef, ltp.R_e2l)
        elev = dem.sample_ecef(pt_ecef)
        if elev is None:
            prev_above = None
            prev_t = t
            continue

        # Approximate surface height in ENU: we convert the surface point
        lat, lon, _ = ecef_to_geodetic(pt_ecef)
        surface_ecef = geodetic_to_ecef(lat, lon, elev)
        surface_enu = ecef_to_enu(surface_ecef, ltp.origin_ecef, ltp.R_e2l)

        above = pt_enu[2] > surface_enu[2]

        if prev_above is not None and prev_above and not above:
            # Sign change → refine with bisection
            lo, hi = prev_t, t
            for _ in range(16):
                mid = (lo + hi) / 2.0
                mid_enu = origin + mid * direction
                mid_ecef = enu_to_ecef(mid_enu, ltp.origin_ecef, ltp.R_e2l)
                mid_elev = dem.sample_ecef(mid_ecef)
                if mid_elev is None:
                    break
                lat_m, lon_m, _ = ecef_to_geodetic(mid_ecef)
                surf_ecef_m = geodetic_to_ecef(lat_m, lon_m, mid_elev)
                surf_enu_m = ecef_to_enu(
                    surf_ecef_m, ltp.origin_ecef, ltp.R_e2l
                )
                if mid_enu[2] > surf_enu_m[2]:
                    lo = mid
                else:
                    hi = mid
            return origin + (lo + hi) / 2.0 * direction

        prev_above = above
        prev_t = t

    return None


def _intersect_rays_dem_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    ltp: LTPReference,
    dem: DEMSampler,
    cam_alt_m: float,
    n_steps: int = 64,
    n_bisect: int = 16,
) -> np.ndarray:
    """
    Vectorized ray-DEM intersection for N rays simultaneously.

    Instead of looping over rays, this loops over t-values and processes
    all N rays in batch using vectorized NumPy operations.

    Parameters
    ----------
    origins    : (N, 3) ray origins in ENU
    directions : (N, 3) unit ray directions in ENU
    cam_alt_m  : approximate camera altitude (sets search range)
    n_steps    : coarse marching steps
    n_bisect   : bisection refinement steps

    Returns
    -------
    results : (N, 3) intersection points in ENU; rows are NaN for misses.
    """
    N = len(origins)
    results = np.full((N, 3), np.nan, dtype=np.float64)

    t_min = max(0.0, cam_alt_m * 0.85)
    t_max = cam_alt_m * 1.15
    ts = np.linspace(t_min, t_max, n_steps)

    # Per-ray state
    prev_above = np.full(N, False)
    prev_valid = np.full(N, False)
    prev_t = np.full(N, ts[0])
    found = np.zeros(N, dtype=bool)
    lo = np.zeros(N, dtype=np.float64)
    hi = np.zeros(N, dtype=np.float64)

    R_l2e = ltp.R_e2l.T

    for t in ts:
        active = ~found
        if not active.any():
            break

        # Batch compute candidate points in ENU
        pts_enu = origins[active] + t * directions[active]

        # Batch ENU → ECEF
        pts_ecef = (R_l2e @ pts_enu.T).T + ltp.origin_ecef

        # Batch ECEF → geodetic
        lats, lons, _ = batch_ecef_to_geodetic(pts_ecef)

        # Batch DEM sample
        elevs = dem.sample_lonlat_batch(lons, lats)
        elev_valid = ~np.isnan(elevs)

        # For valid elevations, compute surface height in ENU
        above = np.full(active.sum(), False)
        cur_valid = np.full(active.sum(), False)

        if elev_valid.any():
            surf_ecef = batch_geodetic_to_ecef(
                lats[elev_valid], lons[elev_valid], elevs[elev_valid],
            )
            surf_enu = (ltp.R_e2l @ (surf_ecef - ltp.origin_ecef).T).T
            above[elev_valid] = pts_enu[elev_valid, 2] > surf_enu[:, 2]
            cur_valid[elev_valid] = True

        # Detect sign changes among active rays
        sign_change = (
            prev_valid[active] & prev_above[active]
            & cur_valid & ~above
        )

        if sign_change.any():
            # Map back to global indices
            global_idx = np.where(active)[0][sign_change]
            lo[global_idx] = prev_t[active][sign_change]
            hi[global_idx] = t
            found[global_idx] = True

        # Update state for active rays
        prev_above[active] = above
        prev_valid[active] = cur_valid
        prev_t[active] = t

    # ── Bisection refinement for rays that found a crossing ──
    to_refine = np.where(found)[0]
    if len(to_refine) > 0:
        r_lo = lo[to_refine]
        r_hi = hi[to_refine]
        r_origins = origins[to_refine]
        r_dirs = directions[to_refine]

        for _ in range(n_bisect):
            mid = (r_lo + r_hi) * 0.5
            pts_enu = r_origins + mid[:, np.newaxis] * r_dirs
            pts_ecef = (R_l2e @ pts_enu.T).T + ltp.origin_ecef
            lats, lons, _ = batch_ecef_to_geodetic(pts_ecef)
            elevs = dem.sample_lonlat_batch(lons, lats)
            elev_valid = ~np.isnan(elevs)

            # Default: keep mid as-is for invalid DEM lookups
            above = np.ones(len(to_refine), dtype=bool)
            if elev_valid.any():
                surf_ecef = batch_geodetic_to_ecef(
                    lats[elev_valid], lons[elev_valid], elevs[elev_valid],
                )
                surf_enu = (ltp.R_e2l @ (surf_ecef - ltp.origin_ecef).T).T
                above[elev_valid] = pts_enu[elev_valid, 2] > surf_enu[:, 2]

            r_lo = np.where(above, mid, r_lo)
            r_hi = np.where(above, r_hi, mid)

        final_t = (r_lo + r_hi) * 0.5
        results[to_refine] = r_origins + final_t[:, np.newaxis] * r_dirs

    return results


def _diagnose_center_ray(
    pose: FramePose,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    ltp: LTPReference,
) -> None:
    """
    Shoot a single ray from the image centre and report where it
    lands in geodetic coordinates.  This diagnostic exposes rotation
    errors — if the reported lat/lon is far from the expected scene
    location, the rotation is wrong.
    """
    K = pose.intrinsic_matrix()
    K_inv = np.linalg.inv(K)
    center_px = np.array([[pose.img_width / 2.0, pose.img_height / 2.0]])

    origins, directions = _unproject_pixels(center_px, K_inv, R_wc, t_wc)
    origin = origins[0]
    d = directions[0]

    # March along the ray to find where it would hit z ≈ 0 (ENU ground)
    if abs(d[2]) > 1e-10:
        t_ground = -origin[2] / d[2]
    else:
        t_ground = 0.0

    ground_enu = origin + t_ground * d
    ground_ecef = enu_to_ecef(ground_enu, ltp.origin_ecef, ltp.R_e2l)
    lat, lon, alt = ecef_to_geodetic(ground_ecef)

    logger.info(
        "  Center-ray diagnostic for frame %d: "
        "cam_ENU=(%.0f, %.0f, %.0f)  "
        "dir_ENU=(%.4f, %.4f, %.4f)  "
        "ground hit → lat=%.4f° lon=%.4f° (t=%.0f m)",
        pose.image_id,
        origin[0], origin[1], origin[2],
        d[0], d[1], d[2],
        lat, lon, t_ground,
    )


# ──────────────────────── Public API ──────────────────────────────────────── #


def _sample_image_colors(
    image_path: Path,
    pixels: np.ndarray,
) -> np.ndarray:
    """
    Sample RGB values from a TIFF at the given pixel coordinates.

    Returns (M, 3) uint8 array.  Falls back to white if the image
    cannot be loaded.
    """
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    n = len(pixels)
    if img is None:
        return np.full((n, 3), 255, dtype=np.uint8)

    h, w = img.shape[:2]
    is_16bit = img.dtype != np.uint8

    # For 16-bit images compute percentile bounds for scaling
    if is_16bit:
        valid = img[img != 0]
        if len(valid) > 0:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
        else:
            vmin, vmax = 0.0, 65535.0
    else:
        vmin, vmax = 0.0, 255.0

    colors = np.full((n, 3), 255, dtype=np.uint8)
    for i in range(n):
        col = int(round(pixels[i, 0]))
        row = int(round(pixels[i, 1]))
        if not (0 <= col < w and 0 <= row < h):
            continue
        val = img[row, col]
        if is_16bit:
            if np.ndim(val) == 0:
                gray = int(np.clip((float(val) - vmin) / max(vmax - vmin, 1) * 255, 0, 255))
                colors[i] = [gray, gray, gray]
            else:
                scaled = np.clip((val.astype(np.float32) - vmin) / max(vmax - vmin, 1) * 255, 0, 255).astype(np.uint8)
                if len(scaled) >= 3:
                    colors[i] = [scaled[2], scaled[1], scaled[0]]  # BGR → RGB
                else:
                    colors[i] = scaled[0]
        else:
            if np.ndim(val) == 0:
                colors[i] = [val, val, val]
            elif len(val) >= 3:
                colors[i] = [val[2], val[1], val[0]]  # BGR → RGB
            else:
                colors[i] = val[0]
    return colors


def cast_rays_for_frame(
    pose: FramePose,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    ltp: LTPReference,
    dem: DEMSampler,
    stride: int = DOWNSAMPLE_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sparse 3-D points for a single frame by ray-casting against a DEM.

    Returns
    -------
    points_enu : (M, 3)  intersection points in ENU
    pixels     : (M, 2)  corresponding pixel coordinates
    colors     : (M, 3)  RGB sampled from source image at each pixel
    """
    # Log where the center ray lands (diagnoses rotation errors)
    _diagnose_center_ray(pose, R_wc, t_wc, ltp)

    pixels = _pixel_grid(pose.img_width, pose.img_height, stride)
    K = pose.intrinsic_matrix()
    K_inv = np.linalg.inv(K)

    origins, directions = _unproject_pixels(pixels, K_inv, R_wc, t_wc)

    # Camera altitude above LTP ground (used to set ray search range)
    cam_alt_m = origins[0, 2]  # ENU-Z of camera centre
    if cam_alt_m < 1000.0:
        logger.warning(
            "Frame %d: camera ENU-Z = %.1f m (expected ~498 km). "
            "Rotation or LTP origin may be incorrect.",
            pose.image_id,
            cam_alt_m,
        )
        # Fall back to nominal satellite altitude
        cam_alt_m = 498_000.0

    all_pts = _intersect_rays_dem_batch(
        origins, directions, ltp, dem, cam_alt_m=cam_alt_m,
    )

    # Filter to hits (non-NaN rows)
    hit_mask = ~np.isnan(all_pts[:, 0])
    if not hit_mask.any():
        logger.warning(
            "No DEM hits for frame %d (%s)", pose.image_id, pose.image_path.name
        )
        return np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 3))

    points_enu = all_pts[hit_mask]
    px_arr = pixels[hit_mask]

    # Sample real RGB from source TIFF at observation pixel locations
    colors = _sample_image_colors(pose.image_path, px_arr)

    logger.info(
        "Frame %d: %d / %d rays hit DEM",
        pose.image_id,
        hit_mask.sum(),
        len(pixels),
    )
    return points_enu, px_arr, colors


def cast_rays_all_frames(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    ltp: LTPReference,
    dem: DEMSampler,
    stride: int = DOWNSAMPLE_STRIDE,
) -> tuple[np.ndarray, list[list[tuple[float, float, int]]]]:
    """
    Aggregate 3-D points across all frames.

    Returns
    -------
    all_points : (P, 6)   [X, Y, Z, R, G, B]  in ENU
    observations : list[list[(px_x, px_y, point3d_id)]]
        Per-frame list of 2-D ↔ 3-D associations for ``images.txt``.
    """
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    observations: list[list[tuple[float, float, int]]] = []

    point_id_offset = 0

    for pose, R, t in zip(poses, rotations, translations):
        pts, pxs, cols = cast_rays_for_frame(pose, R, t, ltp, dem, stride)
        obs_for_frame = []
        for j in range(len(pts)):
            pid = point_id_offset + j + 1  # 1-based COLMAP IDs
            obs_for_frame.append((pxs[j, 0], pxs[j, 1], pid))
        observations.append(obs_for_frame)

        all_points.append(pts)
        all_colors.append(cols)
        point_id_offset += len(pts)

    if all_points:
        pts_arr = np.vstack(all_points)
        col_arr = np.vstack(all_colors)
    else:
        pts_arr = np.empty((0, 3))
        col_arr = np.empty((0, 3))

    # Combine [X, Y, Z, R, G, B]
    if len(pts_arr) > 0:
        combined = np.hstack([pts_arr, col_arr.astype(np.float64)])
    else:
        combined = np.empty((0, 6))

    logger.info("Total 3-D points from ray-casting: %d", len(combined))
    return combined, observations


# ──────────────── Multi-View Track Merging ─────────────────────────────── #


def merge_multiview_tracks(
    points: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    merge_radius_m: float | None = None,
    stride: int = DOWNSAMPLE_STRIDE,
) -> tuple[np.ndarray, list[list[tuple[float, float, int]]]]:
    """
    Build multi-view tracks by reprojecting each 3D point through every
    camera and recording additional observations where the point falls
    within the image bounds.

    After adding cross-view observations, spatially close points (within
    ``merge_radius_m``) that share observations in the same frame are
    merged into a single point with a combined track.

    Parameters
    ----------
    points : (P, 6)  [X, Y, Z, R, G, B]
    observations : per-frame observation lists (single-view from ray-casting)
    poses, rotations, translations : camera parameters
    merge_radius_m : spatial merge tolerance in metres.  Defaults to
        half the DEM grid spacing estimated from the stride and typical
        satellite GSD (~3 m/px).
    stride : pixel stride used during ray-casting (for default merge radius)

    Returns
    -------
    merged_points : (P', 6)  deduplicated points
    merged_observations : updated per-frame observation lists
    """
    if len(points) == 0:
        return points, observations

    n_points = len(points)

    if merge_radius_m is None:
        # Estimate from stride * GSD.  SuperDove GSD ~3 m → stride 200 = 600 m
        merge_radius_m = stride * 3.0 * 0.5

    logger.info(
        "Building multi-view tracks (merge radius = %.1f m) ...", merge_radius_m,
    )

    # --- Phase A: add cross-view observations ---
    # For each camera, project all points and record hits.
    # new_obs[frame_idx] will collect (px_x, px_y, pt3d_id) tuples.
    new_obs: list[list[tuple[float, float, int]]] = [list(fo) for fo in observations]

    # Track which (frame, point) pairs already exist to avoid duplicates
    existing: set[tuple[int, int]] = set()
    for fi, frame_obs in enumerate(observations):
        for _, _, pt3d_id in frame_obs:
            existing.add((fi, pt3d_id))

    added = 0
    n_frames = len(poses)
    for fi, (pose, R_wc, t_wc) in enumerate(zip(poses, rotations, translations)):
        logger.info(
            "  Frame %d/%d: projecting %d points ...", fi + 1, n_frames, n_points,
        )
        K = pose.intrinsic_matrix()
        w, h = pose.img_width, pose.img_height
        added_this_frame = 0

        for pi in range(n_points):
            pt3d_id = pi + 1
            if (fi, pt3d_id) in existing:
                continue

            p_cam = R_wc @ points[pi, :3] + t_wc
            if p_cam[2] <= 0:
                continue
            uv = K @ p_cam
            px = uv[0] / uv[2]
            py = uv[1] / uv[2]

            if 0 <= px < w and 0 <= py < h:
                new_obs[fi].append((float(px), float(py), pt3d_id))
                existing.add((fi, pt3d_id))
                added_this_frame += 1

        added += added_this_frame
        logger.info("    → %d new observations (running total: %d)", added_this_frame, added)

    logger.info("  Added %d cross-view observations.", added)

    # --- Phase B: grid-based spatial dedup (non-chaining) ---
    # Snap points to a 2D grid so that points in the same cell merge,
    # but merges cannot cascade across cell boundaries (unlike union-find).
    from collections import defaultdict

    cell_size = merge_radius_m * 2.0
    cell_x = np.floor(points[:, 0] / cell_size).astype(np.int64)
    cell_y = np.floor(points[:, 1] / cell_size).astype(np.int64)

    cell_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(n_points):
        cell_groups[(int(cell_x[i]), int(cell_y[i]))].append(i)

    old_to_new: dict[int, int] = {}
    merged_pts: list[np.ndarray] = []
    for new_id, (_, members) in enumerate(sorted(cell_groups.items())):
        row = points[members].mean(axis=0)
        merged_pts.append(row)
        new_pt3d_id = new_id + 1
        for m in members:
            old_to_new[m + 1] = new_pt3d_id  # old 1-based → new 1-based

    merged_points = np.array(merged_pts) if merged_pts else np.empty((0, 6))

    # Remap observations to new point IDs and deduplicate per-frame
    merged_observations: list[list[tuple[float, float, int]]] = []
    for frame_obs in new_obs:
        seen: set[int] = set()
        remapped: list[tuple[float, float, int]] = []
        for px_x, px_y, old_id in frame_obs:
            new_id = old_to_new.get(old_id)
            if new_id is None or new_id in seen:
                continue
            seen.add(new_id)
            remapped.append((px_x, px_y, new_id))
        merged_observations.append(remapped)

    n_merged = n_points - len(merged_points)
    # Compute track length stats
    track_counts: dict[int, int] = {}
    for frame_obs in merged_observations:
        for _, _, pid in frame_obs:
            track_counts[pid] = track_counts.get(pid, 0) + 1
    if track_counts:
        lengths = list(track_counts.values())
        avg_track = sum(lengths) / len(lengths)
        max_track = max(lengths)
        multi_view = sum(1 for v in lengths if v > 1)
    else:
        avg_track = max_track = multi_view = 0

    logger.info(
        "  Merged %d duplicate points (%d → %d unique).",
        n_merged, n_points, len(merged_points),
    )
    logger.info(
        "  Track stats: avg=%.1f views, max=%d views, %d/%d have multi-view tracks.",
        avg_track, max_track, multi_view, len(merged_points),
    )

    # Per-frame observation counts after merge (helps diagnose data loss)
    for fi, frame_obs in enumerate(merged_observations):
        if len(frame_obs) == 0:
            logger.warning("  Frame %d: 0 observations after merge!", fi + 1)
        else:
            logger.info("  Frame %d: %d observations after merge.", fi + 1, len(frame_obs))

    return merged_points, merged_observations
