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
    colors     : (M, 3)  placeholder RGB (white)
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

    hit_pts = []
    hit_px = []
    for i in range(len(pixels)):
        pt = _intersect_ray_dem(
            origins[i], directions[i], ltp, dem, cam_alt_m=cam_alt_m
        )
        if pt is not None:
            hit_pts.append(pt)
            hit_px.append(pixels[i])

    if not hit_pts:
        logger.warning(
            "No DEM hits for frame %d (%s)", pose.image_id, pose.image_path.name
        )
        return np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 3))

    points_enu = np.array(hit_pts, dtype=np.float64)
    px_arr = np.array(hit_px, dtype=np.float64)
    colors = np.full((len(hit_pts), 3), 255, dtype=np.uint8)

    logger.info(
        "Frame %d: %d / %d rays hit DEM",
        pose.image_id,
        len(hit_pts),
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
