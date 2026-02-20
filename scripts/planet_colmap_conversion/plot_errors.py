# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# plot_errors.py — Diagnostic visualisations for the Planet L1A → COLMAP
# conversion pipeline.
#
# Section 6 of the work plan:
#   6.1  Residual Heatmap         (RPC vs Pinhole across the image plane)
#   6.2  World Axes Projection    (verify rotation convention)
#   6.3  Frustum Visualisation    (export PLY for MeshLab inspection)
#   6.6  Reprojection Overlay     (sparse points → image, RGB-colored)
#   6.7  Elevation Reprojection   (sparse points → image, elevation-coded)
#   6.8  Cross-Camera Reproj.     (all points through each camera)
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from coord_transform import LTPReference
from parse_metadata import FramePose
from rpc_check import RPCCheckResult

logger = logging.getLogger(__name__)

# Try to import matplotlib — if unavailable we simply skip plots.
try:
    import matplotlib
    matplotlib.use("Agg")          # headless backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════ 6.1  Residual Heatmap ════════════════════════════════ #

def plot_residual_heatmap(
    result: RPCCheckResult,
    output_path: Path,
    img_width: int = 8880,
    img_height: int = 5304,
) -> None:
    """
    Generate a 2-D heatmap of pixel residuals (RPC vs pinhole) overlaid on
    the image domain.

    Interpretation (from the work plan):
        - Uniform colour       → good alignment
        - Radial gradient      → incorrect k1/k2 scaling (intrinsics error)
        - Linear gradient      → boresight misalignment or timing offset
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available — skipping residual heatmap.")
        return

    xs = [r.pinhole_px[0] for r in result.residuals]
    ys = [r.pinhole_px[1] for r in result.residuals]
    errs = [r.error_px for r in result.residuals]

    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(xs, ys, c=errs, cmap="hot", s=10, norm=Normalize(0, max(errs)))
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # image convention: y-down
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Row (px)")
    ax.set_title(
        f"RPC vs Pinhole Residual — Frame {result.image_id}  "
        f"(mean={result.mean_error:.2f} px)"
    )
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Error (px)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved residual heatmap: %s", output_path)


# ══════════════════ 6.2  World Axes Projection ══════════════════════════ #

def plot_world_axes(
    pose: FramePose,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    output_path: Path,
    axis_length_m: float = 500.0,
) -> None:
    """
    Project East (1,0,0), North (0,1,0), Up (0,0,1) unit vectors from the
    LTP origin into the camera view and draw arrows on the image centre.

    Interpretation:
        - "Up" arrow should point towards camera (or vanish for nadir)
        - "North" should match satellite heading direction
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available — skipping world axes plot.")
        return

    K = pose.intrinsic_matrix()
    cx, cy = pose.cx, pose.cy

    # Project the origin and the three axis tips
    def project(pt_world: np.ndarray) -> np.ndarray | None:
        p_cam = R_wc @ pt_world + t_wc
        if p_cam[2] <= 0:
            return None
        uv = K @ p_cam
        return uv[:2] / uv[2]

    # Use the camera centre as the "origin" in the world
    R_cw = R_wc.T
    cam_enu = -R_cw @ t_wc

    origin_px = project(cam_enu)
    if origin_px is None:
        logger.warning("Camera origin projects behind camera — skipping axes plot.")
        return

    axes = {
        "East":  np.array([axis_length_m, 0, 0]),
        "North": np.array([0, axis_length_m, 0]),
        "Up":    np.array([0, 0, axis_length_m]),
    }
    colours = {"East": "red", "North": "green", "Up": "blue"}

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, pose.img_width)
    ax.set_ylim(pose.img_height, 0)
    ax.set_aspect("equal")
    ax.set_title(f"World Axes Projection — Frame {pose.image_id}")
    ax.scatter([cx], [cy], c="black", marker="+", s=100, zorder=5)

    for name, direction in axes.items():
        tip = cam_enu + direction
        tip_px = project(tip)
        if tip_px is None:
            continue
        dx = tip_px[0] - origin_px[0]
        dy = tip_px[1] - origin_px[1]
        ax.annotate(
            name,
            xy=(origin_px[0], origin_px[1]),
            xytext=(origin_px[0] + dx, origin_px[1] + dy),
            arrowprops=dict(arrowstyle="->", color=colours[name], lw=2),
            fontsize=12,
            color=colours[name],
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved world axes plot: %s", output_path)


# ══════════════════ 6.3  Frustum PLY Export ═════════════════════════════ #

def export_frustum_ply(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    points_enu: np.ndarray | None,
    output_path: Path,
    frustum_length: float = 10_000.0,
    scale: float = 1.0,
) -> None:
    """
    Write a PLY file containing camera frustums and the sparse point cloud
    for visual inspection in MeshLab.

    Section 6.3:
        Ensure cameras are positioned ~475 km above the point cloud,
        not inside it or below it.
    """
    vertices: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []

    for pose, R_wc, t_wc in zip(poses, rotations, translations):
        R_cw = R_wc.T
        cam_enu = -R_cw @ t_wc
        cam_enu_s = cam_enu * scale

        K = pose.intrinsic_matrix()
        K_inv = np.linalg.inv(K)

        # Four image corners in camera frame
        corners_px = np.array([
            [0, 0, 1],
            [pose.img_width, 0, 1],
            [pose.img_width, pose.img_height, 1],
            [0, pose.img_height, 1],
        ], dtype=np.float64)

        base_idx = len(vertices)
        vertices.append(cam_enu_s)  # camera centre

        for cpx in corners_px:
            d_cam = K_inv @ cpx
            d_cam = d_cam / np.linalg.norm(d_cam) * frustum_length
            tip_world = cam_enu + R_cw @ d_cam
            vertices.append(tip_world * scale)
            tip_idx = len(vertices) - 1
            edges.append((base_idx, tip_idx))

        # Connect corners
        for j in range(4):
            edges.append((base_idx + 1 + j, base_idx + 1 + (j + 1) % 4))

    # Add point cloud points (if any)
    if points_enu is not None and len(points_enu) > 0:
        for pt in points_enu:
            vertices.append(pt[:3] * scale)

    # Write PLY
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_verts = len(vertices)
    n_edges = len(edges)

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {n_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

    logger.info("Saved frustum PLY: %s  (%d verts, %d edges)", output_path, n_verts, n_edges)


# ══════════════════ 6.4  Per-Camera Elevation Maps ═══════════════════════ #


def generate_elevation_maps(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    ltp: LTPReference,
    dem: "DEMSampler",
    output_dir: Path,
    diag_stride: int = 15,
) -> None:
    """
    For each camera, project a grid of pixels to the ground via a
    flat-ground (z=0) approximation and sample DEM elevations.

    At ~498 km altitude with < 2 km terrain relief the flat-ground
    shortcut introduces negligible pixel-mapping error while being
    orders of magnitude faster than iterative ray-DEM intersection.
    """
    import cv2

    from coord_transform import batch_ecef_to_geodetic, enu_to_ecef
    from ray_caster import _pixel_grid, _unproject_pixels

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(poses)
    for fi, (pose, R_wc, t_wc) in enumerate(zip(poses, rotations, translations)):
        pixels = _pixel_grid(pose.img_width, pose.img_height, diag_stride)
        n_rays = len(pixels)
        logger.info(
            "Elevation map: frame %d/%d (id=%d, %d rays) ...",
            fi + 1, n_frames, pose.image_id, n_rays,
        )
        K = pose.intrinsic_matrix()
        K_inv = np.linalg.inv(K)

        origins, directions = _unproject_pixels(pixels, K_inv, R_wc, t_wc)

        # Flat-ground shortcut: find where each ray hits z=0 in ENU
        dz = directions[:, 2]
        valid_dz = np.abs(dz) > 1e-10
        t_ground = np.full(n_rays, np.nan)
        t_ground[valid_dz] = -origins[valid_dz, 2] / dz[valid_dz]

        # Keep only rays that hit the ground in front of the camera
        hit = valid_dz & (t_ground > 0)
        ground_enu = origins[hit] + t_ground[hit, np.newaxis] * directions[hit]

        # Batch ENU → ECEF → geodetic → DEM sample
        ground_ecef = enu_to_ecef(ground_enu, ltp.origin_ecef, ltp.R_e2l)
        lats, lons, _ = batch_ecef_to_geodetic(ground_ecef)
        elevations = dem.sample_lonlat_batch(lons, lats)

        n_dem_hits = int((~np.isnan(elevations)).sum())
        dem_bounds = dem.bounds()
        logger.info(
            "  Ground footprint: lon=[%.4f, %.4f]  lat=[%.4f, %.4f]  "
            "DEM hits: %d/%d  (DEM bounds: lon=[%.2f, %.2f] lat=[%.2f, %.2f])",
            float(lons.min()), float(lons.max()),
            float(lats.min()), float(lats.max()),
            n_dem_hits, len(lons),
            dem_bounds[0], dem_bounds[1], dem_bounds[2], dem_bounds[3],
        )

        # Build a small grid of elevations and resize to full resolution.
        # _pixel_grid uses meshgrid(us, vs) with row-major ravel, so the
        # flat array is in (rows, cols) order of the small grid.
        small_w = len(np.arange(0, pose.img_width, diag_stride))
        small_h = len(np.arange(0, pose.img_height, diag_stride))

        small_elev = np.full(small_h * small_w, np.nan, dtype=np.float32)
        # `hit` is a boolean mask on the full pixel array; map valid
        # elevations back to the corresponding positions in the flat grid.
        hit_indices = np.where(hit)[0]
        valid_elev = ~np.isnan(elevations)
        small_elev[hit_indices[valid_elev]] = elevations[valid_elev]
        small_elev = small_elev.reshape(small_h, small_w)

        out_path = diag_dir / f"elevation_frame_{pose.image_id:04d}.png"
        has_data = ~np.isnan(small_elev)
        if not np.any(has_data):
            empty = np.zeros((pose.img_height, pose.img_width, 3), dtype=np.uint8)
            cv2.imwrite(str(out_path), empty)
            logger.warning(
                "Saved elevation map (EMPTY — no DEM coverage): %s", out_path,
            )
            continue

        emin = float(np.nanmin(small_elev))
        emax = float(np.nanmax(small_elev))
        if emax <= emin:
            emax = emin + 1.0

        # Normalize to 0-1, fill NaN with 0
        norm = np.where(has_data, (small_elev - emin) / (emax - emin), 0.0)

        # Apply viridis colormap (via matplotlib) for intuitive color
        cmap = plt.get_cmap("viridis")
        colored_small = (cmap(norm.astype(np.float64))[:, :, :3] * 255).astype(np.uint8)
        colored_small[~has_data] = 0

        # Resize to full image resolution with nearest-neighbor
        colored_full = cv2.resize(
            colored_small,
            (pose.img_width, pose.img_height),
            interpolation=cv2.INTER_NEAREST,
        )
        # Convert RGB to BGR for cv2
        cv2.imwrite(str(out_path), colored_full[:, :, ::-1])
        logger.info(
            "Saved elevation map: %s  (elev range %.1f–%.1f m)",
            out_path,
            emin,
            emax,
        )


# ══════════════════ 6.5  XY Point Cloud Projection ═════════════════════ #


def generate_xy_projection(
    poses: Sequence[FramePose],
    points_enu: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    raw_image_paths: list[str],
    output_dir: Path,
    pixels_per_meter: float = 0.25,
) -> None:
    """
    Project all 3D points onto the XY (East-North) plane, colored by
    the actual RGB sampled from the satellite imagery at each observation
    pixel.

    Writes a single overview image: ``diagnostics/xy_point_cloud.png``.
    """
    import cv2

    if points_enu is None or len(points_enu) == 0:
        logger.warning("No 3D points — skipping XY projection.")
        return

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Collect (x_enu, y_enu, r, g, b) for every observed point
    pts_xy: list[tuple[float, float]] = []
    pts_rgb: list[tuple[int, int, int]] = []

    # Compute percentile scaling from the first image for consistent RGB
    first_img = cv2.imread(raw_image_paths[0], cv2.IMREAD_UNCHANGED)
    if first_img is not None and first_img.dtype != np.uint8:
        valid_vals = first_img[first_img != 0]
        if len(valid_vals) > 0:
            vmin = float(np.percentile(valid_vals, 2))
            vmax = float(np.percentile(valid_vals, 98))
        else:
            vmin, vmax = 0.0, 65535.0
    else:
        vmin, vmax = 0.0, 255.0

    for i, (pose, frame_obs) in enumerate(zip(poses, observations)):
        if not frame_obs:
            continue

        # Read the raw TIFF for this frame
        tif_path = raw_image_paths[i] if i < len(raw_image_paths) else None
        if tif_path is None:
            continue
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]

        for px_x, px_y, pt3d_id in frame_obs:
            idx = pt3d_id - 1
            if idx < 0 or idx >= len(points_enu):
                continue

            # Sample pixel from the image
            col = int(round(px_x))
            row = int(round(px_y))
            if 0 <= col < w and 0 <= row < h:
                val = img[row, col]
                # Scale to 0-255
                if img.dtype != np.uint8:
                    if np.ndim(val) == 0:
                        val_f = (float(val) - vmin) / max(vmax - vmin, 1)
                        gray = int(np.clip(val_f * 255, 0, 255))
                        r, g, b = gray, gray, gray
                    else:
                        scaled = (val.astype(np.float32) - vmin) / max(
                            vmax - vmin, 1
                        )
                        scaled = np.clip(scaled * 255, 0, 255).astype(np.uint8)
                        if len(scaled) >= 3:
                            b, g, r = int(scaled[0]), int(scaled[1]), int(scaled[2])
                        else:
                            r = g = b = int(scaled[0])
                else:
                    if np.ndim(val) == 0:
                        r = g = b = int(val)
                    elif len(val) >= 3:
                        b, g, r = int(val[0]), int(val[1]), int(val[2])
                    else:
                        r = g = b = int(val[0])
            else:
                r = g = b = 128  # out of bounds fallback

            pts_xy.append((points_enu[idx, 0], points_enu[idx, 1]))
            pts_rgb.append((r, g, b))

    if not pts_xy:
        logger.warning("No sampled points — skipping XY projection.")
        return

    xs = np.array([p[0] for p in pts_xy])
    ys = np.array([p[1] for p in pts_xy])
    colors = np.array(pts_rgb, dtype=np.uint8)

    # Compute raster bounds
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    margin = 100.0  # metres
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    rw = int((xmax - xmin) * pixels_per_meter) + 1
    rh = int((ymax - ymin) * pixels_per_meter) + 1
    # Clamp to reasonable size
    rw = min(rw, 8000)
    rh = min(rh, 8000)

    canvas = np.zeros((rh, rw, 3), dtype=np.uint8)

    for (ex, ey), (r, g, b) in zip(pts_xy, pts_rgb):
        cx = int((ex - xmin) / (xmax - xmin) * (rw - 1))
        cy = int((ymax - ey) / (ymax - ymin) * (rh - 1))  # flip Y for image
        if 0 <= cx < rw and 0 <= cy < rh:
            canvas[cy, cx] = [r, g, b]

    out_path = diag_dir / "xy_point_cloud.png"
    # Convert RGB to BGR for cv2
    cv2.imwrite(str(out_path), canvas[:, :, ::-1])
    logger.info(
        "Saved XY point cloud: %s  (%d points, %d×%d px)",
        out_path,
        len(pts_xy),
        rw,
        rh,
    )


# ══════════════════ 6.6  Per-Camera Reprojection Overlay (RGB) ═══════════ #


def _project_point(
    point_enu: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float] | None:
    """Project a single ENU point into pixel coords. Returns None if behind camera."""
    p_cam = R_wc @ point_enu + t_wc
    if p_cam[2] <= 0:
        return None
    uv = K @ p_cam
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def generate_reprojection_overlay(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    points_enu: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    output_dir: Path,
    point_radius: int = 8,
) -> None:
    """
    6.6 — For each camera, reproject its observed 3D points back onto the
    image and draw them as filled circles colored by the sampled image RGB.

    Validates the full K → R_wc → t_wc → 3D → reproject chain and shows
    whether point coverage is spatially reasonable.
    """
    import cv2

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    n_frames = len(poses)

    for fi, (pose, R_wc, t_wc, frame_obs) in enumerate(
        zip(poses, rotations, translations, observations)
    ):
        if not frame_obs:
            continue

        logger.info(
            "RGB overlay: frame %d/%d (id=%d, %d observations) ...",
            fi + 1, n_frames, pose.image_id, len(frame_obs),
        )

        img_path = image_dir / pose.image_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not load image %s — skipping RGB overlay.", img_path)
            continue

        canvas = img.copy()
        K = pose.intrinsic_matrix()
        h, w = img.shape[:2]
        n_drawn = 0

        for obs_x, obs_y, pt3d_id in frame_obs:
            idx = pt3d_id - 1
            if idx < 0 or idx >= len(points_enu):
                continue

            proj = _project_point(points_enu[idx, :3], R_wc, t_wc, K)
            if proj is None:
                continue
            px, py = int(round(proj[0])), int(round(proj[1]))
            if not (0 <= px < w and 0 <= py < h):
                continue

            ox, oy = int(round(obs_x)), int(round(obs_y))
            if 0 <= ox < w and 0 <= oy < h:
                bgr = img[oy, ox].tolist()
            else:
                bgr = [128, 128, 128]

            cv2.circle(canvas, (px, py), point_radius + 2, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (px, py), point_radius, bgr, thickness=-1, lineType=cv2.LINE_AA)
            n_drawn += 1

        out_path = diag_dir / f"reproject_rgb_{pose.image_id:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        logger.info(
            "Saved RGB reprojection overlay: %s  (%d points drawn)", out_path, n_drawn,
        )


# ══════════════════ 6.7  Per-Camera Elevation Reprojection ══════════════ #


def generate_elevation_reprojection(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    points_enu: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    output_dir: Path,
    point_radius: int = 8,
) -> None:
    """
    6.7 — Same as 6.6 but color-codes each projected point by its ENU-Z
    elevation using the viridis colormap.  A terrain gradient should be
    visible matching the expected topography.
    """
    import cv2

    if not HAS_MPL:
        logger.warning("matplotlib not available — skipping elevation reprojection.")
        return

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"

    # Compute global elevation range across all points for consistent coloring
    all_z = points_enu[:, 2]
    z_min, z_max = float(all_z.min()), float(all_z.max())
    if z_max <= z_min:
        z_max = z_min + 1.0

    cmap = plt.get_cmap("viridis")
    n_frames = len(poses)

    for fi, (pose, R_wc, t_wc, frame_obs) in enumerate(
        zip(poses, rotations, translations, observations)
    ):
        if not frame_obs:
            continue

        logger.info(
            "Elevation overlay: frame %d/%d (id=%d, %d observations) ...",
            fi + 1, n_frames, pose.image_id, len(frame_obs),
        )

        img_path = image_dir / pose.image_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not load image %s — skipping elevation overlay.", img_path)
            continue

        canvas = img.copy()
        K = pose.intrinsic_matrix()
        h, w = img.shape[:2]
        n_drawn = 0

        for _, _, pt3d_id in frame_obs:
            idx = pt3d_id - 1
            if idx < 0 or idx >= len(points_enu):
                continue

            pt = points_enu[idx, :3]
            proj = _project_point(pt, R_wc, t_wc, K)
            if proj is None:
                continue
            px, py = int(round(proj[0])), int(round(proj[1]))
            if not (0 <= px < w and 0 <= py < h):
                continue

            # Map elevation to color
            t_norm = (pt[2] - z_min) / (z_max - z_min)
            rgba = cmap(t_norm)
            bgr = [int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)]

            cv2.circle(canvas, (px, py), point_radius, bgr, thickness=-1, lineType=cv2.LINE_AA)
            n_drawn += 1

        out_path = diag_dir / f"reproject_elev_{pose.image_id:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        logger.info(
            "Saved elevation reprojection: %s  (%d points, elev range %.1f–%.1f m)",
            out_path, n_drawn, z_min, z_max,
        )


# ══════════════════ 6.8  Cross-Camera Reprojection Check ═══════════════ #


def generate_crosscam_reprojection(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    points_enu: np.ndarray,
    observations: list[list[tuple[float, float, int]]],
    output_dir: Path,
    point_radius: int = 6,
) -> None:
    """
    6.8 — For each camera, project ALL 3D points (from all cameras) and
    draw them on the image.

    - Green circles: points originally observed by this camera
    - Blue circles:  points from other cameras that reproject in-bounds

    If blue points cluster away from where green points are, there is a
    pose inconsistency between cameras.
    """
    import cv2

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"

    # Build set of point IDs per frame for quick lookup
    own_ids_per_frame: list[set[int]] = []
    for frame_obs in observations:
        own_ids_per_frame.append({pt3d_id for _, _, pt3d_id in frame_obs})

    n_frames = len(poses)
    n_pts = len(points_enu)

    for fi, (pose, R_wc, t_wc) in enumerate(zip(poses, rotations, translations)):
        logger.info(
            "Cross-cam: frame %d/%d (id=%d, projecting %d points) ...",
            fi + 1, n_frames, pose.image_id, n_pts,
        )
        img_path = image_dir / pose.image_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not load image %s — skipping cross-cam overlay.", img_path)
            continue

        canvas = img.copy()
        K = pose.intrinsic_matrix()
        h, w = img.shape[:2]
        own_ids = own_ids_per_frame[fi]
        n_own, n_cross = 0, 0

        # Draw cross-camera points first (blue, underneath)
        for i in range(len(points_enu)):
            pt3d_id = i + 1
            if pt3d_id in own_ids:
                continue
            proj = _project_point(points_enu[i, :3], R_wc, t_wc, K)
            if proj is None:
                continue
            px, py = int(round(proj[0])), int(round(proj[1]))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), point_radius, (255, 160, 0), thickness=-1, lineType=cv2.LINE_AA)
                n_cross += 1

        # Draw own-camera points on top (green)
        for i in range(len(points_enu)):
            pt3d_id = i + 1
            if pt3d_id not in own_ids:
                continue
            proj = _project_point(points_enu[i, :3], R_wc, t_wc, K)
            if proj is None:
                continue
            px, py = int(round(proj[0])), int(round(proj[1]))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), point_radius, (0, 220, 0), thickness=-1, lineType=cv2.LINE_AA)
                n_own += 1

        out_path = diag_dir / f"reproject_crosscam_{pose.image_id:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        logger.info(
            "Saved cross-camera reprojection: %s  (own=%d, cross=%d)",
            out_path, n_own, n_cross,
        )


# ══════════════════ Convenience: generate all diagnostics ═══════════════ #

def generate_all_diagnostics(
    poses: Sequence[FramePose],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    rpc_results: Sequence[RPCCheckResult | None],
    points_enu: np.ndarray | None,
    output_dir: Path,
    scale: float = 1.0,
) -> None:
    """Run all three diagnostic outputs."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 6.1 — Residual heatmaps (one per frame with RPC data)
    for result in rpc_results:
        if result is not None:
            plot_residual_heatmap(
                result,
                diag_dir / f"residual_heatmap_{result.image_id:04d}.png",
            )

    # 6.2 — World axes (first frame only, as a sanity check)
    if poses:
        plot_world_axes(
            poses[0], rotations[0], translations[0],
            diag_dir / "world_axes_frame1.png",
        )

    # 6.3 — Frustum + point cloud
    export_frustum_ply(
        poses, rotations, translations, points_enu,
        diag_dir / "frustums.ply",
        scale=scale,
    )
