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
