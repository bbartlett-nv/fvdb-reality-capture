# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Convert a SplatKing LiDAR-mode capture into a COLMAP dataset compatible with ``frgs reconstruct``.

SplatKing (https://radiancefields.com/splatking) exports per-frame JPEG images,
ARKit camera metadata (intrinsics + camera-to-world transform), LiDAR depth maps,
and an accumulated LiDAR point cloud.  This script reads the ``splatpack.json``
manifest, converts ARKit poses to COLMAP convention, and writes binary COLMAP
sparse files (cameras.bin, images.bin, points3D.bin).

Pipeline:
  Phase 1 -- Parse splatpack.json, filter frames by quality/tracking, copy images.
  Phase 2 -- Write COLMAP binary sparse files from ARKit poses.
  Phase 3 -- Arrange output into the images/ + sparse/0/ layout expected by frgs.

Requires:
  - numpy, scipy, Pillow, tqdm
"""

import argparse
import json
import logging
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ARKit uses a right-handed Y-up world frame.  COLMAP / fVDB viewer expect Z-up.
# 90-degree rotation around the X-axis: X -> X, Y -> Z, Z -> -Y
R_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# OpenGL camera convention (Y-up, -Z-forward) -> OpenCV/COLMAP (Y-down, Z-forward)
R_FLIP = np.diag([1.0, -1.0, -1.0])

COLMAP_PINHOLE_MODEL_ID = 1


def _load_image_rotated(image_path: Path, metadata_width: int, metadata_height: int) -> Image.Image:
    """
    Load a JPEG and ensure its pixel layout matches the ARKit metadata resolution.

    SplatKing may store JPEGs in the raw sensor orientation (e.g. 1440x1920
    portrait) without an EXIF orientation tag, while the ARKit metadata reports
    the logical resolution (e.g. 1920x1440 landscape).  This function applies
    EXIF transpose first, then rotates 90 degrees CW if the dimensions are
    still swapped.
    """
    img = ImageOps.exif_transpose(Image.open(image_path))
    if img.size[0] == metadata_height and img.size[1] == metadata_width:
        img = img.transpose(Image.Transpose.ROTATE_90)
    return img


# ---------------------------------------------------------------------------
# Phase 1: Parse SplatKing session
# ---------------------------------------------------------------------------


def _parse_arkit_intrinsics(flat: list[float]) -> dict:
    """
    Parse a flat 9-element ARKit intrinsics array (column-major) into fx, fy, cx, cy.

    ARKit stores the 3x3 intrinsics matrix in column-major order.  When read
    correctly the standard camera matrix ``K`` is recovered::

        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    K = np.array(flat).reshape(3, 3, order="F")
    return {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}


def _parse_arkit_c2w(flat: list[float]) -> np.ndarray:
    """
    Parse a flat 16-element ARKit transform array (column-major) into a 4x4
    camera-to-world matrix.
    """
    return np.array(flat).reshape(4, 4, order="F")


def parse_splatking_session(input_dir: Path, quality_threshold: float) -> dict:
    """
    Read ``splatpack.json`` (or fall back to per-frame JSON files), filter
    frames by quality score and tracking state, and return structured metadata.
    """
    splatpack_path = input_dir / "splatpack.json"
    if splatpack_path.exists():
        with open(splatpack_path) as f:
            manifest = json.load(f)
        pairs = manifest.get("pairs", [])
        logger.info("Loaded splatpack.json: %d pairs, schema %s.", len(pairs), manifest.get("schema", "?"))
    else:
        logger.info("No splatpack.json found; falling back to per-frame wide_*.json files.")
        pairs = None

    frames: list[dict] = []
    skipped_quality = 0
    skipped_tracking = 0
    skipped_no_image = 0

    if pairs is not None:
        for pair in pairs:
            stream = pair["streams"][0]
            image_file = stream["imageFile"]
            image_path = input_dir / image_file
            if not image_path.exists():
                skipped_no_image += 1
                continue

            camera = stream["metadata"]["camera"]
            extra = stream.get("extraMetadata", {})

            if camera.get("trackingState") != "normal":
                skipped_tracking += 1
                continue

            quality = extra.get("qualityScore", 1.0)
            if quality < quality_threshold:
                skipped_quality += 1
                continue

            intrinsics = _parse_arkit_intrinsics(camera["intrinsics"])
            c2w = _parse_arkit_c2w(camera["transform"])
            resolution = camera["imageResolution"]

            depth_info = None
            for aux in stream.get("auxiliaryOutputs", []):
                if aux.get("type") == "depth":
                    depth_info = {
                        "filename": aux["filename"],
                        "width": aux["width"],
                        "height": aux["height"],
                        "bytes_per_row": aux["bytesPerRow"],
                    }
                    break

            frames.append(
                {
                    "image_file": image_file,
                    "image_path": image_path,
                    "intrinsics": intrinsics,
                    "c2w": c2w,
                    "image_width": resolution["width"],
                    "image_height": resolution["height"],
                    "quality_score": quality,
                    "depth_info": depth_info,
                    "capture_id": pair.get("captureId", image_file),
                    "timestamp": pair.get("timestamp", float(len(frames))),
                }
            )
    else:
        json_files = sorted(input_dir.glob("wide_*.json"))
        for json_path in json_files:
            stem = json_path.stem
            image_path = input_dir / f"{stem}.jpg"
            if not image_path.exists():
                skipped_no_image += 1
                continue

            with open(json_path) as f:
                meta = json.load(f)

            camera = meta["metadata"]["camera"]
            extra = meta.get("extraMetadata", {})

            if camera.get("trackingState") != "normal":
                skipped_tracking += 1
                continue

            quality = extra.get("qualityScore", 1.0)
            if quality < quality_threshold:
                skipped_quality += 1
                continue

            intrinsics = _parse_arkit_intrinsics(camera["intrinsics"])
            c2w = _parse_arkit_c2w(camera["transform"])
            resolution = camera["imageResolution"]

            depth_info = None
            for aux in meta.get("auxiliaryOutputs", []):
                if aux.get("type") == "depth":
                    depth_info = {
                        "filename": aux["filename"],
                        "width": aux["width"],
                        "height": aux["height"],
                        "bytes_per_row": aux["bytesPerRow"],
                    }
                    break

            frames.append(
                {
                    "image_file": f"{stem}.jpg",
                    "image_path": image_path,
                    "intrinsics": intrinsics,
                    "c2w": c2w,
                    "image_width": resolution["width"],
                    "image_height": resolution["height"],
                    "quality_score": quality,
                    "depth_info": depth_info,
                    "capture_id": stem,
                    "timestamp": float(len(frames)),
                }
            )

    logger.info(
        "Parsed %d frames (skipped: %d quality, %d tracking, %d missing image).",
        len(frames),
        skipped_quality,
        skipped_tracking,
        skipped_no_image,
    )

    lidar_path = input_dir / "lidar_pointcloud_world_xyz.bin"
    has_lidar = lidar_path.exists()
    if has_lidar:
        logger.info("Found LiDAR point cloud: %s", lidar_path)

    return {
        "frames": frames,
        "input_dir": input_dir,
        "lidar_pointcloud_path": lidar_path if has_lidar else None,
    }


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------


def _load_lidar_xyz_yup(bin_path: Path) -> np.ndarray:
    """
    Read a SplatKing ``lidar_pointcloud_world_xyz.bin`` file.

    Returns an (N, 3) float32 array of XYZ coordinates in ARKit's Y-up world
    frame (before any coordinate conversion).
    """
    raw = np.fromfile(bin_path, dtype=np.float32)
    if len(raw) % 3 != 0:
        logger.warning("LiDAR point cloud size (%d floats) is not a multiple of 3; truncating.", len(raw))
        raw = raw[: len(raw) - len(raw) % 3]
    return raw.reshape(-1, 3)


def colorize_points_from_images(
    xyz_yup: np.ndarray,
    frames: list[dict],
    margin: int = 2,
) -> np.ndarray:
    """
    Assign RGB colors to 3D points by projecting them into camera images.

    For each frame, the function projects the points (in ARKit Y-up world
    coordinates) into the camera using the frame's intrinsics and c2w matrix,
    then samples RGB from the JPEG image.  Colors are averaged across all
    frames that observe a given point.

    Parameters
    ----------
    xyz_yup : (N, 3) float32
        Point positions in ARKit Y-up world frame.
    frames : list[dict]
        Parsed frame metadata (must contain ``c2w``, ``intrinsics``,
        ``image_path``, ``image_width``, ``image_height``).
    margin : int
        Pixel margin to keep points away from image edges.

    Returns
    -------
    (N, 3) float32 array of RGB values (0-255).
    """
    n_pts = len(xyz_yup)
    color_sum = np.zeros((n_pts, 3), dtype=np.float64)
    color_count = np.zeros(n_pts, dtype=np.int32)

    for frame in tqdm(frames, desc="Colorizing points"):
        c2w = frame["c2w"]
        w2c = np.linalg.inv(c2w)
        intr = frame["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        # Project into ARKit camera space (OpenGL: camera looks along -Z)
        pts_cam = (w2c[:3, :3] @ xyz_yup.T).T + w2c[:3, 3]

        # Points in front of the camera have negative Z in OpenGL convention
        in_front = pts_cam[:, 2] < -1e-3
        neg_z = -pts_cam[:, 2]

        u = fx * (pts_cam[:, 0] / neg_z) + cx
        v = fy * (pts_cam[:, 1] / neg_z) + cy

        img = np.array(_load_image_rotated(frame["image_path"], frame["image_width"], frame["image_height"]))
        img_h, img_w = img.shape[:2]

        visible = in_front & (u >= margin) & (u < img_w - margin) & (v >= margin) & (v < img_h - margin)
        if not np.any(visible):
            continue

        ui = u[visible].astype(np.int32)
        vi = v[visible].astype(np.int32)

        color_sum[visible] += img[vi, ui, :3].astype(np.float64)
        color_count[visible] += 1

    observed = color_count > 0
    rgb = np.full((n_pts, 3), 200.0, dtype=np.float32)
    rgb[observed] = (color_sum[observed] / color_count[observed, np.newaxis]).astype(np.float32)

    n_colored = int(observed.sum())
    pct = 100.0 * n_colored / max(n_pts, 1)
    logger.info("Colorized %d / %d points (%.1f%%) from %d frames.", n_colored, n_pts, pct, len(frames))
    return rgb


def load_lidar_pointcloud(bin_path: Path, frames: list[dict] | None = None) -> np.ndarray:
    """
    Read a SplatKing ``lidar_pointcloud_world_xyz.bin`` file and optionally
    colorize from camera images.

    When *frames* is provided, the LiDAR points are projected into each
    camera image to sample RGB colors.  Otherwise, points are assigned a
    neutral gray (200, 200, 200).

    Returns an (N, 6) float32 array with columns (x, y, z, r, g, b) in the
    Z-up world frame expected by COLMAP / fVDB viewer.
    """
    xyz_yup = _load_lidar_xyz_yup(bin_path)

    if frames:
        rgb = colorize_points_from_images(xyz_yup, frames)
    else:
        rgb = np.full((len(xyz_yup), 3), 200.0, dtype=np.float32)

    xyz_zup = (R_Y_UP_TO_Z_UP @ xyz_yup.T).T
    return np.column_stack([xyz_zup, rgb]).astype(np.float32)


def load_depth_map(bin_path: Path, width: int, height: int) -> np.ndarray:
    """
    Read a SplatKing depth ``.bin`` file (raw float32, row-major).

    Returns an (H, W) float32 array with depth values in meters.
    """
    raw = np.fromfile(bin_path, dtype=np.float32)
    bytes_per_row = width * 4
    expected_from_dims = width * height
    if len(raw) == expected_from_dims:
        return raw.reshape(height, width)

    # Handle bytesPerRow padding (1024 bytes per row for 256-wide depth)
    file_bytes = bin_path.stat().st_size
    actual_bpr = file_bytes // height
    stride_pixels = actual_bpr // 4
    if stride_pixels > width:
        padded = np.fromfile(bin_path, dtype=np.float32).reshape(height, stride_pixels)
        return padded[:, :width]

    return raw.reshape(height, width)


def backproject_depth_to_world(
    depth: np.ndarray,
    intrinsics: dict,
    c2w: np.ndarray,
    stride: int = 4,
    image_path: Path | None = None,
    image_width: int = 1920,
    image_height: int = 1440,
) -> np.ndarray:
    """
    Back-project a depth map to world-space 3D points (Z-up).

    Parameters
    ----------
    depth : (H_d, W_d) float32
        Depth in meters at the depth-map resolution.
    intrinsics : dict
        Camera intrinsics (fx, fy, cx, cy) at the *image* resolution.
    c2w : (4, 4)
        ARKit camera-to-world matrix (Y-up world).
    stride : int
        Sub-sample the depth grid by this factor.
    image_path : Path, optional
        If provided, sample RGB colors from the JPEG image.
    image_width, image_height : int
        Full image resolution (for scaling intrinsics to depth resolution).

    Returns
    -------
    (N, 6) float32 array: x, y, z, r, g, b in Z-up world frame.
    """
    dh, dw = depth.shape
    scale_x = dw / image_width
    scale_y = dh / image_height

    fx_d = intrinsics["fx"] * scale_x
    fy_d = intrinsics["fy"] * scale_y
    cx_d = intrinsics["cx"] * scale_x
    cy_d = intrinsics["cy"] * scale_y

    vs, us = np.mgrid[0:dh:stride, 0:dw:stride]
    us = us.ravel().astype(np.float32)
    vs = vs.ravel().astype(np.float32)
    zs = depth[vs.astype(int), us.astype(int)]

    valid = np.isfinite(zs) & (zs > 0.01) & (zs < 50.0)
    us, vs, zs = us[valid], vs[valid], zs[valid]

    xs_cam = (us - cx_d) / fx_d * zs
    ys_cam = (vs - cy_d) / fy_d * zs
    pts_cam = np.stack([xs_cam, ys_cam, zs], axis=-1)

    R_cw = c2w[:3, :3]
    t_cw = c2w[:3, 3]
    pts_world_yup = (R_cw @ pts_cam.T).T + t_cw
    pts_world_zup = (R_Y_UP_TO_Z_UP @ pts_world_yup.T).T

    if image_path is not None and image_path.exists():
        img = np.array(_load_image_rotated(image_path, image_width, image_height))
        ih, iw = img.shape[:2]
        img_us = (us / scale_x).astype(int).clip(0, iw - 1)
        img_vs = (vs / scale_y).astype(int).clip(0, ih - 1)
        rgb = img[img_vs, img_us, :3].astype(np.float32)
    else:
        rgb = np.full((len(pts_world_zup), 3), 200.0, dtype=np.float32)

    return np.column_stack([pts_world_zup, rgb]).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 2: Write COLMAP binary sparse files
# ---------------------------------------------------------------------------


def _arkit_c2w_to_colmap_w2c(c2w: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """
    Convert an ARKit camera-to-world matrix (Y-up world, OpenGL camera) to
    COLMAP world-to-camera quaternion and translation (Z-up world, OpenCV camera).

    Returns (qw, qx, qy, qz, t_wc) where t_wc is a 3-element array.
    """
    R_cw = c2w[:3, :3]
    t_cw = c2w[:3, 3]

    # Y-up -> Z-up world rotation
    rot_cw = Rotation.from_matrix(R_cw)
    rot_cw_zup = Rotation.from_matrix(R_Y_UP_TO_Z_UP @ rot_cw.as_matrix())
    t_cw_zup = R_Y_UP_TO_Z_UP @ t_cw

    # Camera-to-world -> world-to-camera
    rot_wc = rot_cw_zup.inv()
    t_wc = -rot_wc.as_matrix() @ t_cw_zup

    # OpenGL -> OpenCV camera convention
    rot_wc_colmap = Rotation.from_matrix(R_FLIP @ rot_wc.as_matrix())
    t_wc = R_FLIP @ t_wc

    quat_xyzw = rot_wc_colmap.as_quat()  # scipy: (x, y, z, w)
    qw, qx, qy, qz = float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])
    return qw, qx, qy, qz, t_wc


def _write_points3d_bin(path: Path, world_points: np.ndarray | None) -> int:
    """Write a COLMAP points3D.bin file.  Returns the number of points written."""
    with open(path, "wb") as f:
        if world_points is not None and len(world_points) > 0:
            f.write(struct.pack("<Q", len(world_points)))
            for pid, pt in enumerate(world_points):
                x, y, z, r, g, b = pt
                f.write(struct.pack("<Q", pid + 1))
                f.write(struct.pack("<ddd", float(x), float(y), float(z)))
                f.write(struct.pack("<BBB", int(r), int(g), int(b)))
                f.write(struct.pack("<d", 0.0))  # error
                f.write(struct.pack("<Q", 0))  # track_length
            return len(world_points)
        else:
            f.write(struct.pack("<Q", 0))
            return 0


def write_colmap_sparse(
    frames: list[dict],
    sparse_dir: Path,
    world_points: np.ndarray | None,
) -> Path:
    """
    Write COLMAP binary sparse files from parsed SplatKing frames.

    Uses a single shared PINHOLE camera (averaged intrinsics across all frames,
    since all images come from the same device lens).
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if not frames:
        logger.error("No frames to write.")
        sys.exit(1)

    w = frames[0]["image_width"]
    h = frames[0]["image_height"]

    fx = float(np.mean([f["intrinsics"]["fx"] for f in frames]))
    fy = float(np.mean([f["intrinsics"]["fy"] for f in frames]))
    cx = float(np.mean([f["intrinsics"]["cx"] for f in frames]))
    cy = float(np.mean([f["intrinsics"]["cy"] for f in frames]))

    # --- cameras.bin ---
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 0))  # camera_id
        f.write(struct.pack("<i", COLMAP_PINHOLE_MODEL_ID))
        f.write(struct.pack("<Q", w))
        f.write(struct.pack("<Q", h))
        f.write(struct.pack("<dddd", fx, fy, cx, cy))

    # --- images.bin ---
    num_images = len(frames)
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", num_images))
        for idx, frame in enumerate(frames):
            qw, qx, qy, qz, t_wc = _arkit_c2w_to_colmap_w2c(frame["c2w"])
            image_id = idx + 1
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", t_wc[0], t_wc[1], t_wc[2]))
            f.write(struct.pack("<I", 0))  # camera_id
            f.write(frame["image_file"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D

    # --- points3D.bin ---
    num_points = _write_points3d_bin(sparse_dir / "points3D.bin", world_points)

    logger.info(
        "Wrote COLMAP sparse files to %s (1 camera, %d images, %d points).",
        sparse_dir,
        num_images,
        num_points,
    )
    return sparse_dir


# ---------------------------------------------------------------------------
# cuSFM pose refinement (--refine)
# ---------------------------------------------------------------------------

SPLATKING_CAMERA_NAME = "wide"
SPLATKING_CAMERA_PARAMS_ID = "0"


def _quaternion_to_axis_angle_dict(quat_xyzw: np.ndarray) -> dict:
    """Convert a quaternion (x, y, z, w) to cuSFM axis-angle dict."""
    rot = Rotation.from_quat(quat_xyzw)
    rotvec = rot.as_rotvec()
    angle_rad = np.linalg.norm(rotvec)
    if angle_rad < 1e-12:
        return {"x": 0.0, "y": 0.0, "z": 1.0, "angle_degrees": 0.0}
    axis = rotvec / angle_rad
    return {
        "x": float(axis[0]),
        "y": float(axis[1]),
        "z": float(axis[2]),
        "angle_degrees": float(np.degrees(angle_rad)),
    }


def _build_camera_params(
    sensor_id: int,
    sensor_name: str,
    intrinsics: dict,
    image_width: int,
    image_height: int,
    sensor_to_vehicle: dict,
    frequency: int = 1,
) -> dict:
    """Build a single camera_params entry for cuSFM frames_meta.json."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    return {
        "sensor_meta_data": {
            "sensor_id": sensor_id,
            "sensor_type": "CAMERA",
            "sensor_name": sensor_name,
            "frequency": frequency,
            "sensor_to_vehicle_transform": sensor_to_vehicle,
        },
        "calibration_parameters": {
            "image_width": image_width,
            "image_height": image_height,
            "projection_matrix": {
                "data": [
                    fx, 0.0, cx, 0.0,
                    0.0, fy, cy, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
                "row_count": 3,
                "column_count": 4,
            },
        },
        "camera_projection_model_type": "PINHOLE",
    }


def generate_frames_meta(frames: list[dict], work_dir: Path) -> Path:
    """
    Write a ``frames_meta.json`` file in cuSFM KeyframesMetadataCollection
    format using parsed SplatKing frame data with ARKit pose priors.
    """
    identity_transform = {
        "axis_angle": {"x": 0.0, "y": 0.0, "z": 0.0, "angle_degrees": 0.0},
        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    avg_intrinsics = {
        "fx": float(np.mean([f["intrinsics"]["fx"] for f in frames])),
        "fy": float(np.mean([f["intrinsics"]["fy"] for f in frames])),
        "cx": float(np.mean([f["intrinsics"]["cx"] for f in frames])),
        "cy": float(np.mean([f["intrinsics"]["cy"] for f in frames])),
    }

    camera_params = _build_camera_params(
        sensor_id=0,
        sensor_name=SPLATKING_CAMERA_NAME,
        intrinsics=avg_intrinsics,
        image_width=frames[0]["image_width"],
        image_height=frames[0]["image_height"],
        sensor_to_vehicle=identity_transform,
    )

    keyframes = []
    for idx, frame in enumerate(frames):
        c2w = frame["c2w"]
        R_cw = c2w[:3, :3]
        t_cw = c2w[:3, 3]

        rot_cw_zup = Rotation.from_matrix(R_Y_UP_TO_Z_UP @ R_cw @ R_FLIP)
        t_cw_zup = R_Y_UP_TO_Z_UP @ t_cw

        axis_angle = _quaternion_to_axis_angle_dict(rot_cw_zup.as_quat())
        translation = {
            "x": float(t_cw_zup[0]),
            "y": float(t_cw_zup[1]),
            "z": float(t_cw_zup[2]),
        }
        camera_to_world = {"axis_angle": axis_angle, "translation": translation}

        timestamp_us = str(int(frame["timestamp"] * 1_000_000))
        keyframes.append(
            {
                "id": str(idx),
                "camera_params_id": SPLATKING_CAMERA_PARAMS_ID,
                "timestamp_microseconds": timestamp_us,
                "image_name": frame["image_file"],
                "camera_to_world": camera_to_world,
                "synced_sample_id": timestamp_us,
            }
        )

    meta = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {
            SPLATKING_CAMERA_PARAMS_ID: "0",
        },
        "camera_params_id_to_camera_params": {
            SPLATKING_CAMERA_PARAMS_ID: camera_params,
        },
    }

    out_path = work_dir / "frames_meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s (%d keyframe entries).", out_path, len(keyframes))
    return out_path


def run_cusfm(
    work_dir: Path,
    min_inter_frame_distance: float,
    min_inter_frame_rotation: float,
    extra_args: list[str] | None = None,
) -> Path:
    """Invoke ``cusfm_cli`` and return the path to the sparse output directory."""
    cusfm_base = work_dir / "cusfm"
    if cusfm_base.exists():
        shutil.rmtree(cusfm_base)
        logger.info("Cleaned previous cuSFM workspace: %s", cusfm_base)
    cmd = [
        "cusfm_cli",
        "--input_dir",
        str(work_dir),
        "--cusfm_base_dir",
        str(cusfm_base),
        "--export_binary_colmap_files",
        "--output_rgb",
        "--skip_cuvslam",
        f"--min_inter_frame_distance={min_inter_frame_distance}",
        f"--min_inter_frame_rotation_degrees={min_inter_frame_rotation}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running cuSFM: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("cusfm_cli exited with code %d", result.returncode)
        sys.exit(result.returncode)

    sparse_dir = cusfm_base / "sparse"
    if not sparse_dir.exists():
        logger.error("cuSFM sparse output not found at %s", sparse_dir)
        sys.exit(1)

    n_files = len(list(sparse_dir.iterdir()))
    logger.info("cuSFM finished. Sparse output at %s (%d files).", sparse_dir, n_files)
    return sparse_dir


def _copy_images_to_work_dir(frames: list[dict], work_dir: Path) -> Path:
    """Copy rotated images to work_dir/images/ for cuSFM consumption."""
    images_work = work_dir / "images"
    if images_work.exists():
        shutil.rmtree(images_work)
    images_work.mkdir(parents=True)

    meta_w = frames[0]["image_width"]
    meta_h = frames[0]["image_height"]
    logger.info("Copying %d rotated images to %s ...", len(frames), images_work)
    for frame in tqdm(frames, desc="Preparing images for cuSFM"):
        dst = images_work / frame["image_file"]
        if not dst.exists():
            img = _load_image_rotated(frame["image_path"], meta_w, meta_h)
            img.save(str(dst), quality=95)

    for frame in frames:
        link = work_dir / frame["image_file"]
        target = images_work / frame["image_file"]
        if not link.exists():
            link.symlink_to(target)

    return images_work


# ---------------------------------------------------------------------------
# Phase 3: Arrange COLMAP output
# ---------------------------------------------------------------------------


def arrange_colmap_output(
    input_dir: Path,
    frames: list[dict],
    sparse_dir: Path,
    output_dir: Path,
    images_source: Path | None = None,
) -> None:
    """
    Create the directory layout expected by ``frgs reconstruct``::

        output_dir/
            images/          -- JPEG images
            sparse/0/        -- cameras.bin, images.bin, points3D.bin

    When *images_source* is provided (e.g. work_dir/images/ with pre-rotated
    images from cuSFM), those images are copied directly.  Otherwise, images
    are loaded from the source directory and rotated to match intrinsics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dst = output_dir / "images"
    if images_dst.exists():
        if images_dst.is_symlink():
            images_dst.unlink()
        else:
            shutil.rmtree(images_dst)

    if images_source is not None and images_source.is_dir():
        logger.info("Copying pre-rotated images from %s ...", images_source)
        shutil.copytree(str(images_source), str(images_dst))
    else:
        images_dst.mkdir(parents=True, exist_ok=True)
        logger.info("Copying %d images to %s (rotating if needed) ...", len(frames), images_dst)
        meta_w = frames[0]["image_width"]
        meta_h = frames[0]["image_height"]
        for frame in tqdm(frames, desc="Copying images"):
            src = frame["image_path"]
            dst = images_dst / frame["image_file"]
            if not dst.exists():
                img = _load_image_rotated(src, meta_w, meta_h)
                img.save(str(dst), quality=95)

    sparse_dst = output_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for item in sparse_dir.iterdir():
        if item.is_file():
            dst_file = sparse_dst / item.name
            if dst_file.exists():
                dst_file.unlink()
            shutil.copy2(item, dst_file)

    logger.info("COLMAP dataset ready at %s", output_dir)
    logger.info("  images/ -> %d files", len(list(images_dst.iterdir())))
    logger.info("  sparse/0/ -> %d files", len(list(sparse_dst.iterdir())))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a SplatKing LiDAR-mode capture to a COLMAP dataset compatible with frgs reconstruct."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to a SplatKing session folder (e.g. PhotoSeries_YYYYMMDD_HHMMSS/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the frgs-compatible COLMAP dataset.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.5,
        help="Minimum qualityScore (0-1) to include a frame.",
    )
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help=(
            "Back-project per-frame LiDAR depth maps to produce a denser point cloud "
            "in addition to the session-level LiDAR point cloud."
        ),
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=4,
        help="Pixel stride for depth-map back-projection subsampling (requires --with-depth).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable cuSFM pose refinement using ARKit poses as priors (requires cusfm_cli on PATH).",
    )
    parser.add_argument(
        "--min-inter-frame-distance",
        type=float,
        default=0.06,
        help="cuSFM minimum translational distance (meters) between keyframes. Requires --refine.",
    )
    parser.add_argument(
        "--min-inter-frame-rotation",
        type=float,
        default=1.5,
        help="cuSFM minimum rotational change (degrees) between keyframes. Requires --refine.",
    )
    parser.add_argument(
        "--skip-cusfm",
        action="store_true",
        help="With --refine, stop after generating cuSFM metadata (do not run cuSFM).",
    )
    parser.add_argument(
        "--cusfm-extra-args",
        nargs="*",
        default=[],
        help="Additional arguments to pass to cusfm_cli. Requires --refine.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Parse session metadata
    logger.info("=== Phase 1: Parsing SplatKing session ===")
    session = parse_splatking_session(input_dir, args.quality_threshold)
    frames = session["frames"]

    if len(frames) == 0:
        logger.error("No frames passed filtering. Check --quality-threshold or input data.")
        sys.exit(1)

    logger.info("Image resolution: %dx%d", frames[0]["image_width"], frames[0]["image_height"])
    logger.info(
        "Intrinsics (avg): fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
        np.mean([f["intrinsics"]["fx"] for f in frames]),
        np.mean([f["intrinsics"]["fy"] for f in frames]),
        np.mean([f["intrinsics"]["cx"] for f in frames]),
        np.mean([f["intrinsics"]["cy"] for f in frames]),
    )

    # Load point cloud(s) for direct export or --with-depth replacement
    extract_depth = (not args.refine) or args.with_depth
    world_points = None
    lidar_path = session["lidar_pointcloud_path"]
    if lidar_path is not None and extract_depth:
        logger.info("Loading and colorizing LiDAR point cloud from %s ...", lidar_path)
        world_points = load_lidar_pointcloud(lidar_path, frames=frames)
        logger.info("LiDAR point cloud: %d points.", len(world_points))

    if args.with_depth and extract_depth:
        logger.info("=== Back-projecting per-frame depth maps (stride=%d) ===", args.depth_stride)
        depth_chunks: list[np.ndarray] = []
        for frame in tqdm(frames, desc="Back-projecting depth"):
            di = frame["depth_info"]
            if di is None:
                continue
            depth_path = input_dir / di["filename"]
            if not depth_path.exists():
                continue
            depth = load_depth_map(depth_path, di["width"], di["height"])
            pts = backproject_depth_to_world(
                depth,
                frame["intrinsics"],
                frame["c2w"],
                stride=args.depth_stride,
                image_path=frame["image_path"],
                image_width=frame["image_width"],
                image_height=frame["image_height"],
            )
            if len(pts) > 0:
                depth_chunks.append(pts)

        if depth_chunks:
            depth_points = np.vstack(depth_chunks).astype(np.float32)
            logger.info("Back-projected %d depth points from %d frames.", len(depth_points), len(depth_chunks))
            if world_points is not None:
                world_points = np.vstack([world_points, depth_points]).astype(np.float32)
            else:
                world_points = depth_points

    if world_points is not None:
        logger.info("Total point cloud: %d points.", len(world_points))

    if args.refine:
        # cuSFM pose-refinement pipeline
        logger.info("=== Phase 1b: Copying rotated images to work directory ===")
        _copy_images_to_work_dir(frames, work_dir)

        logger.info("=== Phase 2: Generating frames_meta.json ===")
        generate_frames_meta(frames, work_dir)

        if args.skip_cusfm:
            logger.info("--skip-cusfm set; stopping after metadata generation.")
            logger.info("cuSFM input directory: %s", work_dir)
            return

        logger.info("=== Phase 3: Running cuSFM ===")
        extra = list(args.cusfm_extra_args) if args.cusfm_extra_args else []
        sparse_dir = run_cusfm(
            work_dir,
            args.min_inter_frame_distance,
            args.min_inter_frame_rotation,
            extra_args=extra or None,
        )

        if args.with_depth and world_points is not None:
            logger.info("=== --with-depth: Replacing cuSFM points3D.bin with LiDAR depth points ===")
            n = _write_points3d_bin(sparse_dir / "points3D.bin", world_points)
            logger.info("Wrote %d LiDAR depth points to %s", n, sparse_dir / "points3D.bin")
    else:
        # Default: direct COLMAP export from ARKit poses
        logger.info("=== Phase 2: Writing COLMAP sparse files ===")
        sparse_dir = work_dir / "sparse_direct"
        write_colmap_sparse(frames, sparse_dir, world_points)

    # Final phase: Arrange output
    logger.info("=== Arranging COLMAP output ===")
    if args.refine:
        arrange_colmap_output(input_dir, frames, sparse_dir, output_dir, images_source=work_dir / "images")
    else:
        arrange_colmap_output(input_dir, frames, sparse_dir, output_dir)

    logger.info("Done. Run: frgs reconstruct %s -o <output.ply>", output_dir)


if __name__ == "__main__":
    main()
