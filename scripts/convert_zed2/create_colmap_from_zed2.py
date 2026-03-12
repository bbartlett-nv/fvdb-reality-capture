#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""
Convert a ZED2 .svo recording into a COLMAP dataset compatible with ``frgs reconstruct``.

By default the script exports ZED tracking poses directly (no refinement).
Pass ``--refine`` to enable CUDA-accelerated pose refinement via cuSFM.

Pipeline:
  Phase 1 -- Extract stereo frames, intrinsics, and ZED tracking poses from the .svo file.
  Phase 2 -- (--refine only) Generate ``frames_meta.json`` for cuSFM.
  Phase 3 -- (--refine only) Run cuSFM for pose refinement (stereo-aware, with ZED pose priors).
  Phase 4 -- Arrange the sparse output into the COLMAP layout expected by frgs.

Requires:
  - ZED SDK with Python API (pyzed)
  - scipy, numpy, tqdm
  - pyCuSFM (cusfm_cli on PATH) -- only when using --refine
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
from scipy.spatial.transform import Rotation
from tqdm import tqdm

logger = logging.getLogger(__name__)

LEFT_CAMERA_NAME = "zed2_left"
RIGHT_CAMERA_NAME = "zed2_right"
LEFT_CAMERA_PARAMS_ID = "0"
RIGHT_CAMERA_PARAMS_ID = "1"

# 90-degree rotation around the X-axis: converts ZED's Y-up world frame to the
# Z-up convention expected by COLMAP and the fVDB viewer.
# Maps: X -> X,  Y -> Z,  Z -> -Y
R_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


# ---------------------------------------------------------------------------
# Phase 1: SVO extraction
# ---------------------------------------------------------------------------


def _quaternion_to_axis_angle_dict(quat_xyzw: np.ndarray) -> dict:
    """Convert a quaternion (x, y, z, w) to cuSFM axis-angle dict."""
    rot = Rotation.from_quat(quat_xyzw)  # scipy uses (x, y, z, w)
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


def _enhance_clahe(sl_mat, clahe, path: str) -> None:
    """Apply CLAHE to the L channel of a ZED ``sl.Mat`` and write to *path*."""
    import cv2

    bgra = sl_mat.get_data()
    bgr = bgra[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(path, enhanced)


def _laplacian_variance(sl_mat) -> float:
    """Return the variance of the Laplacian (higher = sharper)."""
    import cv2

    bgra = sl_mat.get_data()
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _save_blur_histogram(values: list[float], threshold: float | None, path) -> None:
    """Save a histogram of per-frame sharpness values to *path*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=40, edgecolor="black", alpha=0.7)
    if threshold and threshold > 0:
        ax.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.1f}")
        ax.legend()
    ax.set_xlabel("Laplacian Variance (sharpness)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Blur Metric Distribution")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


def extract_from_svo(
    svo_path: Path,
    work_dir: Path,
    frame_stride: int,
    image_format: str,
    extract_depth: bool = False,
    depth_stride: int = 8,
    low_light: bool = False,
    blur_threshold: float | None = None,
) -> dict:
    """
    Open a .svo file, enable positional tracking, and extract stereo frames,
    calibration, and per-frame poses.

    When *extract_depth* is True, also retrieve per-frame XYZRGBA point clouds
    from the ZED depth engine, subsample them, and transform to world
    coordinates.  The result is stored under ``"world_points"`` as an (N, 6)
    float32 array (x, y, z, r, g, b) -- or ``None`` when depth is not
    extracted.
    """
    try:
        import pyzed.sl as sl
    except ImportError:
        logger.error("pyzed (ZED SDK Python API) is not installed. Install the ZED SDK first.")
        sys.exit(1)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    if low_light:
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        logger.info("Low-light mode: using NEURAL depth with relaxed confidence thresholds + CLAHE.")

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        logger.error("Failed to open SVO file: %s (error: %s)", svo_path, status)
        sys.exit(1)

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3
    if low_light:
        tracking_params.enable_pose_smoothing = True
    zed.enable_positional_tracking(tracking_params)

    cam_info = zed.get_camera_information()
    cal = cam_info.camera_configuration.calibration_parameters
    left_cam = cal.left_cam
    right_cam = cal.right_cam

    left_intrinsics = {
        "fx": float(left_cam.fx),
        "fy": float(left_cam.fy),
        "cx": float(left_cam.cx),
        "cy": float(left_cam.cy),
    }
    right_intrinsics = {
        "fx": float(right_cam.fx),
        "fy": float(right_cam.fy),
        "cx": float(right_cam.cx),
        "cy": float(right_cam.cy),
    }
    image_width = int(left_cam.image_size.width)
    image_height = int(left_cam.image_size.height)
    baseline = float(cal.get_camera_baseline())

    left_dir = work_dir / LEFT_CAMERA_NAME
    right_dir = work_dir / RIGHT_CAMERA_NAME
    for d in (left_dir, right_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    left_image = sl.Mat()
    right_image = sl.Mat()
    pose = sl.Pose()
    pc_mat = sl.Mat() if extract_depth else None

    runtime = sl.RuntimeParameters()
    if low_light:
        runtime.confidence_threshold = 100
        runtime.texture_confidence_threshold = 100
        runtime.enable_fill_mode = True

    clahe = None
    if low_light:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    svo_length = zed.get_svo_number_of_frames()
    frames = []
    world_points_chunks: list[np.ndarray] = []
    frame_idx = 0
    skipped_tracking = 0
    skipped_blur = 0
    sharpness_values: list[float] = []
    ext = f".{image_format}"

    logger.info("Extracting frames from SVO (%d total, stride=%d)...", svo_length, frame_stride)

    with tqdm(total=svo_length, desc="Extracting SVO frames") as pbar:
        while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            if frame_idx % frame_stride == 0:
                state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                if state != sl.POSITIONAL_TRACKING_STATE.OK:
                    skipped_tracking += 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                if blur_threshold is not None:
                    sharpness = _laplacian_variance(left_image)
                    sharpness_values.append(sharpness)
                    if blur_threshold > 0 and sharpness < blur_threshold:
                        skipped_blur += 1
                        frame_idx += 1
                        pbar.update(1)
                        continue

                ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

                img_name = f"{ts_ns}{ext}"
                if clahe is not None:
                    _enhance_clahe(left_image, clahe, str(left_dir / img_name))
                    _enhance_clahe(right_image, clahe, str(right_dir / img_name))
                else:
                    left_image.write(str(left_dir / img_name))
                    right_image.write(str(right_dir / img_name))

                orientation = pose.get_orientation()
                quat_xyzw = np.array(
                    [orientation.get()[0], orientation.get()[1], orientation.get()[2], orientation.get()[3]]
                )
                translation = pose.get_translation().get()

                frames.append(
                    {
                        "timestamp_ns": int(ts_ns),
                        "timestamp_us": int(ts_ns // 1000),
                        "image_name_left": f"{LEFT_CAMERA_NAME}/{img_name}",
                        "image_name_right": f"{RIGHT_CAMERA_NAME}/{img_name}",
                        "quat_xyzw": quat_xyzw.tolist(),
                        "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
                        "tracking_state": str(state),
                    }
                )

                if extract_depth:
                    zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
                    xyzrgba = pc_mat.get_data()  # (H, W, 4) float32
                    s = depth_stride
                    sub = xyzrgba[::s, ::s].reshape(-1, 4)

                    xyz = sub[:, :3]
                    valid = np.isfinite(xyz).all(axis=1)
                    xyz = xyz[valid]
                    color_packed = sub[valid, 3]

                    if len(xyz) > 0:
                        color_bytes = color_packed.view(np.uint8).reshape(-1, 4)
                        r = color_bytes[:, 0].astype(np.float32)
                        g = color_bytes[:, 1].astype(np.float32)
                        b = color_bytes[:, 2].astype(np.float32)

                        rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
                        t_cw = np.array([float(translation[0]), float(translation[1]), float(translation[2])])
                        xyz_world = (rot_mat @ xyz.T).T + t_cw
                        xyz_world = (R_Y_UP_TO_Z_UP @ xyz_world.T).T
                        chunk = np.column_stack([xyz_world, r, g, b])
                        world_points_chunks.append(chunk)

            frame_idx += 1
            pbar.update(1)

    zed.disable_positional_tracking()
    zed.close()

    if skipped_tracking > 0:
        logger.warning("Skipped %d frames with unreliable tracking (state != OK).", skipped_tracking)
    if skipped_blur > 0:
        logger.warning("Skipped %d blurry frames (Laplacian variance < %.1f).", skipped_blur, blur_threshold)
    if sharpness_values:
        hist_path = work_dir / "blur_histogram.png"
        _save_blur_histogram(sharpness_values, blur_threshold, hist_path)
        logger.info("Saved blur histogram to %s", hist_path)
    logger.info("Extracted %d frame pairs from %d SVO frames.", len(frames), svo_length)

    world_points = None
    if world_points_chunks:
        world_points = np.vstack(world_points_chunks).astype(np.float32)
        logger.info("Accumulated %d depth points from %d frames.", len(world_points), len(world_points_chunks))

    return {
        "frames": frames,
        "left_intrinsics": left_intrinsics,
        "right_intrinsics": right_intrinsics,
        "baseline": baseline,
        "image_width": image_width,
        "image_height": image_height,
        "world_points": world_points,
    }


# ---------------------------------------------------------------------------
# Phase 2: Generate frames_meta.json
# ---------------------------------------------------------------------------


def _build_camera_params(
    sensor_id: int,
    sensor_name: str,
    intrinsics: dict,
    image_width: int,
    image_height: int,
    sensor_to_vehicle: dict,
    frequency: int = 30,
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


def generate_frames_meta(extraction: dict, work_dir: Path) -> Path:
    """
    Write a ``frames_meta.json`` file in cuSFM KeyframesMetadataCollection
    format using extracted SVO data.
    """
    frames = extraction["frames"]
    baseline = extraction["baseline"]

    identity_transform = {
        "axis_angle": {"x": 0.0, "y": 0.0, "z": 0.0, "angle_degrees": 0.0},
        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    right_transform = {
        "axis_angle": {"x": 0.0, "y": 0.0, "z": 0.0, "angle_degrees": 0.0},
        "translation": {"x": baseline, "y": 0.0, "z": 0.0},
    }

    left_params = _build_camera_params(
        sensor_id=0,
        sensor_name=LEFT_CAMERA_NAME,
        intrinsics=extraction["left_intrinsics"],
        image_width=extraction["image_width"],
        image_height=extraction["image_height"],
        sensor_to_vehicle=identity_transform,
    )
    right_params = _build_camera_params(
        sensor_id=1,
        sensor_name=RIGHT_CAMERA_NAME,
        intrinsics=extraction["right_intrinsics"],
        image_width=extraction["image_width"],
        image_height=extraction["image_height"],
        sensor_to_vehicle=right_transform,
    )

    keyframes = []
    for idx, frame in enumerate(frames):
        quat_xyzw = np.array(frame["quat_xyzw"])
        t_cw_vec = np.array(frame["translation"])

        rot_cw = Rotation.from_quat(quat_xyzw)
        rot_cw_zup = Rotation.from_matrix(R_Y_UP_TO_Z_UP @ rot_cw.as_matrix())
        t_cw_zup = R_Y_UP_TO_Z_UP @ t_cw_vec

        axis_angle = _quaternion_to_axis_angle_dict(rot_cw_zup.as_quat())
        translation = {"x": float(t_cw_zup[0]), "y": float(t_cw_zup[1]), "z": float(t_cw_zup[2])}
        camera_to_world = {"axis_angle": axis_angle, "translation": translation}
        synced_id = str(idx)

        keyframes.append(
            {
                "id": str(idx * 2),
                "camera_params_id": LEFT_CAMERA_PARAMS_ID,
                "timestamp_microseconds": str(frame["timestamp_us"]),
                "image_name": frame["image_name_left"],
                "camera_to_world": camera_to_world,
                "synced_sample_id": synced_id,
            }
        )

        keyframes.append(
            {
                "id": str(idx * 2 + 1),
                "camera_params_id": RIGHT_CAMERA_PARAMS_ID,
                "timestamp_microseconds": str(frame["timestamp_us"]),
                "image_name": frame["image_name_right"],
                "camera_to_world": camera_to_world,
                "synced_sample_id": synced_id,
            }
        )

    meta = {
        "keyframes_metadata": keyframes,
        "initial_pose_type": "EGO_MOTION",
        "camera_params_id_to_session_name": {LEFT_CAMERA_PARAMS_ID: "0", RIGHT_CAMERA_PARAMS_ID: "0"},
        "camera_params_id_to_camera_params": {LEFT_CAMERA_PARAMS_ID: left_params, RIGHT_CAMERA_PARAMS_ID: right_params},
        "stereo_pair": [
            {
                "left_camera_param_id": LEFT_CAMERA_PARAMS_ID,
                "right_camera_param_id": RIGHT_CAMERA_PARAMS_ID,
                "baseline_meters": baseline,
            }
        ],
    }

    out_path = work_dir / "frames_meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s (%d keyframe entries).", out_path, len(keyframes))
    return out_path


# ---------------------------------------------------------------------------
# Phase 3: cuSFM reconstruction
# ---------------------------------------------------------------------------


def run_cusfm(
    work_dir: Path,
    min_inter_frame_distance: float,
    min_inter_frame_rotation: float,
    extra_args: list[str] | None = None,
) -> Path:
    """Invoke ``cusfm_cli`` and return the path to the sparse output directory."""
    cusfm_base = work_dir / "cusfm"
    cmd = [
        "cusfm_cli",
        "--input_dir",
        str(work_dir),
        "--cusfm_base_dir",
        str(cusfm_base),
        "--export_binary_colmap_files",
        "--output_rgb",
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


# ---------------------------------------------------------------------------
# Direct COLMAP export (no cuSFM)
# ---------------------------------------------------------------------------

COLMAP_PINHOLE_MODEL_ID = 1


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
                f.write(struct.pack("<Q", 0))     # track_length
            return len(world_points)
        else:
            f.write(struct.pack("<Q", 0))
            return 0


def write_colmap_sparse_from_extraction(extraction: dict, sparse_dir: Path) -> Path:
    """
    Write COLMAP binary sparse files directly from ZED extraction data,
    bypassing cuSFM refinement.  Returns *sparse_dir*.

    Produces cameras.bin, images.bin, and points3D.bin (populated from ZED
    depth when ``extraction["world_points"]`` is available).
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)
    frames = extraction["frames"]
    baseline = extraction["baseline"]
    w = extraction["image_width"]
    h = extraction["image_height"]

    # --- cameras.bin ---
    left = extraction["left_intrinsics"]
    right = extraction["right_intrinsics"]
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))
        for cam_id, intr in enumerate([left, right]):
            f.write(struct.pack("<I", cam_id))
            f.write(struct.pack("<i", COLMAP_PINHOLE_MODEL_ID))
            f.write(struct.pack("<Q", w))
            f.write(struct.pack("<Q", h))
            f.write(struct.pack("<dddd", intr["fx"], intr["fy"], intr["cx"], intr["cy"]))

    # --- images.bin ---
    num_images = len(frames) * 2
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", num_images))
        for idx, frame in enumerate(frames):
            quat_xyzw = np.array(frame["quat_xyzw"])
            t_cw = np.array(frame["translation"])

            # Y-up -> Z-up world rotation on the camera-to-world transform
            rot_cw = Rotation.from_quat(quat_xyzw)
            rot_cw_zup = Rotation.from_matrix(R_Y_UP_TO_Z_UP @ rot_cw.as_matrix())
            t_cw_zup = R_Y_UP_TO_Z_UP @ t_cw

            # Camera-to-world -> world-to-camera (in Z-up world)
            rot_wc = rot_cw_zup.inv()
            t_wc = -rot_wc.as_matrix() @ t_cw_zup

            # OpenGL -> OpenCV camera frame convention
            R_FLIP = np.diag([1.0, -1.0, -1.0])
            rot_wc_colmap = Rotation.from_matrix(R_FLIP @ rot_wc.as_matrix())
            t_wc = R_FLIP @ t_wc

            quat_wc_xyzw = rot_wc_colmap.as_quat()  # scipy: (x, y, z, w)
            qw = quat_wc_xyzw[3]
            qx = quat_wc_xyzw[0]
            qy = quat_wc_xyzw[1]
            qz = quat_wc_xyzw[2]

            # Left image
            image_id_left = idx * 2 + 1
            f.write(struct.pack("<I", image_id_left))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", t_wc[0], t_wc[1], t_wc[2]))
            f.write(struct.pack("<I", 0))
            f.write(frame["image_name_left"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))

            # Right image – same rotation, translation offset by -baseline in camera x
            t_wc_right = t_wc.copy()
            t_wc_right[0] -= baseline
            image_id_right = idx * 2 + 2
            f.write(struct.pack("<I", image_id_right))
            f.write(struct.pack("<dddd", qw, qx, qy, qz))
            f.write(struct.pack("<ddd", t_wc_right[0], t_wc_right[1], t_wc_right[2]))
            f.write(struct.pack("<I", 1))
            f.write(frame["image_name_right"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))

    # --- points3D.bin ---
    num_points = _write_points3d_bin(
        sparse_dir / "points3D.bin", extraction.get("world_points"),
    )
    logger.info(
        "Wrote COLMAP sparse files to %s (%d cameras, %d images, %d points).",
        sparse_dir, 2, num_images, num_points,
    )
    return sparse_dir


# ---------------------------------------------------------------------------
# Phase 4: Arrange COLMAP layout for frgs
# ---------------------------------------------------------------------------


def arrange_colmap_output(work_dir: Path, sparse_dir: Path, output_dir: Path) -> None:
    """
    Create the directory layout expected by ``frgs reconstruct``:
      output_dir/images/  (symlink to work_dir containing zed2_left/, zed2_right/)
      output_dir/sparse/0/  (cameras.bin, images.bin, points3D.bin)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dst = output_dir / "images"
    if images_dst.exists():
        if images_dst.is_symlink():
            images_dst.unlink()
        else:
            shutil.rmtree(images_dst)

    images_dst.mkdir(parents=True, exist_ok=True)

    for cam_name in [LEFT_CAMERA_NAME, RIGHT_CAMERA_NAME]:
        src = work_dir / cam_name
        dst = images_dst / cam_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    sparse_dst = output_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for item in sparse_dir.iterdir():
        if item.is_file():
            dst_file = sparse_dst / item.name
            if dst_file.exists():
                dst_file.unlink()
            shutil.copy2(item, dst_file)

    logger.info("COLMAP dataset ready at %s", output_dir)
    logger.info("  images/ -> %s", images_dst)
    logger.info("  sparse/0/ -> %d files", len(list(sparse_dst.iterdir())))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_extraction_metadata(extraction: dict, work_dir: Path) -> None:
    """Write extraction metadata for traceability (not consumed by cuSFM)."""
    meta = {
        "num_frames": len(extraction["frames"]),
        "left_intrinsics": extraction["left_intrinsics"],
        "right_intrinsics": extraction["right_intrinsics"],
        "baseline_m": extraction["baseline"],
        "image_width": extraction["image_width"],
        "image_height": extraction["image_height"],
        "timestamps_ns": [f["timestamp_ns"] for f in extraction["frames"]],
    }
    path = work_dir / "extraction_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote extraction metadata to %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a ZED2 .svo recording to a COLMAP dataset. Pass --refine to enable cuSFM pose refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--svo-path", type=Path, required=True, help="Path to the input .svo file.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for the frgs-compatible COLMAP dataset."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for intermediate files. Defaults to <output-dir>/_work.",
    )
    parser.add_argument("--frame-stride", type=int, default=5, help="Extract every Nth frame from the SVO.")
    parser.add_argument(
        "--image-format", choices=["jpeg", "png"], default="jpeg", help="Image format for extracted frames."
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable cuSFM pose refinement instead of exporting ZED tracking poses directly.",
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
        "--with-depth",
        action="store_true",
        help="Include ZED depth points in the output. Always on by default without --refine; with --refine, replaces cuSFM feature-matched points.",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=8,
        help="Pixel stride for depth point-cloud subsampling. Higher = fewer points.",
    )
    parser.add_argument(
        "--low-light",
        action="store_true",
        help=(
            "Optimize for under-exposed / low-light captures: use NEURAL depth mode, "
            "relax depth-confidence thresholds, and apply CLAHE image enhancement. "
            "Requires opencv-python."
        ),
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=None,
        help=(
            "Skip frames with Laplacian variance below this value (lower = blurrier). "
            "Enabled automatically at 50.0 with --low-light. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--tight-poses",
        action="store_true",
        help="Use tighter pose constraints for cuSFM (recommended for indoor ZED2 captures). Requires --refine.",
    )
    parser.add_argument(
        "--cusfm-extra-args", nargs="*", default=[], help="Additional arguments to pass to cusfm_cli. Requires --refine."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    svo_path = args.svo_path.resolve()
    if not svo_path.exists():
        logger.error("SVO file not found: %s", svo_path)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    work_dir = args.work_dir.resolve() if args.work_dir else output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    blur_threshold = args.blur_threshold
    if blur_threshold is None and args.low_light:
        blur_threshold = 50.0

    # Phase 1
    logger.info("=== Phase 1: Extracting from SVO ===")
    extraction = extract_from_svo(
        svo_path, work_dir, args.frame_stride, args.image_format,
        extract_depth=(not args.refine) or args.with_depth, depth_stride=args.depth_stride,
        low_light=args.low_light, blur_threshold=blur_threshold,
    )
    _write_extraction_metadata(extraction, work_dir)

    if len(extraction["frames"]) == 0:
        logger.error("No frames extracted from SVO. Check the file and frame-stride setting.")
        sys.exit(1)

    if args.refine:
        # cuSFM pose-refinement pipeline
        logger.info("=== Phase 2: Generating frames_meta.json ===")
        generate_frames_meta(extraction, work_dir)

        if args.skip_cusfm:
            logger.info("--skip-cusfm set; stopping after metadata generation.")
            logger.info("cuSFM input directory: %s", work_dir)
            return

        logger.info("=== Phase 3: Running cuSFM ===")
        extra = list(args.cusfm_extra_args) if args.cusfm_extra_args else []
        if args.tight_poses:
            config_dir = str(Path(__file__).resolve().parent / "configs" / "zed2_indoor")
            extra = ["--config_dir", config_dir] + extra
            logger.info("Using tight-pose config: %s", config_dir)
        sparse_dir = run_cusfm(
            work_dir,
            args.min_inter_frame_distance,
            args.min_inter_frame_rotation,
            extra_args=extra or None,
        )

        if args.with_depth and extraction.get("world_points") is not None:
            logger.info("=== --with-depth: Replacing cuSFM points3D.bin with ZED depth points ===")
            n = _write_points3d_bin(sparse_dir / "points3D.bin", extraction["world_points"])
            logger.info("Wrote %d ZED depth points to %s", n, sparse_dir / "points3D.bin")
    else:
        # Default: direct COLMAP export from ZED tracking poses
        logger.info("=== Writing COLMAP sparse from ZED poses (no refinement) ===")
        sparse_dir = work_dir / "sparse_direct"
        write_colmap_sparse_from_extraction(extraction, sparse_dir)

    # Phase 4
    logger.info("=== Phase 4: Arranging COLMAP output ===")
    arrange_colmap_output(work_dir, sparse_dir, output_dir)

    logger.info("Done. Run: frgs reconstruct %s -o <output.ply>", output_dir)


if __name__ == "__main__":
    main()
