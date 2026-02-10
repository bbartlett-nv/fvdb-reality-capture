# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# parse_metadata.py — Extract satellite ephemeris, attitude, and intrinsics
# from Planet SuperDove L1A JSON metadata.
#
# References:
#   - Planet Basic L1A All-Frames User Guide (ref [8] in work plan)
#   - Section 1.1, 2.2.2, 3 of the work plan
#
"""
Phase 1 of the pipeline:  metadata parsing and trajectory extraction.

For each L1A TIFF file shipped with a sidecar ``.tif.json``, this module
extracts:

* **Satellite position** ``C_sat`` in ECEF (metres), converted from the
  geodetic lat/lng/alt stored in ``extended.sat``.
* **Attitude quaternion** ``q_sat`` describing the rotation ECEF → Boresight,
  read from ``extended.satellite_attitude.from_taaser``.
* **Timestamps** for each frame.
* **Focal length** in pixels — derived from the GSD reported in the metadata
  and the known hardware pixel pitch (5.5 µm for the IMPERX 47MP CCD).
* **RPC coefficients** for downstream validation (``rpc_check.py``).

The principal-point offset and distortion coefficients are static per
satellite and therefore live in ``paths.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from coord_transform import geodetic_to_ecef
from paths import IMG_HEIGHT, IMG_WIDTH, PIXEL_PITCH_UM, SAT_CX, SAT_CY, SAT_DIST_COEFFS

logger = logging.getLogger(__name__)

# Pixel pitch in metres (for focal-length derivation)
_PIXEL_PITCH_M = PIXEL_PITCH_UM * 1e-6


# ──────────────────────────── Data Classes ─────────────────────────────────── #


@dataclass
class FramePose:
    """Pose of a single L1A frame."""

    image_path: Path
    image_id: int
    timestamp: str  # ISO-8601

    # ECEF satellite position (metres)
    C_sat: np.ndarray  # (3,)

    # Quaternion ECEF → Body  [qw, qx, qy, qz]
    q_sat: np.ndarray  # (4,)

    # Sensor intrinsics --------------------------------------------------
    focal_length_px: float  # focal length in pixels
    img_width: int = IMG_WIDTH
    img_height: int = IMG_HEIGHT
    cx: float = SAT_CX  # principal point x (pixels)
    cy: float = SAT_CY  # principal point y (pixels)

    # Distortion in *pixel domain* (OpenCV order: k1, k2, p1, p2, k3)
    dist_coeffs_px: list[float] = field(default_factory=lambda: list(SAT_DIST_COEFFS))

    # RPC coefficients (filled if available)
    rpc: dict[str, Any] | None = None

    # Satellite heading angle (degrees, from metadata)
    satellite_heading: float | None = None

    # Precise acquisition epoch (UNIX float, seconds since 1970-01-01 UTC).
    # Parsed from the filename timestamp (YYYYMMDD_HHMMSS_CC_SATID).
    acquisition_epoch: float | None = None

    def intrinsic_matrix(self) -> np.ndarray:
        """Return the 3×3 camera intrinsic matrix K (pixel units)."""
        f = self.focal_length_px
        return np.array(
            [
                [f, 0.0, self.cx],
                [0.0, f, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def normalized_dist_coeffs(self) -> list[float]:
        """
        Return distortion coefficients scaled from pixel domain to the
        *normalised image coordinate* domain expected by OpenCV / COLMAP.

        Section 3.2 of the work plan:
            k_norm = k_px · f^(2n)

        For k1 (n=1):  k1_norm = k1_px · f²
        For k2 (n=2):  k2_norm = k2_px · f⁴
        """
        f = self.focal_length_px
        k1_px, k2_px, p1, p2, k3 = self.dist_coeffs_px
        return [
            k1_px * f**2,
            k2_px * f**4,
            p1,
            p2,
            k3,
        ]


# ──────────────────────── Metadata Parsing ─────────────────────────────────── #


def _get_extended(meta: dict) -> dict:
    """
    Return the ``extended`` block from the metadata JSON.

    Planet L1A sidecar files store all per-frame data under a top-level
    ``extended`` key.
    """
    ext = meta.get("extended")
    if ext is None:
        # Fallback: the root *is* the extended block (or a flat structure)
        return meta
    return ext


def _extract_position(ext: dict) -> np.ndarray:
    """
    Extract satellite position in ECEF (metres).

    The metadata stores geodetic coordinates under ``extended.sat``:
        lat (degrees), lng (degrees), alt (km).
    We convert to ECEF using the WGS-84 ellipsoid.
    """
    sat = ext["sat"]
    lat_deg = float(sat["lat"])
    lon_deg = float(sat["lng"])
    alt_km = float(sat["alt"])
    alt_m = alt_km * 1000.0
    return geodetic_to_ecef(lat_deg, lon_deg, alt_m)


def _extract_quaternion(ext: dict) -> np.ndarray:
    """
    Extract the attitude quaternion describing ECEF → Boresight rotation.

    The metadata stores the quaternion under
    ``extended.satellite_attitude.from_taaser`` with keys::

        quat s  (scalar / real part)
        quat x
        quat y
        quat z

    We return [qw, qx, qy, qz] (real-first convention).
    """
    att = ext["satellite_attitude"]["from_taaser"]
    qs = float(att["quat s"])
    qx = float(att["quat x"])
    qy = float(att["quat y"])
    qz = float(att["quat z"])
    return np.array([qs, qx, qy, qz], dtype=np.float64)


def _extract_focal_length(ext: dict) -> float:
    """
    Derive the focal length in **pixels** from the metadata.

    The Planet L1A metadata does not contain the physical focal length
    directly.  Instead we derive it from the reported GSD and the known
    hardware pixel pitch:

        f_m   = altitude_m × pixel_pitch_m / GSD_m
        f_px  = f_m / pixel_pitch_m  =  altitude_m / GSD_m

    The GSD (metres) is found at ``extended.GRD.gsd`` or
    ``extended.footprint.gsd``, and the altitude (km) at
    ``extended.sat.alt``.

    The pixel pitch is the IMPERX 47MP hardware constant (5.5 µm) defined
    in ``paths.py``.
    """
    # --- Try explicit focal length keys first (future-proofing) ----------
    for key in ("camera_focal_length_mm", "focal_length_mm"):
        f_mm = ext.get(key)
        if f_mm is not None:
            return float(f_mm) / (_PIXEL_PITCH_M * 1e3)

    f_px = ext.get("camera_focal_length_px") or ext.get("focal_length_px")
    if f_px is not None:
        return float(f_px)

    # --- Derive from GSD and altitude ------------------------------------
    gsd: float | None = None
    alt_km: float | None = None

    # GSD sources
    grd_block = ext.get("GRD") or ext.get("grd_block")
    if grd_block and "gsd" in grd_block:
        gsd = float(grd_block["gsd"])
    elif "footprint" in ext and "gsd" in ext["footprint"]:
        gsd = float(ext["footprint"]["gsd"])
    elif "ground" in ext and "scale" in ext["ground"]:
        gsd = float(ext["ground"]["scale"])

    # Altitude source
    sat = ext.get("sat")
    if sat and "alt" in sat:
        alt_km = float(sat["alt"])

    if gsd is not None and alt_km is not None and gsd > 0:
        alt_m = alt_km * 1000.0
        # f_px = altitude_m / GSD_m
        focal_px = alt_m / gsd
        logger.info(
            "Derived focal length from GSD=%.3f m, alt=%.1f km → f=%.1f px",
            gsd,
            alt_km,
            focal_px,
        )
        return focal_px

    raise ValueError(
        "Cannot determine focal length — no focal length key found in "
        "metadata and could not derive from GSD/altitude."
    )


def _extract_rpcs(ext: dict, rpc_type: str = "estimated_rpc") -> dict[str, Any] | None:
    """
    Return the Rational Polynomial Coefficients block if present.

    We prefer ``estimated_rpc`` but fall back to ``rpc`` or ``vendor_rpc``.
    """
    for key in (rpc_type, "rpc", "vendor_rpc"):
        if key in ext:
            return ext[key]
    return None


def _parse_acquisition_epoch(image_path: Path) -> float | None:
    """
    Parse the acquisition time from the Planet L1A TIFF filename and return
    it as a UNIX epoch float (seconds since 1970-01-01 UTC).

    Filename format:  ``YYYYMMDD_HHMMSS_CC_SATID.tif``
    where CC is centiseconds (0–99).

    Example:  ``20251012_192217_04_2534.tif`` → 2025-10-12 19:22:17.04 UTC
    """
    stem = image_path.stem  # e.g. "20251012_192217_04_2534"
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
        centisec = int(parts[2])
        dt = dt.replace(microsecond=centisec * 10000, tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, IndexError):
        return None


def _extract_timestamp(ext: dict) -> str:
    """
    Best-effort timestamp extraction.

    Tries several locations in the metadata.
    """
    # camera section often has the closest-to-capture timestamp
    cam = ext.get("camera", {})
    ts = cam.get("timestamp")
    if ts:
        return str(ts)

    # strip start_time is a UNIX epoch float
    strip = ext.get("strip", {})
    start = strip.get("start_time")
    if start is not None:
        return str(start)

    # Generic fallback
    sat = ext.get("sat", {})
    ts = sat.get("timestamp")
    if ts:
        return str(ts)

    return ""


def parse_l1a_metadata(
    metadata_path: Path,
    image_path: Path,
    image_id: int,
    rpc_type: str = "estimated_rpc",
) -> FramePose:
    """
    Parse a single L1A ``.tif.json`` sidecar and return a ``FramePose``.

    Parameters
    ----------
    metadata_path : Path
        Path to the sidecar JSON file.
    image_path : Path
        Corresponding L1A TIFF file (stored for later reference).
    image_id : int
        Unique COLMAP image ID (1-based).
    rpc_type : str
        Which RPC block to use (``estimated_rpc`` | ``rpc`` | ``vendor_rpc``).
    """
    with open(metadata_path, "r") as fh:
        meta = json.load(fh)

    ext = _get_extended(meta)

    focal_length_px = _extract_focal_length(ext)
    C_sat = _extract_position(ext)
    q_sat = _extract_quaternion(ext)
    rpc = _extract_rpcs(ext, rpc_type)
    timestamp = _extract_timestamp(ext)
    acquisition_epoch = _parse_acquisition_epoch(image_path)

    # Satellite heading (if available)
    sat_block = ext.get("sat", {})
    heading = sat_block.get("satellite_azimuth_mean")

    logger.info(
        "Parsed %s  —  pos=%.0f,%.0f,%.0f  f=%.1f px  epoch=%.3f",
        metadata_path.name,
        *C_sat,
        focal_length_px,
        acquisition_epoch or 0.0,
    )

    return FramePose(
        image_path=image_path,
        image_id=image_id,
        timestamp=timestamp,
        C_sat=C_sat,
        q_sat=q_sat,
        focal_length_px=focal_length_px,
        rpc=rpc,
        satellite_heading=heading,
        acquisition_epoch=acquisition_epoch,
    )


def parse_all_frames(
    source_dir: Path,
    rpc_type: str = "estimated_rpc",
) -> list[FramePose]:
    """
    Walk ``source_dir`` for sidecar JSON metadata files and match each to its
    TIFF.

    Planet L1A sidecar convention: the JSON file appends ``.json`` to the
    full TIFF filename, e.g.::

        20251012_192221_29_2534.tif      ← image
        20251012_192221_29_2534.tif.json ← metadata sidecar
    """
    # Look for *.tif.json sidecar files
    meta_files = sorted(source_dir.glob("*.tif.json"))
    if not meta_files:
        raise FileNotFoundError(f"No *.tif.json sidecar files found in {source_dir}")

    poses: list[FramePose] = []
    for idx, mf in enumerate(meta_files, start=1):
        # Derive the TIFF path by stripping the trailing ".json"
        # e.g. 20251012_192221_29_2534.tif.json → 20251012_192221_29_2534.tif
        image_path = mf.with_suffix("")  # removes the last .json suffix
        if not image_path.exists():
            logger.warning("No TIFF found for %s — skipping.", mf.name)
            continue
        poses.append(parse_l1a_metadata(mf, image_path, image_id=idx, rpc_type=rpc_type))

    # Sort by acquisition epoch (precise) or string timestamp (fallback)
    poses.sort(key=lambda p: p.acquisition_epoch if p.acquisition_epoch is not None else 0.0)

    # Re-assign sequential IDs after sorting
    for i, p in enumerate(poses, start=1):
        p.image_id = i

    logger.info("Parsed %d frames from %s", len(poses), source_dir)
    return poses
