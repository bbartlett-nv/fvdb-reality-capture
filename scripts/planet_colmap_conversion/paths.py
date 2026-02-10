# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# paths.py — Configurable data paths and sensor constants for Planet SuperDove
# L1A → COLMAP conversion pipeline.
#
from pathlib import Path

# ──────────────────────────── Data Directories ────────────────────────────── #

SOURCE_DIR = Path("/media/bbartlett/datadrive1/Data/planet/registration/satellite_raw_format/images")
DEM_DIR = Path("/media/bbartlett/datadrive1/Data/planet/registration/satellite_raw_format/dems")
DST_DIR = Path("/media/bbartlett/datadrive1/Data/planet/registration/colmap_format/")
ORTHO_TEMP_DIR = DST_DIR / "ortho_temp"  # Intermediate orthos

# ──────────────────────────── Processing Params ───────────────────────────── #

DOWNSAMPLE_STRIDE = 200  # Sample every Nth pixel from DEM grid for 3D points
TARGET_RPC_TYPE = "estimated_rpc"
OUTPUT_PIXEL_SIZE = 3.0
IMG_NODATA = 0
DEM_NODATA = -9999
TARGET_MAX = 255
TARGET_MIN = 0

# Scale factor applied to *all* exported coordinates (meters → km) so that
# 3DGS / COLMAP stay within reasonable float32 range.
SCALE_FACTOR = 1.0 / 1000.0  # Section 7.1 of the work plan

# ──────────────────────────── Sensor Constants ────────────────────────────── #
# Planet SuperDove PSB.SD sensor (Section 1.1 of the work plan)
# Camera: IMPERX 47MP (KAI-47052 CCD) + PSBlue telescope

IMG_WIDTH = 8880  # pixels (full butcher-block width)
IMG_HEIGHT = 5304  # pixels (full butcher-block height)
SAT_ID = "2534"

# Hardware specs for the IMPERX 47MP CCD
PIXEL_PITCH_UM = 5.5  # micrometres per pixel

# Camera center (initial, before per-satellite offset)
CAMERA_CENTER_X = IMG_WIDTH / 2.0  # 4440.0
CAMERA_CENTER_Y = IMG_HEIGHT / 2.0  # 2652.0

# ──────────────── Planet "psblue" Distortion Model ────────────────────────── #
# These are *pixel-domain* radial distortion coefficients and principal-point
# offsets shipped with the L1A product.  The "rem" key means the polynomial is
# meant to be *removed* (undistorted) — i.e.  the distortion is in the
# "add-to-distort" convention once negated.
# Section 3.1 of the work plan discusses rem vs. add semantics.

psblue_camera_models = {
    "2534": {
        "rem": {"k1": 9.425660962786705e-10, "k2": 3.5048503541604264e-18},
        "add": {"k1": -9.479595111585808e-10, "k2": -2.6704175296068837e-19},
        "center": {
            "x": CAMERA_CENTER_X + 53.478401,
            "y": CAMERA_CENTER_Y - 21.147406,
        },
    }
}

# Convenience shortcuts for the active satellite
SAT_DIST_COEFFS = [
    psblue_camera_models[SAT_ID]["add"]["k1"],
    psblue_camera_models[SAT_ID]["add"]["k2"],
    0.0,  # p1 (tangential — zero for SuperDove)
    0.0,  # p2
    0.0,  # k3
]
SAT_CX = psblue_camera_models[SAT_ID]["center"]["x"]
SAT_CY = psblue_camera_models[SAT_ID]["center"]["y"]
