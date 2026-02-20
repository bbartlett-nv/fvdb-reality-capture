# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# coord_transform.py — Coordinate system transformations for the
# Planet SuperDove L1A → COLMAP pipeline.
#
# Implements the full transformation chain described in Section 2 of the
# work plan:
#
#   P_C  =  T_{B→C}  ·  T_{E→B}  ·  T_{L→E}  ·  P_L
#
# where
#   L  =  Local Tangent Plane  (ENU, the COLMAP "world")
#   E  =  ECEF  (WGS-84)
#   B  =  Spacecraft Body frame
#   C  =  Camera / OpenCV frame
#
# References:
#   - Navipedia ECEF↔ENU (ref [12])
#   - Section 2.1–2.2 of the work plan
#
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# WGS-84 ellipsoid parameters
_WGS84_A = 6_378_137.0           # semi-major axis  (m)
_WGS84_B = 6_356_752.314245      # semi-minor axis  (m)
_WGS84_E2 = 1.0 - (_WGS84_B ** 2) / (_WGS84_A ** 2)  # first eccentricity²


# ═══════════════════ Geodetic ↔ ECEF ═══════════════════════════════════════ #

def geodetic_to_ecef(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
) -> np.ndarray:
    """
    Convert geodetic coordinates (WGS-84) to ECEF (metres).

    Parameters
    ----------
    lat_deg, lon_deg : float
        Latitude / longitude in decimal degrees.
    alt_m : float
        Altitude above ellipsoid in metres.

    Returns
    -------
    np.ndarray  shape (3,)
        [X, Y, Z] in ECEF metres.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)

    X = (N + alt_m) * cos_lat * cos_lon
    Y = (N + alt_m) * cos_lat * sin_lon
    Z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat

    return np.array([X, Y, Z], dtype=np.float64)


def ecef_to_geodetic(ecef: np.ndarray) -> tuple[float, float, float]:
    """
    Convert ECEF (metres) to geodetic lat/lon (degrees) + alt (metres).
    Uses the iterative Bowring method.
    """
    X, Y, Z = ecef
    lon = np.degrees(np.arctan2(Y, X))
    p = np.sqrt(X ** 2 + Y ** 2)
    lat = np.degrees(np.arctan2(Z, p * (1.0 - _WGS84_E2)))

    # Iterate for lat / alt
    for _ in range(10):
        sin_lat = np.sin(np.radians(lat))
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
        lat = np.degrees(np.arctan2(Z + _WGS84_E2 * N * sin_lat, p))

    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    alt = p / cos_lat - N if abs(cos_lat) > 1e-10 else abs(Z) / abs(sin_lat) - N * (1 - _WGS84_E2)

    return lat, lon, alt


# ═══════════════════ Batch Geodetic ↔ ECEF ════════════════════════════════ #


def batch_geodetic_to_ecef(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
) -> np.ndarray:
    """
    Vectorized geodetic → ECEF for N points.

    Parameters
    ----------
    lat_deg, lon_deg, alt_m : (N,) arrays

    Returns
    -------
    (N, 3) ECEF array
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    N_prime = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)

    X = (N_prime + alt_m) * cos_lat * cos_lon
    Y = (N_prime + alt_m) * cos_lat * sin_lon
    Z = (N_prime * (1.0 - _WGS84_E2) + alt_m) * sin_lat

    return np.column_stack([X, Y, Z])


def batch_ecef_to_geodetic(
    ecef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized ECEF → geodetic for N points using Bowring iteration.

    Parameters
    ----------
    ecef : (N, 3)

    Returns
    -------
    lat_deg, lon_deg, alt_m : each (N,)
    """
    X = ecef[:, 0]
    Y = ecef[:, 1]
    Z = ecef[:, 2]

    lon = np.degrees(np.arctan2(Y, X))
    p = np.sqrt(X ** 2 + Y ** 2)
    lat = np.degrees(np.arctan2(Z, p * (1.0 - _WGS84_E2)))

    for _ in range(10):
        sin_lat = np.sin(np.radians(lat))
        N_prime = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
        lat = np.degrees(np.arctan2(Z + _WGS84_E2 * N_prime * sin_lat, p))

    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    N_prime = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)

    # Altitude: use p/cos(lat) away from poles, Z/sin(lat) near poles
    alt = np.where(
        np.abs(cos_lat) > 1e-10,
        p / cos_lat - N_prime,
        np.abs(Z) / np.maximum(np.abs(sin_lat), 1e-30) - N_prime * (1 - _WGS84_E2),
    )

    return lat, lon, alt


# ═══════════════════ ENU ↔ ECEF ═══════════════════════════════════════════ #

def R_ecef_to_enu(lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    Rotation matrix from ECEF to ENU at a given geodetic reference.

    Section 2.2.1 of the work plan (transposed gives L→E).

    Returns
    -------
    np.ndarray  shape (3, 3)
        R_{E→L} such that  P_L = R · (P_E - O_E)
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    sp, cp = np.sin(phi), np.cos(phi)
    sl, cl = np.sin(lam), np.cos(lam)

    # Rows: East, North, Up
    return np.array([
        [-sl,       cl,       0.0],
        [-sp * cl, -sp * sl,  cp],
        [ cp * cl,  cp * sl,  sp],
    ], dtype=np.float64)


def ecef_to_enu(
    point_ecef: np.ndarray,
    origin_ecef: np.ndarray,
    R_e2l: np.ndarray,
) -> np.ndarray:
    """Transform a point (or N×3 array) from ECEF to ENU."""
    diff = point_ecef - origin_ecef
    if diff.ndim == 1:
        return R_e2l @ diff
    return (R_e2l @ diff.T).T


def enu_to_ecef(
    point_enu: np.ndarray,
    origin_ecef: np.ndarray,
    R_e2l: np.ndarray,
) -> np.ndarray:
    """Transform a point (or N×3 array) from ENU back to ECEF."""
    R_l2e = R_e2l.T   # Section 2.2.1:  R_{L→E} = R_{E→L}^T
    if point_enu.ndim == 1:
        return R_l2e @ point_enu + origin_ecef
    return (R_l2e @ point_enu.T).T + origin_ecef


# ═══════════════════ Body-to-Camera Static Transform ══════════════════════ #

def R_body_to_camera() -> np.ndarray:
    """
    Static rotation from the satellite Body frame to the OpenCV Camera frame.

    Section 2.2.3 of the work plan:
        Body:   +Z = boresight (nadir), +X = velocity, +Y = cross-track
        Camera: +Z = optical axis (forward), +X = right, +Y = down

    The typical boresight alignment is:
        Camera_X =  Body_Y     (right ≡ cross-track)
        Camera_Y = -Body_X     (down ≡ opposite velocity in nadir)
        Camera_Z =  Body_Z     (optical axis = boresight)

    This is equivalent to a 90° rotation about Z in the body frame:

        R_{B→C} = [[0,  1,  0],
                    [-1, 0,  0],
                    [0,  0,  1]]
    """
    return np.array([
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ], dtype=np.float64)


# ═══════════════════ Full Pose Computation ════════════════════════════════ #

@dataclass
class LTPReference:
    """Local Tangent Plane (ENU) reference origin."""

    lat_deg: float
    lon_deg: float
    alt_m: float
    origin_ecef: np.ndarray    # (3,)   ECEF of the origin
    R_e2l: np.ndarray          # (3,3)  Rotation ECEF → ENU

    @classmethod
    def from_geodetic(cls, lat: float, lon: float, alt: float) -> "LTPReference":
        origin = geodetic_to_ecef(lat, lon, alt)
        R = R_ecef_to_enu(lat, lon)
        return cls(lat, lon, alt, origin, R)

    @classmethod
    def from_ecef(cls, ecef: np.ndarray) -> "LTPReference":
        lat, lon, alt = ecef_to_geodetic(ecef)
        R = R_ecef_to_enu(lat, lon)
        return cls(lat, lon, alt, ecef, R)


def R_eci_to_ecef(epoch_unix: float) -> np.ndarray:
    """
    Compute the 3×3 rotation matrix from ECI (GCRS) to ECEF (ITRS)
    at the given UNIX epoch using Astropy's IAU frame transformations.

    This accounts for Earth rotation (GAST), precession, and nutation.

    Parameters
    ----------
    epoch_unix : float
        UNIX timestamp (seconds since 1970-01-01 UTC).

    Returns
    -------
    np.ndarray  shape (3, 3)
        R such that  P_ecef = R · P_eci
    """
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
    from astropy.time import Time
    import astropy.units as u

    t = Time(epoch_unix, format="unix", scale="utc")

    # Transform three ECI unit vectors to ECEF to build the matrix.
    # Each column of R is an ECI basis vector expressed in ECEF.
    eci_axes = CartesianRepresentation(
        x=[1, 0, 0] * u.m,
        y=[0, 1, 0] * u.m,
        z=[0, 0, 1] * u.m,
    )
    gcrs = GCRS(eci_axes, obstime=t)
    itrs = gcrs.transform_to(ITRS(obstime=t))

    R = np.column_stack([
        itrs.cartesian.xyz[:, 0].value,
        itrs.cartesian.xyz[:, 1].value,
        itrs.cartesian.xyz[:, 2].value,
    ])
    return R  # R_eci2ecef: maps ECI vectors to ECEF


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """
    Convert [qw, qx, qy, qz] to a 3×3 rotation matrix using scipy.

    Planet metadata uses the standard Hamilton / engineering convention
    with real part first.  ``scipy.spatial.transform.Rotation.from_quat``
    expects [qx, qy, qz, qw] so we re-order.
    """
    # scipy wants [x, y, z, w]
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_matrix()


def compute_world_to_camera(
    q_sat: np.ndarray,
    C_sat_ecef: np.ndarray,
    ltp: LTPReference,
    epoch_unix: float | None = None,
    C_sat_ecef_next: np.ndarray | None = None,
    off_nadir_deg: float | None = None,
    azimuth_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the COLMAP world-to-camera rotation ``R`` and translation ``t``
    for one frame using the attitude quaternion.

    The Planet ``from_taaser`` quaternion describes the rotation from ECEF
    (or ECI, if ``epoch_unix`` is provided) to the spacecraft **bus** frame,
    whose Z-axis is aligned with the orbit normal — NOT with nadir.

    To obtain the camera frame (OpenCV: +Z = optical axis toward ground),
    this function builds a **dynamic** body-to-camera rotation by projecting
    the boresight and velocity vectors into the bus frame and constructing
    orthonormal camera axes from them.

    Parameters
    ----------
    q_sat : (4,) quaternion [qw, qx, qy, qz]
    C_sat_ecef : (3,) satellite ECEF position
    ltp : LTPReference
    epoch_unix : float or None
        If provided, the quaternion is treated as ECI→Body and the
        Astropy GCRS→ITRS rotation at this epoch is applied to convert
        it to ECEF→Body.
    C_sat_ecef_next : (3,) or None
        Next frame's ECEF position for velocity estimation.  If ``None``
        an arbitrary perpendicular direction is used.
    off_nadir_deg : float or None
        Off-nadir angle in degrees.  If provided together with
        ``azimuth_deg``, the boresight is tilted away from nadir by
        this amount in the azimuth direction.
    azimuth_deg : float or None
        Satellite azimuth in degrees from north (clockwise).

    Returns
    -------
    R_wc : np.ndarray  (3, 3)
        World-to-camera rotation.
    t_wc : np.ndarray  (3,)
        World-to-camera translation.
    """
    # 1. R from quaternion
    R_q = quaternion_to_rotation(q_sat)

    # 2. If epoch is provided, convert ECI→Body to ECEF→Body
    if epoch_unix is not None:
        R_eci2ecef = R_eci_to_ecef(epoch_unix)
        R_ecef2eci = R_eci2ecef.T
        R_e2b = R_q @ R_ecef2eci  # ECEF→ECI→Body
    else:
        R_e2b = R_q  # Assume already ECEF→Body

    # 3. Build a dynamic body-to-camera rotation.
    #    Start with nadir direction, then tilt if off-nadir metadata is available.
    nadir_ecef = -C_sat_ecef / np.linalg.norm(C_sat_ecef)
    boresight_ecef = nadir_ecef.copy()

    # Apply off-nadir tilt using Rodrigues rotation
    if (
        off_nadir_deg is not None
        and off_nadir_deg > 0
        and azimuth_deg is not None
    ):
        # The metadata ``satellite_azimuth_mean`` is the ground-track
        # azimuth (direction of travel).  The off-nadir tilt direction
        # is approximately perpendicular to the track — empirically
        # offset by −134° from the ground-track azimuth.
        tilt_az = azimuth_deg - 134.0
        az_rad = np.radians(tilt_az)
        off_rad = np.radians(off_nadir_deg)

        # Tilt direction in ENU: azimuth measured clockwise from north
        tilt_enu = np.array([np.sin(az_rad), np.cos(az_rad), 0.0])

        # Convert tilt direction to ECEF
        R_l2e = ltp.R_e2l.T
        tilt_ecef = R_l2e @ tilt_enu
        tilt_ecef = tilt_ecef / np.linalg.norm(tilt_ecef)

        # Rodrigues rotation: rotate nadir toward tilt direction by off_nadir
        axis = np.cross(boresight_ecef, tilt_ecef)
        axis = axis / np.linalg.norm(axis)
        boresight_ecef = (
            boresight_ecef * np.cos(off_rad)
            + np.cross(axis, boresight_ecef) * np.sin(off_rad)
        )

    boresight_body = R_e2b @ boresight_ecef

    if C_sat_ecef_next is not None:
        vel_ecef = C_sat_ecef_next - C_sat_ecef
    else:
        vel_ecef = np.cross(np.array([0.0, 0.0, 1.0]), nadir_ecef)
    vel_ecef = vel_ecef / np.linalg.norm(vel_ecef)
    vel_body = R_e2b @ vel_ecef

    # Camera Z = boresight (optical axis toward ground, tilted if off-nadir)
    cam_z = boresight_body / np.linalg.norm(boresight_body)
    # Camera X = cross-track (right in image)
    # Negate to match the sensor's pixel readout direction (column order).
    cam_x = np.cross(vel_body, cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)
    # Camera Y = completes the right-handed frame (down in image)
    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y / np.linalg.norm(cam_y)

    # R_{B→C}: rows are camera axes expressed in body coordinates
    R_b2c = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    # 4. R_{L→E} = R_{E→L}^T
    R_l2e = ltp.R_e2l.T

    # Full rotation  World (LTP) → Camera
    R_wc = R_b2c @ R_e2b @ R_l2e

    # Camera centre in LTP
    C_sat_ltp = ecef_to_enu(C_sat_ecef, ltp.origin_ecef, ltp.R_e2l)

    # COLMAP translation:  t = -R · C
    t_wc = -R_wc @ C_sat_ltp

    return R_wc, t_wc


def compute_nadir_world_to_camera(
    C_sat_ecef: np.ndarray,
    ltp: LTPReference,
    C_sat_ecef_next: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a nadir-pointing world-to-camera pose directly from the
    satellite ECEF position, bypassing the attitude quaternion.

    This is a robust fallback when the quaternion convention is unknown
    or suspected to be incorrect (e.g. ECI→Body instead of ECEF→Body).

    The camera is constructed so that:
        - Optical axis (cam +Z) points from satellite toward Earth centre (nadir)
        - cam +X is approximately along-track (velocity direction)
        - cam +Y completes the right-handed frame (cross-track, pointing "down"
          in OpenCV convention)

    If ``C_sat_ecef_next`` is provided (the next frame's ECEF position), the
    velocity vector is estimated from the position difference.  Otherwise
    an arbitrary perpendicular direction is used.

    Parameters
    ----------
    C_sat_ecef : (3,) satellite ECEF position
    ltp : LTPReference
    C_sat_ecef_next : (3,) optional next-frame ECEF position for velocity est.

    Returns
    -------
    R_wc, t_wc : world-to-camera rotation and translation in ENU/LTP frame.
    """
    # --- Build an ECEF camera frame ---
    # Boresight points from satellite to Earth centre (nadir)
    nadir_ecef = -C_sat_ecef / np.linalg.norm(C_sat_ecef)

    # Velocity direction (along-track)
    if C_sat_ecef_next is not None:
        vel_ecef = C_sat_ecef_next - C_sat_ecef
    else:
        # Use a vector roughly perpendicular to nadir
        # (cross with world-Z gives a decent east-ish direction)
        vel_ecef = np.cross(np.array([0, 0, 1.0]), nadir_ecef)
    vel_ecef = vel_ecef / np.linalg.norm(vel_ecef)

    # Gram–Schmidt to orthogonalise velocity w.r.t. nadir
    vel_ecef = vel_ecef - np.dot(vel_ecef, nadir_ecef) * nadir_ecef
    vel_ecef = vel_ecef / np.linalg.norm(vel_ecef)

    # Cross-track (completes right-handed frame)
    cross_ecef = np.cross(nadir_ecef, vel_ecef)
    cross_ecef = cross_ecef / np.linalg.norm(cross_ecef)

    # Camera frame in ECEF (OpenCV: +X right, +Y down, +Z forward)
    #   cam_Z = nadir   (optical axis toward ground)
    #   cam_X = cross   (right / cross-track)
    #   cam_Y = -vel    (down — opposite of velocity for a south-going pass)
    # Build R_{ECEF→Cam} where each ROW is the camera axis in ECEF
    R_ecef2cam = np.array([
        cross_ecef,
        -vel_ecef,
        nadir_ecef,
    ], dtype=np.float64)

    # --- Convert to ENU (LTP) world frame ---
    R_l2e = ltp.R_e2l.T  # ENU → ECEF
    R_wc = R_ecef2cam @ R_l2e  # ENU → ECEF → Camera

    # Camera centre in ENU
    C_sat_ltp = ecef_to_enu(C_sat_ecef, ltp.origin_ecef, ltp.R_e2l)
    t_wc = -R_wc @ C_sat_ltp

    return R_wc, t_wc


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3×3 rotation matrix to [qw, qx, qy, qz] quaternion.

    COLMAP uses [qw, qx, qy, qz] ordering.
    """
    r = Rotation.from_matrix(R)
    q_xyzw = r.as_quat()   # scipy: [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
