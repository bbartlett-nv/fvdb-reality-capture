# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""Canonical binary-mask IO and cache-fingerprint helpers.

All mask consumers operate on contiguous two-dimensional boolean arrays. Image
formats are encoded as 0/255 byte rasters and NumPy masks are encoded as bool.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from fvdb_reality_capture.sfm_scene import SfmScene

MaskFormat = Literal["png", "jpg", "npy"]

_MASK_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_MASK_SUFFIXES = _MASK_IMAGE_SUFFIXES | {".npy"}
_FINGERPRINT_SCHEMA_VERSION = 1


def mask_path_is_declared(mask_path: str | pathlib.Path) -> bool:
    """Return whether metadata declares a mask path."""
    return bool(str(mask_path).strip())


def canonicalize_binary_mask(mask: np.ndarray, *, source: str = "mask") -> np.ndarray:
    """Convert supported mask encodings to a contiguous two-dimensional bool array.

    Boolean masks are preserved. Numeric 0-1 masks use normalized semantics
    (floating point values are valid above 0.5; integer values are valid above
    zero), while byte-range masks are valid above 127. Multi-channel arrays are
    accepted only when every channel is identical, avoiding silent channel
    selection for ambiguous masks.
    """
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        elif array.shape[2] in (3, 4):
            first_channel = array[..., 0]
            if not np.array_equal(array, np.repeat(first_channel[..., None], array.shape[2], axis=2)):
                raise ValueError(
                    f"Ambiguous multi-channel mask from {source}: channels differ for shape {array.shape}. "
                    "Provide a single-channel mask or identical channels."
                )
            array = first_channel
        else:
            raise ValueError(
                f"Mask from {source} must have shape (H, W), (H, W, 1), or identical RGB/RGBA channels; "
                f"got {array.shape}"
            )
    if array.ndim != 2:
        raise ValueError(f"Mask from {source} must have shape (H, W); got {array.shape}")

    if array.dtype == np.bool_:
        return np.ascontiguousarray(array)
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise TypeError(f"Mask from {source} must have a boolean or numeric dtype; got {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"Mask from {source} contains non-finite values")
    if array.size == 0:
        return np.ascontiguousarray(array, dtype=np.bool_)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(
            f"Mask from {source} has values outside supported 0-1 or 0-255 ranges: min={minimum}, max={maximum}"
        )
    if maximum <= 1.0:
        threshold = 0.5 if np.issubdtype(array.dtype, np.floating) else 0
    else:
        threshold = 127
    return np.ascontiguousarray(array > threshold)


def load_binary_mask(mask_path: str | pathlib.Path) -> np.ndarray:
    """Load a declared PNG/JPEG/NPY mask as a canonical 2D boolean array."""
    if not mask_path_is_declared(mask_path):
        raise ValueError("Mask path is empty; no mask was declared")

    path = pathlib.Path(str(mask_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Declared mask file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in _MASK_SUFFIXES:
        raise ValueError(
            f"Unsupported mask file format {suffix!r} for {path}; supported formats are PNG, JPEG, and NPY"
        )

    if suffix == ".npy":
        try:
            decoded = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Failed to decode NPY mask {path}: {error}") from error
    else:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Failed to decode mask image: {path}")
    return canonicalize_binary_mask(decoded, source=str(path))


def encode_binary_mask(mask: np.ndarray, mask_format: MaskFormat) -> np.ndarray:
    """Encode a mask for cache persistence using one canonical representation."""
    canonical = canonicalize_binary_mask(mask)
    if mask_format in ("png", "jpg"):
        return canonical.astype(np.uint8) * np.uint8(255)
    if mask_format == "npy":
        return canonical
    raise ValueError(f"Unsupported mask format {mask_format!r}; expected 'png', 'jpg', or 'npy'")


def mask_bbox_xyxy_count(mask: np.ndarray) -> np.ndarray:
    """Return tight half-open ``[xmin, ymin, xmax, ymax, valid_count]`` bounds."""
    canonical = canonicalize_binary_mask(mask)
    valid_count = int(np.count_nonzero(canonical))
    if valid_count == 0:
        return np.zeros((5,), dtype=np.int64)
    valid_rows = np.flatnonzero(np.any(canonical, axis=1))
    valid_columns = np.flatnonzero(np.any(canonical, axis=0))
    return np.array(
        [valid_columns[0], valid_rows[0], valid_columns[-1] + 1, valid_rows[-1] + 1, valid_count],
        dtype=np.int64,
    )


def bbox_metadata_contains_mask(metadata_bbox: np.ndarray, mask_bbox: np.ndarray) -> bool:
    """Return whether cached bbox/count metadata contains the decoded mask exactly in count."""
    metadata = np.asarray(metadata_bbox, dtype=np.int64)
    decoded = np.asarray(mask_bbox, dtype=np.int64)
    if metadata.shape != (5,) or decoded.shape != (5,):
        return False
    if int(metadata[4]) != int(decoded[4]):
        return False
    if int(decoded[4]) == 0:
        return np.array_equal(metadata, np.zeros((5,), dtype=np.int64))
    return bool(
        metadata[0] <= decoded[0]
        and metadata[1] <= decoded[1]
        and metadata[2] >= decoded[2]
        and metadata[3] >= decoded[3]
    )


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_identity(source_path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(str(source_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Source image file does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def mask_source_fingerprint(mask_path: str | pathlib.Path) -> dict[str, Any] | None:
    """Fingerprint both the identity and raw/canonical content of a declared mask."""
    if not mask_path_is_declared(mask_path):
        return None
    path = pathlib.Path(str(mask_path)).expanduser()
    canonical = load_binary_mask(path)
    canonical_digest = hashlib.sha256()
    canonical_digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    canonical_digest.update(canonical.tobytes(order="C"))
    return {
        **_source_file_identity(path),
        "raw_sha256": _sha256_file(path),
        "canonical_sha256": canonical_digest.hexdigest(),
        "shape": [int(value) for value in canonical.shape],
        "valid_count": int(np.count_nonzero(canonical)),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_scene_cache_fingerprint(
    scene: "SfmScene",
    *,
    algorithm: str,
    algorithm_version: int,
    settings: dict[str, Any],
    include_source_images: bool,
) -> dict[str, Any]:
    """Build a versioned fingerprint over ordered scene/camera/mask inputs."""
    ordered_images: list[dict[str, Any]] = []
    for image in scene.images:
        camera = image.camera_metadata
        image_record: dict[str, Any] = {
            "image_id": int(image.image_id),
            "camera_id": int(image.camera_id),
            "dimensions_hw": [int(camera.height), int(camera.width)],
            "world_to_camera": np.asarray(image.world_to_camera_matrix, dtype=np.float64).tolist(),
            "camera_to_world": np.asarray(image.camera_to_world_matrix, dtype=np.float64).tolist(),
            "projection_matrix": np.asarray(camera.projection_matrix, dtype=np.float64).tolist(),
            "camera_model": camera.camera_model.name,
            "distortion_coeffs": np.asarray(camera.distortion_coeffs, dtype=np.float64).tolist(),
            "source_mask": mask_source_fingerprint(image.mask_path),
        }
        if include_source_images:
            image_record["source_image"] = _source_file_identity(image.image_path)
        ordered_images.append(image_record)

    payload = _json_ready(
        {
            "schema_version": _FINGERPRINT_SCHEMA_VERSION,
            "algorithm": algorithm,
            "algorithm_version": int(algorithm_version),
            "settings": settings,
            "transformation_matrix": np.asarray(scene.transformation_matrix, dtype=np.float64),
            "ordered_images": ordered_images,
        }
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "schema_version": _FINGERPRINT_SCHEMA_VERSION,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "payload": payload,
    }


def cache_fingerprint_matches(cached: Any, expected: dict[str, Any]) -> bool:
    """Compare cache fingerprints without trusting legacy or malformed entries."""
    return bool(
        isinstance(cached, dict)
        and cached.get("schema_version") == _FINGERPRINT_SCHEMA_VERSION
        and cached.get("sha256") == expected.get("sha256")
    )


__all__ = [
    "MaskFormat",
    "bbox_metadata_contains_mask",
    "build_scene_cache_fingerprint",
    "cache_fingerprint_matches",
    "canonicalize_binary_mask",
    "encode_binary_mask",
    "load_binary_mask",
    "mask_bbox_xyxy_count",
    "mask_path_is_declared",
    "mask_source_fingerprint",
]
