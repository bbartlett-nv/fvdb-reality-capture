# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""Bounded-memory access to trackless binary COLMAP point models.

This module intentionally implements a narrow fast path. It supports binary COLMAP
models whose ``points3D.bin`` records all have zero-length tracks. Cameras and
registered images are loaded through PyCOLMAP from a temporary metadata-only model,
so the original point model is never materialized by PyCOLMAP.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import tempfile
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .adapter import COLMAPAdapter
from .sfm_cache import SfmCache
from .sfm_metadata import SfmCameraMetadata, SfmPosedImageMetadata

BoundsMode = Literal["open", "closed", "half_open"]

_POINTS_HEADER_SIZE = 8
_TRACKLESS_POINT_RECORD_SIZE = 51
_FINGERPRINT_SAMPLE_SIZE = 64 * 1024
_CONTENT_HASH_BLOCK_SIZE = 8 * 1024 * 1024
_POINT_RECORD_DTYPE = np.dtype(
    {
        "names": ("point_id", "xyz", "rgb", "error", "track_length"),
        "formats": ("<u8", ("<f8", (3,)), ("u1", (3,)), "<f8", "<u8"),
        "offsets": (0, 8, 32, 35, 43),
        "itemsize": _TRACKLESS_POINT_RECORD_SIZE,
    }
)


class UnsupportedColmapPartialLoadError(RuntimeError):
    """Raised when a COLMAP model cannot use the bounded-memory fast path."""


@dataclass(frozen=True)
class ColmapPointSelection:
    """Materialized point columns returned by a bounded spatial query."""

    point_ids: np.ndarray
    points: np.ndarray
    points_rgb: np.ndarray
    points_err: np.ndarray

    def __len__(self) -> int:
        return int(self.points.shape[0])


def _resolve_sparse_path(colmap_path: pathlib.Path) -> pathlib.Path:
    sparse_path = colmap_path / "sparse" / "0"
    if not sparse_path.exists():
        sparse_path = colmap_path / "sparse"
    if not sparse_path.exists():
        raise UnsupportedColmapPartialLoadError(f"COLMAP sparse model directory does not exist: {sparse_path}")
    return sparse_path


def probe_trackless_colmap_binary(colmap_path: str | pathlib.Path) -> tuple[bool, str]:
    """Return whether ``colmap_path`` is supported by the partial-load fast path."""

    colmap_path = pathlib.Path(colmap_path)
    try:
        sparse_path = _resolve_sparse_path(colmap_path)
    except UnsupportedColmapPartialLoadError as error:
        return False, str(error)

    required_files = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required_files if not (sparse_path / name).is_file()]
    if missing:
        return (
            False,
            "The bounded-memory COLMAP fast path requires a binary sparse model; "
            f"missing {', '.join(missing)} in {sparse_path}",
        )

    points_path = sparse_path / "points3D.bin"
    file_size = points_path.stat().st_size
    if file_size < _POINTS_HEADER_SIZE:
        return False, f"COLMAP point file is truncated: {points_path}"
    with points_path.open("rb") as file:
        point_count_bytes = file.read(_POINTS_HEADER_SIZE)
    if len(point_count_bytes) != _POINTS_HEADER_SIZE:
        return False, f"Could not read the COLMAP point count from {points_path}"
    point_count = struct.unpack("<Q", point_count_bytes)[0]
    expected_size = _POINTS_HEADER_SIZE + point_count * _TRACKLESS_POINT_RECORD_SIZE
    if file_size != expected_size:
        return (
            False,
            "The bounded-memory COLMAP fast path currently requires every points3D.bin record to have an empty "
            f"visibility track. Expected {expected_size} bytes for {point_count:,} trackless points, found "
            f"{file_size} bytes in {points_path}",
        )
    return True, "binary COLMAP model with trackless points3D.bin"


class ColmapBinaryPointSource:
    """Memory-mapped, bounded-memory point queries over a trackless ``points3D.bin``."""

    def __init__(self, colmap_path: str | pathlib.Path, block_size: int = 1_000_000):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self._colmap_path = pathlib.Path(colmap_path).absolute()
        supported, reason = probe_trackless_colmap_binary(self._colmap_path)
        if not supported:
            raise UnsupportedColmapPartialLoadError(reason)

        self._sparse_path = _resolve_sparse_path(self._colmap_path)
        self._points_path = self._sparse_path / "points3D.bin"
        self._block_size = int(block_size)
        self._initial_stat = self._points_path.stat()
        with self._points_path.open("rb") as file:
            self._point_count = int(struct.unpack("<Q", file.read(_POINTS_HEADER_SIZE))[0])

        self._records: np.memmap | None
        if self._point_count == 0:
            self._records = None
        else:
            self._records = np.memmap(
                self._points_path,
                mode="r",
                dtype=_POINT_RECORD_DTYPE,
                offset=_POINTS_HEADER_SIZE,
                shape=(self._point_count,),
            )
        self._bounds: np.ndarray | None = None
        self._fingerprint = self._compute_fingerprint()
        self._full_fingerprint: str | None = None

    @property
    def colmap_path(self) -> pathlib.Path:
        return self._colmap_path

    @property
    def sparse_path(self) -> pathlib.Path:
        return self._sparse_path

    @property
    def full_fingerprint(self) -> str:
        """Return a lazy full-file SHA-256 for strict cross-process resume validation."""

        self._assert_unchanged()
        if self._full_fingerprint is None:
            digest = hashlib.sha256()
            with self._points_path.open("rb") as point_file:
                for block in iter(lambda: point_file.read(_CONTENT_HASH_BLOCK_SIZE), b""):
                    digest.update(block)
            self._assert_unchanged()
            self._full_fingerprint = digest.hexdigest()
        return self._full_fingerprint

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def fingerprint(self) -> str:
        """A bounded-cost fingerprint of the point source and its filesystem identity."""

        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        stat = self._points_path.stat()
        digest = hashlib.sha256()
        identity = {
            "path": str(self._points_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "point_count": self._point_count,
        }
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        with self._points_path.open("rb") as file:
            digest.update(file.read(_FINGERPRINT_SAMPLE_SIZE))
            if stat.st_size > _FINGERPRINT_SAMPLE_SIZE:
                file.seek(max(0, stat.st_size - _FINGERPRINT_SAMPLE_SIZE))
                digest.update(file.read(_FINGERPRINT_SAMPLE_SIZE))
        return digest.hexdigest()

    def _assert_unchanged(self) -> None:
        stat = self._points_path.stat()
        if stat.st_size != self._initial_stat.st_size or stat.st_mtime_ns != self._initial_stat.st_mtime_ns:
            raise RuntimeError(f"COLMAP point source changed after it was opened: {self._points_path}")

    @staticmethod
    def _validate_bbox(bbox: np.ndarray | None) -> np.ndarray | None:
        if bbox is None:
            return None
        bbox = np.asarray(bbox, dtype=np.float64)
        if bbox.shape != (6,):
            raise ValueError(f"bbox must have shape (6,), got {bbox.shape}")
        if np.any(np.isnan(bbox)):
            raise ValueError("bbox must not contain NaN values")
        if np.any(bbox[:3] >= bbox[3:]):
            raise ValueError(f"bbox minima must be strictly less than maxima, got {bbox}")
        return bbox

    @staticmethod
    def _validate_bounds_mode(bounds_mode: BoundsMode) -> None:
        if bounds_mode not in ("open", "closed", "half_open"):
            raise ValueError(f"Unknown bounds_mode {bounds_mode!r}; expected 'open', 'closed', or 'half_open'")

    @staticmethod
    def _bbox_mask(xyz: np.ndarray, bbox: np.ndarray | None, bounds_mode: BoundsMode) -> np.ndarray:
        if bbox is None:
            return np.ones(xyz.shape[0], dtype=np.bool_)

        mask = np.ones(xyz.shape[0], dtype=np.bool_)
        for axis in range(3):
            if bounds_mode == "open":
                mask &= xyz[:, axis] > bbox[axis]
                mask &= xyz[:, axis] < bbox[axis + 3]
            elif bounds_mode == "closed":
                mask &= xyz[:, axis] >= bbox[axis]
                mask &= xyz[:, axis] <= bbox[axis + 3]
            else:
                mask &= xyz[:, axis] >= bbox[axis]
                mask &= xyz[:, axis] < bbox[axis + 3]
        return mask

    def _iter_record_blocks(self):
        if self._records is None:
            return
        for start in range(0, self._point_count, self._block_size):
            yield self._records[start : min(start + self._block_size, self._point_count)]

    def bounds(self) -> np.ndarray | None:
        """Return ``[xmin, ymin, zmin, xmax, ymax, zmax]`` or ``None`` for an empty model."""

        self._assert_unchanged()
        if self._point_count == 0:
            return None
        if self._bounds is None:
            lower = np.full(3, np.inf, dtype=np.float64)
            upper = np.full(3, -np.inf, dtype=np.float64)
            for records in self._iter_record_blocks():
                xyz = records["xyz"]
                lower = np.minimum(lower, np.min(xyz, axis=0))
                upper = np.maximum(upper, np.max(xyz, axis=0))
            self._bounds = np.concatenate((lower, upper))
        return self._bounds.copy()

    def count_points(self, bbox: np.ndarray | None = None, bounds_mode: BoundsMode = "open") -> int:
        """Count selected points without materializing their columns."""

        self._assert_unchanged()
        bbox = self._validate_bbox(bbox)
        self._validate_bounds_mode(bounds_mode)
        if bbox is None:
            return self._point_count

        count = 0
        for records in self._iter_record_blocks():
            count += int(np.count_nonzero(self._bbox_mask(records["xyz"], bbox, bounds_mode)))
        return count

    def count_points_and_check_strict_id_order(
        self, bbox: np.ndarray | None = None, bounds_mode: BoundsMode = "open"
    ) -> tuple[int, bool]:
        """Count selected points and check file-order IDs in the same bounded scan."""

        self._assert_unchanged()
        bbox = self._validate_bbox(bbox)
        self._validate_bounds_mode(bounds_mode)

        count = 0
        previous_id: int | None = None
        strictly_increasing = True
        for records in self._iter_record_blocks():
            point_ids = records["point_id"]
            if len(point_ids) > 0:
                if previous_id is not None and int(point_ids[0]) <= previous_id:
                    strictly_increasing = False
                if len(point_ids) > 1 and np.any(point_ids[1:] <= point_ids[:-1]):
                    strictly_increasing = False
                previous_id = int(point_ids[-1])
            count += int(np.count_nonzero(self._bbox_mask(records["xyz"], bbox, bounds_mode)))
        return count, strictly_increasing

    def query_points(
        self,
        bbox: np.ndarray | None = None,
        bounds_mode: BoundsMode = "open",
        expected_selected_count: int | None = None,
    ) -> ColmapPointSelection:
        """Materialize only the point columns selected by ``bbox``.

        ``open`` matches :class:`CropScene`'s current strict ``min < x < max``
        semantics. ``half_open`` is useful for assigning non-overlapping chunk cores.
        Supplying ``expected_selected_count`` skips the counting pass and validates
        that materialization produces exactly the caller's precomputed count.
        """

        self._assert_unchanged()
        bbox = self._validate_bbox(bbox)
        self._validate_bounds_mode(bounds_mode)
        if expected_selected_count is None:
            selected_count = self.count_points(bbox, bounds_mode)
        else:
            if isinstance(expected_selected_count, bool) or not isinstance(expected_selected_count, (int, np.integer)):
                raise TypeError(
                    f"expected_selected_count must be a non-negative integer or None, got {expected_selected_count!r}"
                )
            selected_count = int(expected_selected_count)
            if selected_count < 0:
                raise ValueError(f"expected_selected_count must be non-negative, got {selected_count}")

        point_ids = np.empty(selected_count, dtype=np.uint64)
        points = np.empty((selected_count, 3), dtype=np.float32)
        points_rgb = np.empty((selected_count, 3), dtype=np.uint8)
        points_err = np.empty(selected_count, dtype=np.float32)

        output_start = 0
        for records in self._iter_record_blocks():
            mask = self._bbox_mask(records["xyz"], bbox, bounds_mode)
            block_count = int(np.count_nonzero(mask))
            if block_count == 0:
                continue
            output_stop = output_start + block_count
            if output_stop > selected_count:
                raise ValueError(
                    f"Point query selected more than expected_selected_count={selected_count:,}; "
                    f"encountered at least {output_stop:,} matching points"
                )
            point_ids[output_start:output_stop] = records["point_id"][mask]
            points[output_start:output_stop] = records["xyz"][mask]
            points_rgb[output_start:output_stop] = records["rgb"][mask]
            points_err[output_start:output_stop] = records["error"][mask]
            output_start = output_stop

        if output_start != selected_count:
            raise ValueError(
                f"Point query count did not match expected_selected_count={selected_count:,}; "
                f"selected {output_start:,} points"
            )
        return ColmapPointSelection(point_ids, points, points_rgb, points_err)


def _load_metadata_without_points(
    colmap_path: pathlib.Path, sparse_path: pathlib.Path
) -> tuple[dict[int, SfmCameraMetadata], list[SfmPosedImageMetadata], SfmCache]:
    """Load cameras/images via PyCOLMAP while substituting an empty point model."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = pathlib.Path(temporary_directory)
        temporary_sparse_path = temporary_root / "sparse" / "0"
        temporary_sparse_path.mkdir(parents=True)
        os.symlink(sparse_path / "cameras.bin", temporary_sparse_path / "cameras.bin")
        os.symlink(sparse_path / "images.bin", temporary_sparse_path / "images.bin")
        (temporary_sparse_path / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        adapter = COLMAPAdapter(temporary_root)

        image_records = []
        loaded_cameras: dict[int, SfmCameraMetadata] = {}
        for colmap_image_id in adapter.registered_image_ids():
            camera_id = adapter.image_camera_id(colmap_image_id)
            if camera_id not in loaded_cameras:
                loaded_cameras[camera_id] = adapter.camera_metadata(camera_id)
            image_records.append(
                (
                    adapter.image_name(colmap_image_id),
                    camera_id,
                    adapter.world_to_camera_matrix(colmap_image_id),
                )
            )

    image_records.sort(key=lambda record: record[0])
    images_path = colmap_path / "images"
    masks_path = colmap_path / "masks"
    loaded_images = []
    for image_id, (image_name, camera_id, world_to_camera) in enumerate(image_records):
        mask_path = ""
        if masks_path.exists():
            candidate = masks_path / image_name
            if candidate.exists():
                mask_path = str(candidate.absolute())
            elif candidate.with_suffix(".png").exists():
                mask_path = str(candidate.with_suffix(".png").absolute())
        loaded_images.append(
            SfmPosedImageMetadata(
                world_to_camera_matrix=world_to_camera.copy(),
                camera_to_world_matrix=np.linalg.inv(world_to_camera).copy(),
                camera_id=camera_id,
                camera_metadata=loaded_cameras[camera_id],
                image_path=str((images_path / image_name).absolute()),
                mask_path=mask_path,
                point_indices=np.empty((0,), dtype=np.int32),
                image_id=image_id,
            )
        )

    cache = SfmCache.get_cache(colmap_path / "_cache", "sfm_dataset_cache", "Cache for SFM dataset")
    return loaded_cameras, loaded_images, cache


class TracklessColmapSceneSource:
    """Reusable camera metadata and bounded point queries for partial ``SfmScene`` creation."""

    def __init__(self, colmap_path: str | pathlib.Path, block_size: int = 1_000_000):
        self._colmap_path = pathlib.Path(colmap_path).absolute()
        self._point_source = ColmapBinaryPointSource(self._colmap_path, block_size=block_size)
        self._cameras: dict[int, SfmCameraMetadata] | None = None
        self._images: list[SfmPosedImageMetadata] | None = None
        self._cache: SfmCache | None = None

    def _ensure_metadata(self) -> None:
        """Load cameras/images and create the scene cache only when training needs them."""

        if self._cameras is not None:
            return
        cameras, images, cache = _load_metadata_without_points(self._colmap_path, self._point_source.sparse_path)
        self._cameras = cameras
        self._images = images
        self._cache = cache

    @property
    def full_fingerprint(self) -> str:
        return self._point_source.full_fingerprint

    @property
    def point_source(self) -> ColmapBinaryPointSource:
        return self._point_source

    @property
    def point_count(self) -> int:
        return self._point_source.point_count

    @property
    def fingerprint(self) -> str:
        return self._point_source.fingerprint

    def bounds(self) -> np.ndarray | None:
        return self._point_source.bounds()

    def count_points(self, bbox: np.ndarray | None = None, bounds_mode: BoundsMode = "open") -> int:
        return self._point_source.count_points(bbox=bbox, bounds_mode=bounds_mode)

    def count_points_and_check_strict_id_order(
        self, bbox: np.ndarray | None = None, bounds_mode: BoundsMode = "open"
    ) -> tuple[int, bool]:
        return self._point_source.count_points_and_check_strict_id_order(bbox=bbox, bounds_mode=bounds_mode)

    def metadata_scene(self, scene_bbox: np.ndarray):
        """Create a metadata-only ``SfmScene`` with an explicit finite bounding box."""

        self._ensure_metadata()
        assert self._cameras is not None
        assert self._images is not None
        assert self._cache is not None
        from .sfm_scene import SfmScene

        scene_bbox = ColmapBinaryPointSource._validate_bbox(scene_bbox)
        assert scene_bbox is not None
        if not np.all(np.isfinite(scene_bbox)):
            raise ValueError("metadata scene_bbox must contain only finite values")

        return SfmScene(
            cameras=self._cameras,
            images=self._images,
            points=np.empty((0, 3), dtype=np.float32),
            points_err=np.empty((0,), dtype=np.float32),
            points_rgb=np.empty((0, 3), dtype=np.uint8),
            scene_bbox=scene_bbox.copy(),
            transformation_matrix=np.eye(4, dtype=np.float64),
            cache=self._cache,
        )

    def scene_for_bbox(
        self,
        bbox: np.ndarray,
        bounds_mode: BoundsMode = "open",
        expected_selected_count: int | None = None,
    ):
        """Create an ``SfmScene`` containing only points selected by ``bbox``."""

        self._ensure_metadata()
        assert self._cameras is not None
        assert self._images is not None
        assert self._cache is not None
        from .sfm_scene import SfmScene

        bbox = ColmapBinaryPointSource._validate_bbox(bbox)
        assert bbox is not None
        selection = self._point_source.query_points(
            bbox=bbox,
            bounds_mode=bounds_mode,
            expected_selected_count=expected_selected_count,
        )
        return SfmScene(
            cameras=self._cameras,
            images=self._images,
            points=selection.points,
            points_err=selection.points_err,
            points_rgb=selection.points_rgb,
            scene_bbox=bbox.astype(np.float32),
            transformation_matrix=np.eye(4, dtype=np.float64),
            cache=self._cache,
        )


def load_colmap_scene_partial(
    colmap_path: str | pathlib.Path,
    bbox: np.ndarray,
    bounds_mode: BoundsMode = "open",
    block_size: int = 1_000_000,
    expected_selected_count: int | None = None,
):
    """Load a bbox-selected ``SfmScene`` through the trackless binary fast path."""

    return TracklessColmapSceneSource(colmap_path, block_size=block_size).scene_for_bbox(
        bbox=bbox,
        bounds_mode=bounds_mode,
        expected_selected_count=expected_selected_count,
    )


def load_colmap_metadata_scene(
    colmap_path: str | pathlib.Path,
    scene_bbox: np.ndarray,
    block_size: int = 1_000_000,
):
    """Load cameras/images and an explicit finite bbox without materializing any points."""

    return TracklessColmapSceneSource(colmap_path, block_size=block_size).metadata_scene(scene_bbox=scene_bbox)


__all__ = [
    "BoundsMode",
    "ColmapBinaryPointSource",
    "ColmapPointSelection",
    "TracklessColmapSceneSource",
    "UnsupportedColmapPartialLoadError",
    "load_colmap_scene_partial",
    "load_colmap_metadata_scene",
    "probe_trackless_colmap_binary",
]
