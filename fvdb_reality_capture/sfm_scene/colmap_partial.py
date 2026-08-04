# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""Bounded-memory access to binary COLMAP point models.

Fixed-size, trackless ``points3D.bin`` records use a memory-mapped fast path. Models
with visibility tracks are validated and streamed without retaining a global record
offset table. A trackless suffix is memory mapped only after its extent and all of
its zero track lengths have been validated. Cameras and
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
_TRACK_ELEMENT_SIZE = 8
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
    """Raised when a COLMAP model cannot use bounded-memory partial loading."""


@dataclass(frozen=True)
class _FileIdentity:
    """Filesystem identity used to reject changes during or after validation."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, stat: os.stat_result) -> "_FileIdentity":
        return cls(
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
        )


@dataclass(frozen=True)
class _PointFileLayout:
    """Validated split between streamed variable records and a fixed suffix."""

    identity: _FileIdentity
    point_count: int
    variable_prefix_count: int
    tracked_point_count: int
    track_observation_count: int
    trackless_suffix_offset: int

    @property
    def trackless_suffix_count(self) -> int:
        return self.point_count - self.variable_prefix_count


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


def _identity_matches(identity: _FileIdentity, stat: os.stat_result) -> bool:
    current = _FileIdentity.from_stat(stat)
    return current == identity


def _raise_if_validation_source_changed(
    points_path: pathlib.Path,
    identity: _FileIdentity,
    stat: os.stat_result,
) -> None:
    if not _identity_matches(identity, stat):
        raise UnsupportedColmapPartialLoadError(
            f"COLMAP point source changed while its binary layout was being validated: {points_path}"
        )


def _resolve_binary_sparse_path(colmap_path: pathlib.Path) -> pathlib.Path:
    sparse_path = _resolve_sparse_path(colmap_path)
    required_files = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required_files if not (sparse_path / name).is_file()]
    if missing:
        raise UnsupportedColmapPartialLoadError(
            "Bounded-memory COLMAP loading requires a binary sparse model; "
            f"missing {', '.join(missing)} in {sparse_path}"
        )
    return sparse_path


def _validate_trackless_suffix(
    points_path: pathlib.Path,
    identity: _FileIdentity,
    suffix_offset: int,
    suffix_start_index: int,
    suffix_count: int,
    block_size: int,
) -> None:
    """Prove that a size-derived fixed-record suffix has zero track lengths."""

    if suffix_count == 0:
        return
    for start in range(0, suffix_count, block_size):
        mapped_count = min(block_size, suffix_count - start)
        try:
            records = np.memmap(
                points_path,
                mode="r",
                dtype=_POINT_RECORD_DTYPE,
                offset=suffix_offset + start * _TRACKLESS_POINT_RECORD_SIZE,
                shape=(mapped_count,),
            )
        except (OSError, ValueError) as error:
            raise UnsupportedColmapPartialLoadError(
                f"Could not map fixed-record suffix block {start:,} in {points_path}: {error}"
            ) from error

        track_lengths = None
        nonzero = None
        try:
            track_lengths = records["track_length"]
            nonzero = track_lengths != 0
            if np.any(nonzero):
                local_index = int(np.argmax(nonzero))
                record_index = suffix_start_index + start + local_index
                track_length = int(track_lengths[local_index])
                raise UnsupportedColmapPartialLoadError(
                    f"COLMAP point record {record_index:,} in {points_path} declares a visibility track of "
                    f"length {track_length:,}, but the file has only enough bytes for fixed-size records"
                )
        finally:
            del nonzero
            del track_lengths
            del records

    try:
        _raise_if_validation_source_changed(points_path, identity, points_path.stat())
    except OSError as error:
        raise UnsupportedColmapPartialLoadError(
            f"Could not re-stat COLMAP point source after validating it: {points_path}: {error}"
        ) from error


def _inspect_points_file(points_path: pathlib.Path, block_size: int) -> _PointFileLayout:
    """Validate every record boundary and locate a provably trackless suffix."""

    try:
        identity = _FileIdentity.from_stat(points_path.stat())
    except OSError as error:
        raise UnsupportedColmapPartialLoadError(f"Could not stat COLMAP point source {points_path}: {error}") from error

    if identity.size < _POINTS_HEADER_SIZE:
        raise UnsupportedColmapPartialLoadError(f"COLMAP point file is truncated: {points_path}")

    try:
        with points_path.open("rb") as point_file:
            _raise_if_validation_source_changed(points_path, identity, os.fstat(point_file.fileno()))
            point_count_bytes = point_file.read(_POINTS_HEADER_SIZE)
            if len(point_count_bytes) != _POINTS_HEADER_SIZE:
                raise UnsupportedColmapPartialLoadError(f"Could not read the COLMAP point count from {points_path}")
            point_count = int(struct.unpack("<Q", point_count_bytes)[0])

            minimum_file_size = _POINTS_HEADER_SIZE + point_count * _TRACKLESS_POINT_RECORD_SIZE
            if identity.size < minimum_file_size:
                raise UnsupportedColmapPartialLoadError(
                    f"COLMAP point file is truncated: its header declares {point_count:,} points requiring at least "
                    f"{minimum_file_size:,} bytes, but {points_path} contains {identity.size:,} bytes"
                )
            extra_track_bytes = identity.size - minimum_file_size
            if extra_track_bytes % _TRACK_ELEMENT_SIZE != 0:
                raise UnsupportedColmapPartialLoadError(
                    f"COLMAP point file has {extra_track_bytes:,} bytes beyond its fixed records; visibility tracks "
                    f"must occupy a multiple of {_TRACK_ELEMENT_SIZE} bytes: {points_path}"
                )

            record_index = 0
            tracked_point_count = 0
            track_observation_count = 0
            offset = _POINTS_HEADER_SIZE
            suffix_offset = identity.size
            while record_index < point_count:
                remaining_count = point_count - record_index
                remaining_bytes = identity.size - offset
                minimum_remaining = remaining_count * _TRACKLESS_POINT_RECORD_SIZE
                if remaining_bytes < minimum_remaining:
                    raise UnsupportedColmapPartialLoadError(
                        f"COLMAP point record {record_index:,} is truncated in {points_path}: {remaining_bytes:,} "
                        f"bytes remain, but at least {minimum_remaining:,} bytes are required"
                    )

                if remaining_bytes == minimum_remaining:
                    suffix_offset = offset
                    break

                fixed_record = point_file.read(_TRACKLESS_POINT_RECORD_SIZE)
                if len(fixed_record) != _TRACKLESS_POINT_RECORD_SIZE:
                    raise UnsupportedColmapPartialLoadError(
                        f"Could not read fixed fields for COLMAP point record {record_index:,} from {points_path}"
                    )
                track_length = int(struct.unpack_from("<Q", fixed_record, 43)[0])
                if track_length > 0:
                    tracked_point_count += 1
                track_observation_count += track_length
                track_bytes = track_length * _TRACK_ELEMENT_SIZE
                minimum_after_record = (remaining_count - 1) * _TRACKLESS_POINT_RECORD_SIZE
                available_track_bytes = remaining_bytes - _TRACKLESS_POINT_RECORD_SIZE - minimum_after_record
                if track_bytes > available_track_bytes:
                    raise UnsupportedColmapPartialLoadError(
                        f"COLMAP point record {record_index:,} in {points_path} declares a visibility track of "
                        f"length {track_length:,} ({track_bytes:,} bytes), but at most {available_track_bytes:,} "
                        "track bytes are available while reserving the remaining point records"
                    )
                point_file.seek(track_bytes, os.SEEK_CUR)
                offset += _TRACKLESS_POINT_RECORD_SIZE + track_bytes
                record_index += 1
            else:
                suffix_offset = offset

            if record_index == point_count and offset != identity.size:
                raise UnsupportedColmapPartialLoadError(
                    f"COLMAP point file contains {identity.size - offset:,} trailing bytes after "
                    f"{point_count:,} declared records: {points_path}"
                )
            _raise_if_validation_source_changed(points_path, identity, os.fstat(point_file.fileno()))
    except OSError as error:
        raise UnsupportedColmapPartialLoadError(
            f"Could not validate COLMAP point source {points_path}: {error}"
        ) from error

    try:
        _raise_if_validation_source_changed(points_path, identity, points_path.stat())
    except OSError as error:
        raise UnsupportedColmapPartialLoadError(
            f"Could not re-stat COLMAP point source after validating it: {points_path}: {error}"
        ) from error

    layout = _PointFileLayout(
        identity=identity,
        point_count=point_count,
        variable_prefix_count=record_index,
        tracked_point_count=tracked_point_count,
        track_observation_count=track_observation_count,
        trackless_suffix_offset=suffix_offset,
    )
    _validate_trackless_suffix(
        points_path=points_path,
        identity=identity,
        suffix_offset=layout.trackless_suffix_offset,
        suffix_start_index=layout.variable_prefix_count,
        suffix_count=layout.trackless_suffix_count,
        block_size=block_size,
    )
    return layout


def probe_trackless_colmap_binary(colmap_path: str | pathlib.Path) -> tuple[bool, str]:
    """Return whether a binary model supports bounded-memory point loading.

    The historical function name is retained for API compatibility; mixed visibility
    tracks are supported.
    """

    colmap_path = pathlib.Path(colmap_path)
    try:
        sparse_path = _resolve_binary_sparse_path(colmap_path)
        layout = _inspect_points_file(sparse_path / "points3D.bin", block_size=1_000_000)
    except UnsupportedColmapPartialLoadError as error:
        return False, str(error)

    if layout.variable_prefix_count == 0:
        return True, f"binary COLMAP model with {layout.point_count:,} trackless points"
    return (
        True,
        "binary COLMAP model with bounded variable-length tracks "
        f"({layout.tracked_point_count:,} tracked points with {layout.track_observation_count:,} observations "
        f"in {layout.variable_prefix_count:,} streamed records; "
        f"{layout.trackless_suffix_count:,} trackless fixed-size records)",
    )


class ColmapBinaryPointSource:
    """Bounded-memory point queries over fixed- or variable-record ``points3D.bin``."""

    def __init__(self, colmap_path: str | pathlib.Path, block_size: int = 1_000_000):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self._colmap_path = pathlib.Path(colmap_path).absolute()
        self._sparse_path = _resolve_binary_sparse_path(self._colmap_path)
        self._points_path = self._sparse_path / "points3D.bin"
        self._block_size = int(block_size)
        layout = _inspect_points_file(self._points_path, block_size=self._block_size)
        self._initial_identity = layout.identity
        self._point_count = layout.point_count
        self._variable_prefix_count = layout.variable_prefix_count
        self._tracked_point_count = layout.tracked_point_count
        self._track_observation_count = layout.track_observation_count
        self._trackless_suffix_offset = layout.trackless_suffix_offset
        self._trackless_suffix_count = layout.trackless_suffix_count

        self._records: np.memmap | None
        if self._trackless_suffix_count == 0:
            self._records = None
        else:
            self._records = np.memmap(
                self._points_path,
                mode="r",
                dtype=_POINT_RECORD_DTYPE,
                offset=self._trackless_suffix_offset,
                shape=(self._trackless_suffix_count,),
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
                self._assert_open_file_unchanged(point_file)
                for block in iter(lambda: point_file.read(_CONTENT_HASH_BLOCK_SIZE), b""):
                    digest.update(block)
                self._assert_open_file_unchanged(point_file)
            self._assert_unchanged()
            self._full_fingerprint = digest.hexdigest()
        return self._full_fingerprint

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def variable_prefix_count(self) -> int:
        """Number of records streamed before the validated fixed-size suffix."""

        return self._variable_prefix_count

    @property
    def tracked_point_count(self) -> int:
        """Number of point records with non-empty visibility tracks."""

        return self._tracked_point_count

    @property
    def track_observation_count(self) -> int:
        """Total number of (image, point2D) observations skipped from tracks."""

        return self._track_observation_count

    @property
    def trackless_suffix_count(self) -> int:
        """Number of validated fixed-size records served by the memmap fast path."""

        return self._trackless_suffix_count

    @property
    def fingerprint(self) -> str:
        """A bounded-cost fingerprint of the point source and its filesystem identity."""

        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        self._assert_unchanged()
        digest = hashlib.sha256()
        identity = {
            "path": str(self._points_path.resolve()),
            "device": self._initial_identity.device,
            "inode": self._initial_identity.inode,
            "size": self._initial_identity.size,
            "mtime_ns": self._initial_identity.mtime_ns,
            "ctime_ns": self._initial_identity.ctime_ns,
            "point_count": self._point_count,
        }
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        with self._points_path.open("rb") as point_file:
            self._assert_open_file_unchanged(point_file)
            digest.update(point_file.read(_FINGERPRINT_SAMPLE_SIZE))
            if self._initial_identity.size > _FINGERPRINT_SAMPLE_SIZE:
                point_file.seek(max(0, self._initial_identity.size - _FINGERPRINT_SAMPLE_SIZE))
                digest.update(point_file.read(_FINGERPRINT_SAMPLE_SIZE))
            self._assert_open_file_unchanged(point_file)
        self._assert_unchanged()
        return digest.hexdigest()

    def _assert_stat_unchanged(self, stat: os.stat_result) -> None:
        if not _identity_matches(self._initial_identity, stat):
            raise RuntimeError(f"COLMAP point source changed after it was opened: {self._points_path}")

    def _assert_open_file_unchanged(self, point_file) -> None:
        try:
            self._assert_stat_unchanged(os.fstat(point_file.fileno()))
        except OSError as error:
            raise RuntimeError(f"Could not inspect open COLMAP point source {self._points_path}: {error}") from error

    def _assert_unchanged(self) -> None:
        try:
            self._assert_stat_unchanged(self._points_path.stat())
        except OSError as error:
            raise RuntimeError(f"Could not inspect COLMAP point source {self._points_path}: {error}") from error

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

    def _iter_variable_prefix_blocks(self):
        if self._variable_prefix_count == 0:
            return

        records = np.empty(min(self._block_size, self._variable_prefix_count), dtype=_POINT_RECORD_DTYPE)
        with self._points_path.open("rb") as point_file:
            self._assert_open_file_unchanged(point_file)
            point_file.seek(_POINTS_HEADER_SIZE)
            record_index = 0
            offset = _POINTS_HEADER_SIZE
            block_count = 0
            while record_index < self._variable_prefix_count:
                fixed_record = point_file.read(_TRACKLESS_POINT_RECORD_SIZE)
                if len(fixed_record) != _TRACKLESS_POINT_RECORD_SIZE:
                    raise RuntimeError(
                        f"Could not read fixed fields for COLMAP point record {record_index:,} "
                        f"from {self._points_path}"
                    )
                record = np.frombuffer(fixed_record, dtype=_POINT_RECORD_DTYPE, count=1)[0]
                track_bytes = int(record["track_length"]) * _TRACK_ELEMENT_SIZE
                record_stop = offset + _TRACKLESS_POINT_RECORD_SIZE + track_bytes
                if record_stop > self._trackless_suffix_offset:
                    raise RuntimeError(
                        f"COLMAP point record {record_index:,} crossed the validated fixed-record suffix in "
                        f"{self._points_path}"
                    )

                records[block_count] = record
                block_count += 1
                record_index += 1
                offset = record_stop
                point_file.seek(track_bytes, os.SEEK_CUR)

                if block_count == len(records):
                    yield records
                    block_count = 0

            if block_count > 0:
                yield records[:block_count]
            if offset != self._trackless_suffix_offset:
                raise RuntimeError(
                    f"Variable COLMAP records ended at byte {offset:,}, expected the validated suffix at "
                    f"byte {self._trackless_suffix_offset:,}: {self._points_path}"
                )
            self._assert_open_file_unchanged(point_file)

    def _iter_record_blocks(self):
        self._assert_unchanged()
        try:
            yield from self._iter_variable_prefix_blocks()
            if self._records is not None:
                for start in range(0, self._trackless_suffix_count, self._block_size):
                    yield self._records[start : min(start + self._block_size, self._trackless_suffix_count)]
        finally:
            self._assert_unchanged()

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
    """Reusable metadata and bounded point queries for partial ``SfmScene`` creation.

    The historical class name is retained for API compatibility; its point source
    supports both trackless and mixed-track binary COLMAP models.
    """

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
    def variable_prefix_count(self) -> int:
        return self._point_source.variable_prefix_count

    @property
    def tracked_point_count(self) -> int:
        return self._point_source.tracked_point_count

    @property
    def track_observation_count(self) -> int:
        return self._point_source.track_observation_count

    @property
    def trackless_suffix_count(self) -> int:
        return self._point_source.trackless_suffix_count

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
    """Load a bbox-selected ``SfmScene`` through the bounded mixed-track binary fast path."""

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
