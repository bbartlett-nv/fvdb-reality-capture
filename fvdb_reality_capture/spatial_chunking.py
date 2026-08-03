# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import itertools
import math
import numbers
from dataclasses import dataclass
from typing import Sequence

import numpy as np

BoundingBox = tuple[float, float, float, float, float, float]
GridIndex = tuple[int, int, int]
GridShape = tuple[int, int, int]

_UNBOUNDED_SCENE_BBOX = np.array([-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf], dtype=np.float64)


def _validate_bbox(bbox: Sequence[float], name: str) -> BoundingBox:
    try:
        bbox_array = np.asarray(bbox, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain six numeric values") from error

    if bbox_array.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {bbox_array.shape}")
    if not np.all(np.isfinite(bbox_array)):
        raise ValueError(f"{name} must contain only finite values, got {bbox_array.tolist()}")

    bbox_min = bbox_array[:3]
    bbox_max = bbox_array[3:]
    if np.any(bbox_min >= bbox_max):
        raise ValueError(f"{name} minima must be strictly less than maxima, got {bbox_array.tolist()}")
    if not np.all(np.isfinite(bbox_max - bbox_min)):
        raise ValueError(f"{name} extents must be finite, got {bbox_array.tolist()}")

    return (
        float(bbox_array[0]),
        float(bbox_array[1]),
        float(bbox_array[2]),
        float(bbox_array[3]),
        float(bbox_array[4]),
        float(bbox_array[5]),
    )


def _validate_grid_shape(nchunks: Sequence[int]) -> GridShape:
    try:
        values = tuple(nchunks)
    except TypeError as error:
        raise ValueError("nchunks must contain exactly three positive integers") from error

    if len(values) != 3:
        raise ValueError(f"nchunks must contain exactly three positive integers, got {values}")

    validated = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f"nchunks must contain exactly three positive integers, got {values}")
        validated.append(int(value))
    return validated[0], validated[1], validated[2]


def _validate_overlap(overlap_pct: float) -> float:
    if isinstance(overlap_pct, bool) or not isinstance(overlap_pct, numbers.Real):
        raise ValueError(f"overlap_pct must be a finite number in [0, 1), got {overlap_pct!r}")
    overlap = float(overlap_pct)
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap_pct must be a finite number in [0, 1), got {overlap_pct!r}")
    return overlap


def _bbox_from_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> BoundingBox:
    return (
        float(bbox_min[0]),
        float(bbox_min[1]),
        float(bbox_min[2]),
        float(bbox_max[0]),
        float(bbox_max[1]),
        float(bbox_max[2]),
    )


def resolve_chunk_domain(
    scene_bbox: Sequence[float] | None = None,
    point_bbox: Sequence[float] | None = None,
) -> BoundingBox:
    """Resolve the finite spatial domain to partition.

    A finite scene bounding box represents an explicit reconstruction domain and takes precedence over point bounds.
    ``SfmScene`` uses ``[-inf, -inf, -inf, inf, inf, inf]`` when no explicit domain is available; that sentinel falls
    back to ``point_bbox``. Other malformed or partly non-finite scene bounds are rejected rather than silently ignored.
    """

    if scene_bbox is not None:
        try:
            scene_bbox_array = np.asarray(scene_bbox, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("scene_bbox must contain six numeric values") from error
        if scene_bbox_array.shape != (6,):
            raise ValueError(f"scene_bbox must have shape (6,), got {scene_bbox_array.shape}")
        if not np.array_equal(scene_bbox_array, _UNBOUNDED_SCENE_BBOX):
            return _validate_bbox(scene_bbox_array, "scene_bbox")

    if point_bbox is None:
        raise ValueError("A finite point_bbox is required when scene_bbox is not explicitly bounded")
    return _validate_bbox(point_bbox, "point_bbox")


@dataclass(frozen=True)
class ChunkSpec:
    """A deterministic spatial chunk with separate optimization and output-ownership bounds."""

    index: int
    id: str
    grid_index: GridIndex
    core_bbox: BoundingBox
    train_bbox: BoundingBox
    inclusive_max: tuple[bool, bool, bool]

    def owns_center(self, center: Sequence[float]) -> bool:
        """Return whether a Gaussian center belongs to this chunk's non-overlapping core."""

        return chunk_owns_center(center, self)


def plan_spatial_chunks(
    domain_bbox: Sequence[float],
    nchunks: Sequence[int],
    overlap_pct: float,
) -> tuple[ChunkSpec, ...]:
    """Partition a finite domain into deterministic core boxes and clipped training halos.

    ``overlap_pct`` is the total overlap between adjacent, equal-width cores as a fraction of one core width. Each
    participating core therefore receives half of that overlap as a halo on each internal face. Axes with one chunk do
    not receive a halo. Core maximum faces are half-open except at the global maximum, so every finite in-domain center
    has exactly one owner.
    """

    domain = _validate_bbox(domain_bbox, "domain_bbox")
    grid_shape = _validate_grid_shape(nchunks)
    overlap = _validate_overlap(overlap_pct)

    domain_min = np.asarray(domain[:3], dtype=np.float64)
    domain_max = np.asarray(domain[3:], dtype=np.float64)
    edges = tuple(
        np.linspace(domain_min[axis], domain_max[axis], grid_shape[axis] + 1, dtype=np.float64) for axis in range(3)
    )

    chunks = []
    for linear_index, grid_index in enumerate(
        itertools.product(range(grid_shape[0]), range(grid_shape[1]), range(grid_shape[2]))
    ):
        core_min = np.array([edges[axis][grid_index[axis]] for axis in range(3)], dtype=np.float64)
        core_max = np.array([edges[axis][grid_index[axis] + 1] for axis in range(3)], dtype=np.float64)
        core_size = core_max - core_min
        halo = np.array(
            [0.0 if grid_shape[axis] == 1 else 0.5 * overlap * core_size[axis] for axis in range(3)],
            dtype=np.float64,
        )
        train_min = np.maximum(domain_min, core_min - halo)
        train_max = np.minimum(domain_max, core_max + halo)

        chunks.append(
            ChunkSpec(
                index=linear_index,
                id=f"chunk_{linear_index:04d}",
                grid_index=grid_index,
                core_bbox=_bbox_from_corners(core_min, core_max),
                train_bbox=_bbox_from_corners(train_min, train_max),
                inclusive_max=(
                    grid_index[0] == grid_shape[0] - 1,
                    grid_index[1] == grid_shape[1] - 1,
                    grid_index[2] == grid_shape[2] - 1,
                ),
            )
        )

    return tuple(chunks)


def chunk_owns_center(center: Sequence[float], chunk: ChunkSpec) -> bool:
    """Test half-open core ownership for one center, including only global maximum faces."""

    try:
        center_array = np.asarray(center, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("center must contain three numeric values") from error
    if center_array.shape != (3,):
        raise ValueError(f"center must have shape (3,), got {center_array.shape}")
    if not np.all(np.isfinite(center_array)):
        return False

    core_min = np.asarray(chunk.core_bbox[:3], dtype=np.float64)
    core_max = np.asarray(chunk.core_bbox[3:], dtype=np.float64)
    if np.any(center_array < core_min):
        return False
    for axis, include_max in enumerate(chunk.inclusive_max):
        outside_max = center_array[axis] > core_max[axis] if include_max else center_array[axis] >= core_max[axis]
        if outside_max:
            return False
    return True


def chunk_ownership_mask(centers: np.ndarray, chunk: ChunkSpec) -> np.ndarray:
    """Vectorized equivalent of :func:`chunk_owns_center` for an ``(..., 3)`` NumPy array."""

    centers_array = np.asarray(centers)
    if centers_array.ndim < 1 or centers_array.shape[-1] != 3:
        raise ValueError(f"centers must have shape (..., 3), got {centers_array.shape}")
    if not np.issubdtype(centers_array.dtype, np.integer) and not np.issubdtype(centers_array.dtype, np.floating):
        raise ValueError(f"centers must contain real numeric values, got dtype {centers_array.dtype}")

    core_min = np.asarray(chunk.core_bbox[:3], dtype=np.float64)
    core_max = np.asarray(chunk.core_bbox[3:], dtype=np.float64)
    mask = np.all(np.isfinite(centers_array), axis=-1) & np.all(centers_array >= core_min, axis=-1)
    for axis, include_max in enumerate(chunk.inclusive_max):
        if include_max:
            mask &= centers_array[..., axis] <= core_max[axis]
        else:
            mask &= centers_array[..., axis] < core_max[axis]
    return mask


__all__ = [
    "BoundingBox",
    "ChunkSpec",
    "GridIndex",
    "GridShape",
    "chunk_ownership_mask",
    "chunk_owns_center",
    "plan_spatial_chunks",
    "resolve_chunk_domain",
]
