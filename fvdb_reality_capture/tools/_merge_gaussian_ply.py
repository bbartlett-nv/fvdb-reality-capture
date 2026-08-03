# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""Bounded-memory merging of fVDB Gaussian PLY files."""

from __future__ import annotations

import dataclasses
import numbers
import os
import pathlib
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, BinaryIO

import numpy as np
import torch

_FVDB_PLY_MAGIC = "fvdb_ply_af_8198767135"
_FVDB_PLY_VERSION = "fvdb_ply 1.0.0"
_FVDB_GAUSSIAN_VERSION_COMMENT = f"fvdb_gs_ply_version {_FVDB_PLY_VERSION}"
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_METADATA_KEY_LENGTH = 256
_METADATA_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")

_PLY_SCALAR_DTYPES = {
    "char": np.dtype("i1"),
    "int8": np.dtype("i1"),
    "uchar": np.dtype("u1"),
    "uint8": np.dtype("u1"),
    "short": np.dtype("<i2"),
    "int16": np.dtype("<i2"),
    "ushort": np.dtype("<u2"),
    "uint16": np.dtype("<u2"),
    "int": np.dtype("<i4"),
    "int32": np.dtype("<i4"),
    "uint": np.dtype("<u4"),
    "uint32": np.dtype("<u4"),
    "float": np.dtype("<f4"),
    "float32": np.dtype("<f4"),
    "double": np.dtype("<f8"),
    "float64": np.dtype("<f8"),
}

_METADATA_PLY_TYPES = {
    ("f", 4): ("float", np.dtype("<f4")),
    ("f", 8): ("double", np.dtype("<f8")),
    ("i", 4): ("int", np.dtype("<i4")),
    ("u", 4): ("uint", np.dtype("<u4")),
    ("i", 2): ("short", np.dtype("<i2")),
    ("u", 2): ("ushort", np.dtype("<u2")),
    ("i", 1): ("char", np.dtype("i1")),
    ("u", 1): ("uchar", np.dtype("u1")),
}

CoreBoundingBox = tuple[float, float, float, float, float, float]
InclusiveMaxFaces = tuple[bool, bool, bool]


@dataclasses.dataclass(frozen=True)
class GaussianPlyMergeSource:
    """One input PLY and, optionally, the core region it owns.

    Core minima are always inclusive. Each maximum face is exclusive unless the
    corresponding inclusive_max entry is true. Adjacent chunks should use an
    exclusive maximum for the lower chunk and an inclusive minimum for the upper
    chunk so a center on their shared face has exactly one owner.
    """

    path: str | pathlib.Path
    core_bbox: CoreBoundingBox | None = None
    inclusive_max: InclusiveMaxFaces = (False, False, False)


@dataclasses.dataclass(frozen=True)
class GaussianPlyMergeResult:
    """Summary returned by merge_gaussian_ply_files."""

    output_path: pathlib.Path
    source_count: int
    input_gaussians: int
    output_gaussians: int
    filtered_gaussians: int
    retained_per_source: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _PlyProperty:
    name: str
    dtype: np.dtype


@dataclasses.dataclass(frozen=True)
class _PlyElement:
    name: str
    count: int
    properties: tuple[_PlyProperty, ...]
    data_offset: int
    record_size: int


@dataclasses.dataclass(frozen=True)
class _GaussianPlyInfo:
    path: pathlib.Path
    vertex: _PlyElement
    vertex_dtype: np.dtype
    schema: tuple[tuple[str, str], ...]
    stat_signature: tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _TensorMetadata:
    key: str
    array: np.ndarray
    ply_type: str


def gaussian_center_ownership_mask(
    centers: np.ndarray,
    core_bbox: CoreBoundingBox,
    inclusive_max: InclusiveMaxFaces = (False, False, False),
) -> np.ndarray:
    """Return the centers owned by a half-open chunk core.

    Args:
        centers: Array with shape (N, 3).
        core_bbox: Bounds ordered as (min_x, min_y, min_z, max_x, max_y, max_z).
        inclusive_max: Whether each maximum face is inclusive. This is normally
            true only at the outermost face of the complete chunk grid.
    """

    centers_array = np.asarray(centers)
    if centers_array.ndim != 2 or centers_array.shape[1] != 3:
        raise ValueError(f"centers must have shape (N, 3), got {centers_array.shape}")
    if not np.issubdtype(centers_array.dtype, np.floating) and not np.issubdtype(centers_array.dtype, np.integer):
        raise ValueError(f"centers must contain real numeric values, got dtype {centers_array.dtype}")

    lower, upper, normalized_inclusive_max = _validate_core_bounds(core_bbox, inclusive_max)
    if centers_array.dtype == np.dtype(np.float32):
        # Gaussian means are persisted as float32. Compare them to the representable
        # float32 faces so decimal CLI bounds cannot exclude a rounded boundary mean.
        lower = lower.astype(np.float32)
        upper = upper.astype(np.float32)
    mask = np.isfinite(centers_array).all(axis=1)
    mask &= (centers_array >= lower).all(axis=1)
    for axis, maximum_is_inclusive in enumerate(normalized_inclusive_max):
        if maximum_is_inclusive:
            mask &= centers_array[:, axis] <= upper[axis]
        else:
            mask &= centers_array[:, axis] < upper[axis]
    return mask


def merge_gaussian_ply_files(
    sources: Sequence[str | pathlib.Path | GaussianPlyMergeSource],
    output_path: str | pathlib.Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    overwrite: bool = False,
    records_per_block: int = 16_384,
) -> GaussianPlyMergeResult:
    """Merge compatible fVDB Gaussian PLYs with bounded host memory.

    The function makes two passes when any source has a core filter: one to
    determine the output vertex count and one to copy retained records. At most
    records_per_block vertex records are decoded at once. With no core filters,
    the first pass uses header counts and the second pass performs raw block
    copies.

    Input PLY metadata is intentionally ignored. metadata is encoded into the
    output and supports the same scalar and tensor types as fVDB's native
    Gaussian PLY writer. Empty individual inputs are supported. An all-empty or
    all-filtered merge is rejected because fVDB cannot load a zero-vertex
    Gaussian PLY.
    """

    if not isinstance(records_per_block, numbers.Integral) or isinstance(records_per_block, bool):
        raise ValueError("records_per_block must be a positive integer")
    records_per_block = int(records_per_block)
    if records_per_block <= 0:
        raise ValueError("records_per_block must be a positive integer")

    normalized_sources = tuple(_normalize_source(source) for source in sources)
    if not normalized_sources:
        raise ValueError("At least one Gaussian PLY source is required")

    destination = pathlib.Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")

    source_paths = tuple(pathlib.Path(source.path).expanduser().resolve() for source in normalized_sources)
    if destination in source_paths:
        raise ValueError("The output path must not be one of the input PLY paths")

    infos = tuple(_read_gaussian_ply_info(path) for path in source_paths)
    expected_schema = infos[0].schema
    for info in infos[1:]:
        if info.schema != expected_schema:
            raise ValueError(
                f"Incompatible Gaussian vertex schema in {info.path}; expected the schema from {infos[0].path}"
            )

    retained_counts = tuple(
        _count_retained_vertices(info, source, records_per_block)
        for info, source in zip(infos, normalized_sources, strict=True)
    )
    input_count = sum(info.vertex.count for info in infos)
    output_count = sum(retained_counts)
    if output_count == 0:
        raise ValueError("The merge retained no Gaussians; fVDB cannot load a zero-vertex Gaussian PLY")
    normalized_metadata = _normalize_metadata(metadata or {})

    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as output_file:
            temporary_path = pathlib.Path(output_file.name)
            output_file.write(_make_output_header(infos[0].vertex.properties, output_count, normalized_metadata))

            actual_output_count = 0
            for info, source, expected_retained in zip(infos, normalized_sources, retained_counts, strict=True):
                _ensure_source_unchanged(info)
                copied = _copy_retained_vertices(info, source, output_file, records_per_block)
                _ensure_source_unchanged(info)
                if copied != expected_retained:
                    raise RuntimeError(
                        f"Source {info.path} changed while it was being merged: expected {expected_retained} "
                        f"retained vertices, copied {copied}"
                    )
                actual_output_count += copied

            if actual_output_count != output_count:
                raise RuntimeError(f"Expected to write {output_count} vertices, wrote {actual_output_count}")
            for tensor_metadata in normalized_metadata[1]:
                output_file.write(memoryview(tensor_metadata.array).cast("B"))
            output_file.flush()
            os.fsync(output_file.fileno())

        if overwrite:
            os.replace(temporary_path, destination)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                raise FileExistsError(f"Output path was created while merging: {destination}") from error
            temporary_path.unlink()
            temporary_path = None
        _fsync_directory(destination.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return GaussianPlyMergeResult(
        output_path=destination,
        source_count=len(infos),
        input_gaussians=input_count,
        output_gaussians=output_count,
        filtered_gaussians=input_count - output_count,
        retained_per_source=retained_counts,
    )


def validate_gaussian_ply_file(
    path: str | pathlib.Path,
    *,
    expected_vertex_count: int | None = None,
) -> int:
    """Validate a complete fVDB Gaussian PLY and return its declared vertex count."""

    if expected_vertex_count is not None:
        if isinstance(expected_vertex_count, bool) or not isinstance(expected_vertex_count, numbers.Integral):
            raise TypeError("expected_vertex_count must be a nonnegative integer or None")
        expected_vertex_count = int(expected_vertex_count)
        if expected_vertex_count < 0:
            raise ValueError("expected_vertex_count must be a nonnegative integer or None")

    info = _read_gaussian_ply_info(pathlib.Path(path).expanduser().resolve())
    if expected_vertex_count is not None and info.vertex.count != expected_vertex_count:
        raise ValueError(
            f"Gaussian PLY vertex count mismatch for {info.path}: expected {expected_vertex_count:,}, "
            f"found {info.vertex.count:,}"
        )
    return info.vertex.count


def _fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalize_source(source: str | pathlib.Path | GaussianPlyMergeSource) -> GaussianPlyMergeSource:
    if isinstance(source, GaussianPlyMergeSource):
        if source.core_bbox is None:
            if tuple(source.inclusive_max) != (False, False, False):
                raise ValueError("inclusive_max requires core_bbox")
        else:
            _validate_core_bounds(source.core_bbox, source.inclusive_max)
        return source
    if isinstance(source, (str, pathlib.Path)):
        return GaussianPlyMergeSource(source)
    raise TypeError(f"Unsupported Gaussian PLY source type: {type(source).__name__}")


def _validate_core_bounds(
    core_bbox: CoreBoundingBox, inclusive_max: InclusiveMaxFaces
) -> tuple[np.ndarray, np.ndarray, InclusiveMaxFaces]:
    bounds = np.asarray(core_bbox, dtype=np.float64)
    if bounds.shape != (6,):
        raise ValueError("core_bbox must contain (min_x, min_y, min_z, max_x, max_y, max_z)")
    if not np.isfinite(bounds).all():
        raise ValueError("core_bbox values must all be finite")
    lower = bounds[:3]
    upper = bounds[3:]
    if np.any(lower >= upper):
        raise ValueError("Each core_bbox minimum must be less than its maximum")
    if len(inclusive_max) != 3 or any(not isinstance(value, (bool, np.bool_)) for value in inclusive_max):
        raise ValueError("inclusive_max must contain three boolean values")
    normalized_inclusive_max = tuple(bool(value) for value in inclusive_max)
    return lower, upper, normalized_inclusive_max  # type: ignore[return-value]


def _read_gaussian_ply_info(path: pathlib.Path) -> _GaussianPlyInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian PLY source does not exist: {path}")

    with path.open("rb") as input_file:
        header_lines, header_size = _read_ply_header(input_file, path)

    elements: list[tuple[str, int, tuple[_PlyProperty, ...]]] = []
    comments: list[str] = []
    current_name: str | None = None
    current_count = 0
    current_properties: list[_PlyProperty] = []
    saw_format = False

    def finish_current_element() -> None:
        nonlocal current_name, current_count, current_properties
        if current_name is not None:
            elements.append((current_name, current_count, tuple(current_properties)))
        current_name = None
        current_count = 0
        current_properties = []

    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        directive = parts[0]
        if directive == "format":
            if parts != ["format", "binary_little_endian", "1.0"]:
                raise ValueError(f"Only binary_little_endian PLY 1.0 files are supported: {path}")
            saw_format = True
        elif directive == "comment":
            comments.append(line[len("comment") :].lstrip())
        elif directive == "obj_info":
            continue
        elif directive == "element":
            finish_current_element()
            if len(parts) != 3:
                raise ValueError(f"Malformed element declaration in {path}: {line}")
            current_name = parts[1]
            try:
                current_count = int(parts[2])
            except ValueError as error:
                raise ValueError(f"Invalid element count in {path}: {line}") from error
            if current_count < 0:
                raise ValueError(f"Negative element count in {path}: {line}")
        elif directive == "property":
            if current_name is None:
                raise ValueError(f"PLY property declared before an element in {path}")
            if len(parts) >= 2 and parts[1] == "list":
                raise ValueError(f"List PLY properties are not supported for streaming merge: {path}")
            if len(parts) != 3 or parts[1] not in _PLY_SCALAR_DTYPES:
                raise ValueError(f"Unsupported PLY property in {path}: {line}")
            current_properties.append(_PlyProperty(parts[2], _PLY_SCALAR_DTYPES[parts[1]]))
        elif directive == "end_header":
            break
        else:
            raise ValueError(f"Unsupported PLY header directive in {path}: {line}")
    finish_current_element()

    if not saw_format:
        raise ValueError(f"PLY format declaration is missing from {path}")
    version_comments = [comment for comment in comments if comment.startswith("fvdb_gs_ply_version ")]
    if version_comments and version_comments != [_FVDB_GAUSSIAN_VERSION_COMMENT]:
        raise ValueError(f"Unsupported fVDB Gaussian PLY version in {path}: {version_comments}")

    if sum(name == "vertex" for name, _, _ in elements) != 1:
        raise ValueError(f"Expected exactly one vertex element in Gaussian PLY: {path}")

    parsed_elements: list[_PlyElement] = []
    data_offset = header_size
    for name, count, properties in elements:
        if len({prop.name for prop in properties}) != len(properties):
            raise ValueError(f"Duplicate property name in PLY element '{name}': {path}")
        record_size = sum(prop.dtype.itemsize for prop in properties)
        if count and record_size == 0:
            raise ValueError(f"PLY element '{name}' has records but no properties: {path}")
        parsed_elements.append(_PlyElement(name, count, properties, data_offset, record_size))
        data_offset += count * record_size

    stat = path.stat()
    if data_offset != stat.st_size:
        raise ValueError(
            f"PLY body size does not match its element declarations in {path}: expected {data_offset} bytes, "
            f"found {stat.st_size}"
        )

    vertex = next(element for element in parsed_elements if element.name == "vertex")
    _validate_gaussian_vertex_properties(vertex.properties, path)
    vertex_dtype = np.dtype([(prop.name, prop.dtype) for prop in vertex.properties], align=False)
    schema = tuple((prop.name, prop.dtype.str) for prop in vertex.properties)
    return _GaussianPlyInfo(
        path=path,
        vertex=vertex,
        vertex_dtype=vertex_dtype,
        schema=schema,
        stat_signature=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
    )


def _read_ply_header(input_file: BinaryIO, path: pathlib.Path) -> tuple[list[str], int]:
    lines: list[str] = []
    consumed = 0
    while consumed <= _MAX_HEADER_BYTES:
        raw_line = input_file.readline(_MAX_HEADER_BYTES - consumed + 1)
        if not raw_line:
            raise ValueError(f"PLY header is missing end_header: {path}")
        consumed += len(raw_line)
        if consumed > _MAX_HEADER_BYTES:
            raise ValueError(f"PLY header exceeds {_MAX_HEADER_BYTES} bytes: {path}")
        try:
            line = raw_line.rstrip(b"\r\n").decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"PLY header is not valid UTF-8: {path}") from error
        lines.append(line)
        if line == "end_header":
            break
    if not lines or lines[0] != "ply":
        raise ValueError(f"Not a PLY file: {path}")
    return lines, consumed


def _validate_gaussian_vertex_properties(properties: tuple[_PlyProperty, ...], path: pathlib.Path) -> None:
    base_names = (
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    names = tuple(prop.name for prop in properties)
    if names[: len(base_names)] != base_names:
        raise ValueError(f"PLY does not have the canonical fVDB Gaussian base property schema: {path}")
    if any(prop.dtype != np.dtype("<f4") for prop in properties):
        raise ValueError(f"All fVDB Gaussian vertex properties must be float32: {path}")

    remaining_names = names[len(base_names) :]
    sh0_names = tuple(name for name in remaining_names if name.startswith("f_dc_"))
    shn_names = tuple(name for name in remaining_names if name.startswith("f_rest_"))
    expected_names = tuple(f"f_dc_{index}" for index in range(len(sh0_names))) + tuple(
        f"f_rest_{index}" for index in range(len(shn_names))
    )
    if not sh0_names or remaining_names != expected_names:
        raise ValueError(f"PLY does not have contiguous fVDB Gaussian SH properties: {path}")
    if len(shn_names) % len(sh0_names) != 0:
        raise ValueError(f"Higher-order SH property count is not divisible by the channel count: {path}")


def _count_retained_vertices(info: _GaussianPlyInfo, source: GaussianPlyMergeSource, records_per_block: int) -> int:
    if source.core_bbox is None:
        return info.vertex.count
    return sum(
        int(_ownership_mask_for_records(records, source).sum())
        for records in _iter_vertex_blocks(info, records_per_block)
    )


def _copy_retained_vertices(
    info: _GaussianPlyInfo,
    source: GaussianPlyMergeSource,
    output_file: BinaryIO,
    records_per_block: int,
) -> int:
    copied = 0
    if source.core_bbox is None:
        with info.path.open("rb") as input_file:
            input_file.seek(info.vertex.data_offset)
            remaining = info.vertex.count
            while remaining:
                record_count = min(records_per_block, remaining)
                data = input_file.read(record_count * info.vertex.record_size)
                if len(data) != record_count * info.vertex.record_size:
                    raise RuntimeError(f"Unexpected end of vertex data while reading {info.path}")
                output_file.write(data)
                copied += record_count
                remaining -= record_count
        return copied

    for records in _iter_vertex_blocks(info, records_per_block):
        mask = _ownership_mask_for_records(records, source)
        retained_records = records[mask]
        output_file.write(retained_records.tobytes(order="C"))
        copied += int(mask.sum())
    return copied


def _iter_vertex_blocks(info: _GaussianPlyInfo, records_per_block: int) -> Iterator[np.ndarray]:
    with info.path.open("rb") as input_file:
        input_file.seek(info.vertex.data_offset)
        remaining = info.vertex.count
        while remaining:
            record_count = min(records_per_block, remaining)
            byte_count = record_count * info.vertex.record_size
            data = input_file.read(byte_count)
            if len(data) != byte_count:
                raise RuntimeError(f"Unexpected end of vertex data while reading {info.path}")
            yield np.frombuffer(data, dtype=info.vertex_dtype, count=record_count)
            remaining -= record_count


def _ownership_mask_for_records(records: np.ndarray, source: GaussianPlyMergeSource) -> np.ndarray:
    if source.core_bbox is None:
        return np.ones(records.shape[0], dtype=np.bool_)
    centers = np.column_stack((records["x"], records["y"], records["z"]))
    return gaussian_center_ownership_mask(centers, source.core_bbox, source.inclusive_max)


def _ensure_source_unchanged(info: _GaussianPlyInfo) -> None:
    stat = info.path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature != info.stat_signature:
        raise RuntimeError(f"Source PLY changed while preparing the merge: {info.path}")


def _normalize_metadata(
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[_TensorMetadata, ...]]:
    for key in metadata:
        if not isinstance(key, str):
            raise TypeError(f"PLY metadata keys must be strings, got {type(key).__name__}")

    comments: list[str] = []
    tensors: list[_TensorMetadata] = []
    for key in sorted(metadata):
        if not key or len(key) > _MAX_METADATA_KEY_LENGTH or _METADATA_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(
                f"Invalid PLY metadata key '{key}'; keys must contain 1-{_MAX_METADATA_KEY_LENGTH} "
                "alphanumeric or underscore characters"
            )
        if key == "vertex":
            raise ValueError("PLY metadata key 'vertex' is reserved")

        value = metadata[key]
        comment_prefix = f"comment {_FVDB_PLY_MAGIC}{key}"
        if isinstance(value, str):
            if "\n" in value or "\r" in value:
                raise ValueError(f"String PLY metadata value for '{key}' must not contain a newline")
            comments.append(f"{comment_prefix}|str|{value}")
        elif isinstance(value, numbers.Integral):
            integer_value = int(value)
            if not -(2**63) <= integer_value < 2**63:
                raise ValueError(f"Integer PLY metadata value for '{key}' is outside the int64 range")
            comments.append(f"{comment_prefix}|int64|{integer_value}")
        elif isinstance(value, numbers.Real):
            comments.append(f"{comment_prefix}|double|{float(value):.17f}")
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            array = value.detach().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else value
            array = np.asarray(array)
            ply_type_and_dtype = _METADATA_PLY_TYPES.get((array.dtype.kind, array.dtype.itemsize))
            if ply_type_and_dtype is None:
                raise ValueError(
                    f"Unsupported tensor metadata dtype for '{key}': {array.dtype}; expected float32, float64, "
                    "int32, uint32, int16, uint16, int8, or uint8"
                )
            ply_type, output_dtype = ply_type_and_dtype
            little_endian_array = np.ascontiguousarray(array.astype(output_dtype, copy=False))
            shape = ",".join(str(size) for size in little_endian_array.shape)
            shape_suffix = f"{little_endian_array.ndim},{shape}" if shape else f"{little_endian_array.ndim},"
            comments.append(f"{comment_prefix}|tensor|{shape_suffix}")
            tensors.append(_TensorMetadata(key, little_endian_array, ply_type))
        else:
            raise TypeError(
                f"Unsupported PLY metadata value for '{key}': {type(value).__name__}; expected str, int, "
                "float, torch.Tensor, or numpy.ndarray"
            )
    return tuple(comments), tuple(tensors)


def _make_output_header(
    properties: tuple[_PlyProperty, ...],
    vertex_count: int,
    metadata: tuple[tuple[str, ...], tuple[_TensorMetadata, ...]],
) -> bytes:
    comments, tensors = metadata
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment {_FVDB_GAUSSIAN_VERSION_COMMENT}",
        *comments,
        f"element vertex {vertex_count}",
    ]
    lines.extend(f"property float {prop.name}" for prop in properties)
    for tensor in tensors:
        lines.extend((f"element {tensor.key} {tensor.array.size}", f"property {tensor.ply_type} value"))
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = [
    "GaussianPlyMergeResult",
    "GaussianPlyMergeSource",
    "validate_gaussian_ply_file",
    "gaussian_center_ownership_mask",
    "merge_gaussian_ply_files",
]
