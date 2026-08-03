# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import pathlib
import socket
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .spatial_chunking import BoundingBox, ChunkSpec

ChunkStatus = Literal["pending", "complete"]

_MANIFEST_KIND = "fvdb-reality-capture-chunk-run"
_MANIFEST_VERSION = 2
_MANIFEST_FILENAME = "chunk_manifest.json"
_TOP_LEVEL_KEYS = {"kind", "version", "run_signature", "chunk_plan", "chunk_states"}
_COMPLETION_KEYS = (
    "artifact_path",
    "artifact_size",
    "artifact_sha256",
    "initial_count",
    "final_count",
)
_STATE_KEYS = {"id", "index", "status", *_COMPLETION_KEYS}
_SPEC_KEYS = {"id", "index", "grid_index", "core_bbox", "train_bbox", "inclusive_max"}
_ARTIFACT_HASH_BLOCK_SIZE = 1024 * 1024
_SHA256_HEX_LENGTH = 64

_LOCK_KIND = "fvdb-reality-capture-chunk-run-lock"
_LOCK_VERSION = 1
_LOCK_SUFFIX = ".chunk-run.lock"
_MAX_LOCK_METADATA_BYTES = 16 * 1024


class ChunkManifestError(RuntimeError):
    """Raised when a chunk work directory cannot be initialized or safely resumed."""


class ChunkRunLockError(ChunkManifestError):
    """Raised when exclusive ownership of a chunk work directory cannot be obtained."""


class ChunkRunLock:
    """Exclusive whole-run lock for a chunk work directory.

    The lock is a sibling of the work directory so it can be acquired before the directory or
    manifest exists. Existing locks are never removed automatically: after verifying that no run
    is active, an operator must explicitly remove a stale lock using the path in the error.
    """

    def __init__(self, work_dir: str | pathlib.Path):
        self._work_dir = pathlib.Path(work_dir).expanduser().resolve()
        if not self._work_dir.name:
            raise ValueError("work_dir must identify a named directory")
        self._path = self._work_dir.parent / f".{self._work_dir.name}{_LOCK_SUFFIX}"
        self._identity: tuple[int, int] | None = None
        self._metadata: dict[str, Any] | None = None

    @property
    def work_dir(self) -> pathlib.Path:
        return self._work_dir

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def metadata(self) -> dict[str, Any] | None:
        if self._metadata is None:
            return None
        return dict(self._metadata)

    def acquire(self) -> "ChunkRunLock":
        """Acquire the lock atomically or fail without modifying an existing lock."""

        if self._identity is not None:
            raise ChunkRunLockError(f"Chunk run lock is already held by this object: {self._path}")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ChunkRunLockError(f"Failed to create the chunk lock parent directory: {self._path.parent}") from error

        metadata = {
            "kind": _LOCK_KIND,
            "version": _LOCK_VERSION,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "work_dir": str(self._work_dir),
        }
        payload = (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC

        try:
            file_descriptor = os.open(self._path, flags, 0o644)
        except FileExistsError as error:
            raise ChunkRunLockError(self._locked_error_message()) from error
        except OSError as error:
            raise ChunkRunLockError(f"Failed to create chunk run lock: {self._path}") from error

        identity: tuple[int, int] | None = None
        try:
            identity = _file_identity(os.fstat(file_descriptor))
            with os.fdopen(file_descriptor, mode="wb") as lock_file:
                file_descriptor = -1
                lock_file.write(payload)
                lock_file.flush()
                os.fsync(lock_file.fileno())
        except BaseException:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            if identity is not None:
                _unlink_if_identity_matches(self._path, identity)
            raise

        self._identity = identity
        self._metadata = metadata
        return self

    def release(self) -> None:
        """Release this object's lock without removing a replacement lock file."""

        if self._identity is None:
            return
        identity = self._identity
        try:
            removed = _unlink_if_identity_matches(self._path, identity)
        except OSError as error:
            raise ChunkRunLockError(f"Failed to release chunk run lock: {self._path}") from error
        self._identity = None
        self._metadata = None
        if not removed:
            raise ChunkRunLockError(
                f"Chunk run lock changed while held; refusing to remove the unknown lock file: {self._path}"
            )

    def __enter__(self) -> "ChunkRunLock":
        return self.acquire()

    def __exit__(self, exception_type, exception, traceback) -> bool:
        del exception_type, traceback
        try:
            self.release()
        except ChunkRunLockError as release_error:
            if exception is None:
                raise
            exception.add_note(str(release_error))
        return False

    def _locked_error_message(self) -> str:
        owner = _describe_lock_owner(self._path)
        return (
            f"Chunk work directory is already locked: {self._work_dir}. Lock file: {self._path}. "
            f"Recorded owner: {owner}. Another reconstruction may still be active; the lock was not removed. "
            "If you have verified that no run is using this work directory, remove the lock file manually and retry."
        )


@dataclass(frozen=True)
class ChunkRunState:
    """Immutable public view of one chunk status."""

    spec: ChunkSpec
    status: ChunkStatus
    artifact_path: pathlib.Path | None
    artifact_size: int | None
    artifact_sha256: str | None
    initial_count: int | None
    final_count: int | None


def serialize_chunk_spec(spec: ChunkSpec) -> dict[str, Any]:
    """Serialize a :class:`ChunkSpec` using only version-stable JSON values."""

    if not isinstance(spec, ChunkSpec):
        raise TypeError(f"Expected ChunkSpec, got {type(spec).__name__}")
    return {
        "index": spec.index,
        "id": spec.id,
        "grid_index": list(spec.grid_index),
        "core_bbox": list(spec.core_bbox),
        "train_bbox": list(spec.train_bbox),
        "inclusive_max": list(spec.inclusive_max),
    }


def _canonical_json(value: Any, name: str) -> tuple[Any, str]:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON-serializable values") from error
    return json.loads(encoded), encoded


def _decode_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ChunkManifestError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


def _decode_bbox(value: Any, name: str) -> BoundingBox:
    if not isinstance(value, list) or len(value) != 6:
        raise ChunkManifestError(f"{name} must be a list of six finite numbers")
    decoded = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, numbers.Real):
            raise ChunkManifestError(f"{name} must be a list of six finite numbers")
        coordinate_float = float(coordinate)
        if not math.isfinite(coordinate_float):
            raise ChunkManifestError(f"{name} must be a list of six finite numbers")
        decoded.append(coordinate_float)
    if any(decoded[axis] >= decoded[axis + 3] for axis in range(3)):
        raise ChunkManifestError(f"{name} minima must be strictly less than maxima")
    return decoded[0], decoded[1], decoded[2], decoded[3], decoded[4], decoded[5]


def _deserialize_chunk_spec(value: Any, position: int) -> ChunkSpec:
    if not isinstance(value, dict) or set(value) != _SPEC_KEYS:
        raise ChunkManifestError(f"chunk_plan[{position}] does not match the v1 ChunkSpec schema")

    index = _decode_nonnegative_int(value["index"], f"chunk_plan[{position}].index")
    if index != position:
        raise ChunkManifestError(f"chunk_plan[{position}].index must equal its plan position")
    chunk_id = value["id"]
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ChunkManifestError(f"chunk_plan[{position}].id must be a nonempty string")

    raw_grid_index = value["grid_index"]
    if not isinstance(raw_grid_index, list) or len(raw_grid_index) != 3:
        raise ChunkManifestError(f"chunk_plan[{position}].grid_index must contain three nonnegative integers")
    grid_index = tuple(
        _decode_nonnegative_int(coordinate, f"chunk_plan[{position}].grid_index") for coordinate in raw_grid_index
    )

    raw_inclusive_max = value["inclusive_max"]
    if (
        not isinstance(raw_inclusive_max, list)
        or len(raw_inclusive_max) != 3
        or any(not isinstance(flag, bool) for flag in raw_inclusive_max)
    ):
        raise ChunkManifestError(f"chunk_plan[{position}].inclusive_max must contain three booleans")

    core_bbox = _decode_bbox(value["core_bbox"], f"chunk_plan[{position}].core_bbox")
    train_bbox = _decode_bbox(value["train_bbox"], f"chunk_plan[{position}].train_bbox")
    if any(train_bbox[axis] > core_bbox[axis] for axis in range(3)) or any(
        train_bbox[axis + 3] < core_bbox[axis + 3] for axis in range(3)
    ):
        raise ChunkManifestError(f"chunk_plan[{position}].train_bbox must contain core_bbox")

    return ChunkSpec(
        index=index,
        id=chunk_id,
        grid_index=(grid_index[0], grid_index[1], grid_index[2]),
        core_bbox=core_bbox,
        train_bbox=train_bbox,
        inclusive_max=(raw_inclusive_max[0], raw_inclusive_max[1], raw_inclusive_max[2]),
    )


def _serialize_plan(chunks: Sequence[ChunkSpec]) -> list[dict[str, Any]]:
    serialized = [serialize_chunk_spec(chunk) for chunk in chunks]
    if not serialized:
        raise ValueError("chunks must contain at least one ChunkSpec")
    specs = tuple(_deserialize_chunk_spec(value, position) for position, value in enumerate(serialized))
    if len({spec.id for spec in specs}) != len(specs):
        raise ValueError("ChunkSpec ids must be unique")
    if len({spec.grid_index for spec in specs}) != len(specs):
        raise ValueError("ChunkSpec grid indices must be unique")
    return serialized


def _pending_state(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "index": spec["index"],
        "status": "pending",
        "artifact_path": None,
        "artifact_size": None,
        "artifact_sha256": None,
        "initial_count": None,
        "final_count": None,
    }


def _demote_state(record: dict[str, Any]) -> None:
    record["status"] = "pending"
    for key in _COMPLETION_KEYS:
        record[key] = None


class ChunkRunManifest:
    """Persistent v2 state for an immutable spatial chunk run."""

    def __init__(self, work_dir: pathlib.Path, document: dict[str, Any]):
        self._work_dir = work_dir
        self._path = work_dir / _MANIFEST_FILENAME
        self._document = document
        self._specs = tuple(
            _deserialize_chunk_spec(value, position) for position, value in enumerate(document["chunk_plan"])
        )
        self._spec_by_id = {spec.id: spec for spec in self._specs}

    @classmethod
    def open_or_create(
        cls,
        work_dir: str | pathlib.Path,
        run_signature: Mapping[str, Any],
        chunks: Sequence[ChunkSpec],
    ) -> "ChunkRunManifest":
        """Create a new work directory or resume an exactly matching manifest."""

        work_dir_path = pathlib.Path(work_dir).expanduser().resolve()
        if not isinstance(run_signature, Mapping):
            raise TypeError("run_signature must be a mapping")
        normalized_signature, expected_signature_json = _canonical_json(dict(run_signature), "run_signature")
        if not isinstance(normalized_signature, dict):
            raise ValueError("run_signature must serialize to a JSON object")
        serialized_plan = _serialize_plan(chunks)
        _, expected_plan_json = _canonical_json(serialized_plan, "chunks")

        if work_dir_path.exists():
            if not work_dir_path.is_dir():
                raise ChunkManifestError(f"Chunk work path exists and is not a directory: {work_dir_path}")
            manifest_path = work_dir_path / _MANIFEST_FILENAME
            if not _is_regular_nonempty_file(manifest_path):
                raise ChunkManifestError(
                    f"Existing chunk work directory has no valid {_MANIFEST_FILENAME}; refusing to modify it: "
                    f"{work_dir_path}"
                )
            document = _read_document(manifest_path)
            _validate_document(document)
            _, existing_signature_json = _canonical_json(document["run_signature"], "stored run_signature")
            _, existing_plan_json = _canonical_json(document["chunk_plan"], "stored chunk_plan")
            if existing_signature_json != expected_signature_json:
                raise ChunkManifestError("Existing chunk manifest run signature does not match the requested run")
            if existing_plan_json != expected_plan_json:
                raise ChunkManifestError("Existing chunk manifest plan does not exactly match the requested chunk plan")
            manifest = cls(work_dir_path, document)
            manifest.validate_artifacts()
            return manifest

        try:
            work_dir_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ChunkManifestError(
                f"Chunk work path appeared while it was being initialized: {work_dir_path}"
            ) from error

        document = {
            "kind": _MANIFEST_KIND,
            "version": _MANIFEST_VERSION,
            "run_signature": normalized_signature,
            "chunk_plan": serialized_plan,
            "chunk_states": [_pending_state(spec) for spec in serialized_plan],
        }
        manifest = cls(work_dir_path, document)
        manifest._write_document(document)
        return manifest

    @property
    def work_dir(self) -> pathlib.Path:
        return self._work_dir

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def run_signature(self) -> dict[str, Any]:
        normalized, _ = _canonical_json(self._document["run_signature"], "stored run_signature")
        return normalized

    @property
    def chunk_plan(self) -> tuple[ChunkSpec, ...]:
        return self._specs

    @property
    def states(self) -> tuple[ChunkRunState, ...]:
        return tuple(self._public_state(record) for record in self._document["chunk_states"])

    @property
    def pending_chunks(self) -> tuple[ChunkSpec, ...]:
        return tuple(state.spec for state in self.states if state.status == "pending")

    @property
    def completed_chunks(self) -> tuple[ChunkSpec, ...]:
        return tuple(state.spec for state in self.states if state.status == "complete")

    def state(self, chunk_id: str) -> ChunkRunState:
        return self._public_state(self._state_record(chunk_id))

    def mark_complete(
        self,
        chunk_id: str,
        artifact_path: str | pathlib.Path,
        initial_count: int,
        final_count: int,
    ) -> ChunkRunState:
        """Atomically record a complete chunk after validating its artifact."""

        initial_count = _decode_nonnegative_int(initial_count, "initial_count")
        final_count = _decode_nonnegative_int(final_count, "final_count")
        stored_artifact_path = self._normalize_artifact_path(artifact_path)
        resolved_artifact_path = self._resolve_artifact_path(stored_artifact_path)
        if not _is_regular_nonempty_file(resolved_artifact_path):
            raise ValueError("Completed chunk artifact must be an existing regular nonempty file")
        integrity = _artifact_integrity(resolved_artifact_path)
        if integrity is None:
            raise ValueError("Completed chunk artifact changed or became unreadable while it was being hashed")
        artifact_size, artifact_sha256 = integrity

        document = self._copy_document()
        record = self._state_record(chunk_id, document)
        record.update(
            {
                "status": "complete",
                "artifact_path": stored_artifact_path,
                "artifact_size": artifact_size,
                "artifact_sha256": artifact_sha256,
                "initial_count": initial_count,
                "final_count": final_count,
            }
        )
        self._commit(document)
        return self.state(chunk_id)

    def mark_pending(self, chunk_id: str) -> ChunkRunState:
        """Atomically demote a chunk without deleting any artifact."""

        document = self._copy_document()
        record = self._state_record(chunk_id, document)
        _demote_state(record)
        self._commit(document)
        return self.state(chunk_id)

    def validate_artifacts(self) -> tuple[str, ...]:
        """Demote completed chunks whose artifacts no longer match their stored integrity data."""

        document = self._copy_document()
        demoted = []
        for record in document["chunk_states"]:
            if record["status"] == "complete" and not self._artifact_matches(record):
                demoted.append(record["id"])
                _demote_state(record)
        if demoted:
            self._commit(document)
        return tuple(demoted)

    def _public_state(self, record: Mapping[str, Any]) -> ChunkRunState:
        artifact_path = record["artifact_path"]
        return ChunkRunState(
            spec=self._spec_by_id[record["id"]],
            status=record["status"],
            artifact_path=self._resolve_artifact_path(artifact_path) if artifact_path is not None else None,
            artifact_size=record["artifact_size"],
            artifact_sha256=record["artifact_sha256"],
            initial_count=record["initial_count"],
            final_count=record["final_count"],
        )

    def _state_record(self, chunk_id: str, document: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(chunk_id, str):
            raise TypeError("chunk_id must be a string")
        source = self._document if document is None else document
        for record in source["chunk_states"]:
            if record["id"] == chunk_id:
                return record
        raise KeyError(f"Unknown chunk id: {chunk_id}")

    def _normalize_artifact_path(self, artifact_path: str | pathlib.Path) -> str:
        candidate = pathlib.Path(artifact_path).expanduser()
        if not candidate.is_absolute():
            candidate = self._work_dir / candidate
        candidate = pathlib.Path(os.path.abspath(candidate))
        try:
            return candidate.relative_to(self._work_dir).as_posix()
        except ValueError:
            return str(candidate)

    def _resolve_artifact_path(self, stored_path: str) -> pathlib.Path:
        candidate = pathlib.Path(stored_path)
        if not candidate.is_absolute():
            candidate = self._work_dir / candidate
        return pathlib.Path(os.path.abspath(candidate))

    def _artifact_matches(self, record: Mapping[str, Any]) -> bool:
        integrity = _artifact_integrity(self._resolve_artifact_path(record["artifact_path"]))
        if integrity is None:
            return False
        return integrity == (record["artifact_size"], record["artifact_sha256"])

    def _copy_document(self) -> dict[str, Any]:
        normalized, _ = _canonical_json(self._document, "chunk manifest")
        return normalized

    def _commit(self, document: dict[str, Any]) -> None:
        _validate_document(document)
        self._write_document(document)
        self._document = document

    def _write_document(self, document: dict[str, Any]) -> None:
        temporary_path: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{_MANIFEST_FILENAME}.",
                suffix=".tmp",
                dir=self._work_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = pathlib.Path(temporary_file.name)
                json.dump(document, temporary_file, indent=2, sort_keys=True, allow_nan=False)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self._path)
            temporary_path = None
            _fsync_directory(self._work_dir)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _file_identity(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _unlink_if_identity_matches(path: pathlib.Path, expected_identity: tuple[int, int]) -> bool:
    try:
        current_identity = _file_identity(path.lstat())
    except FileNotFoundError:
        return False
    if current_identity != expected_identity:
        return False
    path.unlink()
    return True


def _describe_lock_owner(path: pathlib.Path) -> str:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return "metadata unavailable because the lock path is not a regular file"
        with path.open("rb") as lock_file:
            encoded = lock_file.read(_MAX_LOCK_METADATA_BYTES + 1)
        if len(encoded) > _MAX_LOCK_METADATA_BYTES:
            return "metadata unavailable because the lock file is unexpectedly large"
        metadata = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "metadata unavailable or incomplete"
    if not isinstance(metadata, dict):
        return "metadata unavailable or incomplete"

    details = []
    pid = metadata.get("pid")
    hostname = metadata.get("hostname")
    created_at = metadata.get("created_at_utc")
    if isinstance(pid, int) and not isinstance(pid, bool):
        details.append(f"pid={pid}")
    if isinstance(hostname, str) and hostname:
        details.append(f"hostname={hostname}")
    if isinstance(created_at, str) and created_at:
        details.append(f"created_at_utc={created_at}")
    return ", ".join(details) if details else "metadata unavailable or incomplete"


def _fsync_directory(path: pathlib.Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    directory_descriptor = os.open(path, flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _artifact_integrity(path: pathlib.Path) -> tuple[int, str] | None:
    try:
        path_stat = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_size <= 0:
        return None

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(path, flags)
    except OSError:
        return None

    try:
        opened_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_size <= 0
            or _file_identity(opened_stat) != _file_identity(path_stat)
        ):
            return None
        opened_signature = (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
            opened_stat.st_ctime_ns,
        )
        digest = hashlib.sha256()
        with os.fdopen(file_descriptor, mode="rb") as artifact_file:
            file_descriptor = -1
            while True:
                block = artifact_file.read(_ARTIFACT_HASH_BLOCK_SIZE)
                if not block:
                    break
                digest.update(block)
            final_stat = os.fstat(artifact_file.fileno())
    except (OSError, ValueError):
        return None
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)

    final_signature = (
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        final_stat.st_ctime_ns,
    )
    if final_signature != opened_signature:
        return None
    return final_stat.st_size, digest.hexdigest()


def _is_regular_nonempty_file(path: pathlib.Path) -> bool:
    try:
        file_stat = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(file_stat.st_mode) and file_stat.st_size > 0


def _read_document(path: pathlib.Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ChunkManifestError(f"Failed to read chunk manifest: {path}") from error
    if not isinstance(document, dict):
        raise ChunkManifestError("Chunk manifest root must be a JSON object")
    return document


def _validate_document(document: dict[str, Any]) -> None:
    if set(document) != _TOP_LEVEL_KEYS:
        raise ChunkManifestError("Existing file does not match the chunk manifest v2 schema")
    if (
        document["kind"] != _MANIFEST_KIND
        or isinstance(document["version"], bool)
        or document["version"] != _MANIFEST_VERSION
    ):
        raise ChunkManifestError("Existing file is not a supported chunk manifest v2")
    if not isinstance(document["run_signature"], dict):
        raise ChunkManifestError("Chunk manifest run_signature must be a JSON object")
    _canonical_json(document["run_signature"], "stored run_signature")

    plan = document["chunk_plan"]
    states = document["chunk_states"]
    if not isinstance(plan, list) or not plan:
        raise ChunkManifestError("Chunk manifest plan must be a nonempty list")
    specs = tuple(_deserialize_chunk_spec(value, position) for position, value in enumerate(plan))
    if len({spec.id for spec in specs}) != len(specs) or len({spec.grid_index for spec in specs}) != len(specs):
        raise ChunkManifestError("Chunk manifest plan contains duplicate chunk ids or grid indices")
    if not isinstance(states, list) or len(states) != len(specs):
        raise ChunkManifestError("Chunk manifest states must correspond exactly to its chunk plan")

    for position, (spec, state_value) in enumerate(zip(specs, states, strict=True)):
        if not isinstance(state_value, dict) or set(state_value) != _STATE_KEYS:
            raise ChunkManifestError(f"chunk_states[{position}] does not match the v2 state schema")
        if state_value["id"] != spec.id or state_value["index"] != spec.index:
            raise ChunkManifestError(f"chunk_states[{position}] does not correspond to chunk_plan[{position}]")
        status_value = state_value["status"]
        if status_value == "pending":
            if any(state_value[key] is not None for key in _COMPLETION_KEYS):
                raise ChunkManifestError(f"Pending chunk_states[{position}] must not contain completion data")
        elif status_value == "complete":
            if not isinstance(state_value["artifact_path"], str) or not state_value["artifact_path"]:
                raise ChunkManifestError(f"Complete chunk_states[{position}] must contain an artifact path")
            artifact_size = _decode_nonnegative_int(
                state_value["artifact_size"], f"chunk_states[{position}].artifact_size"
            )
            if artifact_size == 0:
                raise ChunkManifestError(f"chunk_states[{position}].artifact_size must be positive")
            artifact_sha256 = state_value["artifact_sha256"]
            if (
                not isinstance(artifact_sha256, str)
                or len(artifact_sha256) != _SHA256_HEX_LENGTH
                or any(character not in "0123456789abcdef" for character in artifact_sha256)
            ):
                raise ChunkManifestError(
                    f"chunk_states[{position}].artifact_sha256 must be a lowercase 64-character SHA-256 digest"
                )
            _decode_nonnegative_int(state_value["initial_count"], f"chunk_states[{position}].initial_count")
            _decode_nonnegative_int(state_value["final_count"], f"chunk_states[{position}].final_count")
        else:
            raise ChunkManifestError(f"chunk_states[{position}] has unknown status {status_value!r}")


__all__ = [
    "ChunkManifestError",
    "ChunkRunLock",
    "ChunkRunLockError",
    "ChunkRunManifest",
    "ChunkRunState",
    "ChunkStatus",
    "serialize_chunk_spec",
]
