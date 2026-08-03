# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import pathlib
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
import fvdb.viz as fviz
import numpy as np
import torch
from tyro.conf import Positional, arg

from fvdb_reality_capture import GaussianSplat3d, __version__
from fvdb_reality_capture.chunk_manifest import ChunkRunLock, ChunkRunManifest
from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.mask_utils import build_scene_cache_fingerprint
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatOptimizerMCMCConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)
from fvdb_reality_capture.sfm_scene import (
    SfmScene,
    SpatialScaleMode,
    TracklessColmapSceneSource,
    probe_trackless_colmap_binary,
)
from fvdb_reality_capture.spatial_chunking import ChunkSpec, plan_spatial_chunks, resolve_chunk_domain
from fvdb_reality_capture.tools import (
    GaussianPlyMergeSource,
    merge_gaussian_ply_files,
    validate_gaussian_ply_file,
)
from fvdb_reality_capture.transforms import (
    BaseTransform,
    Compose,
    CropScene,
    CropSceneToPoints,
    DownsampleImages,
    FilterImagesWithLowPoints,
    NormalizeScene,
    PercentileFilterPoints,
)

from ._common import (
    DatasetType,
    load_sfm_scene,
    save_model_from_runner,
)


_CHUNK_PIPELINE_VERSION = 2
_HASH_BLOCK_BYTES = 8 * 1024 * 1024
_WRITER_OWNER_FILENAME = ".frgs_chunk_owner.json"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _installed_distribution_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _training_implementation_identity() -> dict[str, object]:
    """Fingerprint Python training code and runtime package versions for safe resume."""

    package_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    source_count = 0
    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with source_path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(_HASH_BLOCK_BYTES), b""):
                digest.update(block)
        source_count += 1

    return {
        "chunk_pipeline_version": _CHUNK_PIPELINE_VERSION,
        "python_source_count": source_count,
        "python_source_sha256": digest.hexdigest(),
        "fvdb_reality_capture_version": __version__,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "opencv_version": cv2.__version__,
        "scipy_version": _installed_distribution_version("scipy"),
        "pycolmap_version": _installed_distribution_version("pycolmap"),
        "fvdb_core_version": _installed_distribution_version("fvdb-core"),
    }


def _hash_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    byte_view = memoryview(array).cast("B")
    for offset in range(0, len(byte_view), _HASH_BLOCK_BYTES):
        digest.update(byte_view[offset : offset + _HASH_BLOCK_BYTES])


def _scene_image_contents_sha256(scene: SfmScene) -> str:
    """Hash ordered training-image bytes while rejecting concurrent file changes."""

    def signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )

    digest = hashlib.sha256()
    for image in scene.images:
        image_path = Path(image.image_path).expanduser().resolve()
        digest.update(str(int(image.image_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(image_path).encode("utf-8"))
        digest.update(b"\0")

        path_stat = image_path.stat()
        with image_path.open("rb") as image_file:
            opened_stat = os.fstat(image_file.fileno())
            if (opened_stat.st_dev, opened_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
                raise RuntimeError(f"Training image changed while it was being opened: {image_path}")
            for block in iter(lambda: image_file.read(_HASH_BLOCK_BYTES), b""):
                digest.update(block)
            final_opened_stat = os.fstat(image_file.fileno())
        final_path_stat = image_path.stat()

        if signature(opened_stat) != signature(final_opened_stat) or signature(opened_stat) != signature(
            final_path_stat
        ):
            raise RuntimeError(f"Training image changed while it was being fingerprinted: {image_path}")
        digest.update(b"\0")

    return digest.hexdigest()


def _scene_resume_fingerprint(
    scene: SfmScene, *, include_points: bool, include_image_contents: bool = False
) -> dict[str, object]:
    scene_fingerprint = build_scene_cache_fingerprint(
        scene,
        algorithm="chunked_reconstruction_resume",
        algorithm_version=_CHUNK_PIPELINE_VERSION,
        settings={},
        include_source_images=True,
    )
    result: dict[str, object] = {
        "scene_schema_version": scene_fingerprint["schema_version"],
        "scene_sha256": scene_fingerprint["sha256"],
    }
    if include_image_contents:
        result["image_contents_sha256"] = _scene_image_contents_sha256(scene)
    if include_points:
        point_digest = hashlib.sha256()
        _hash_array(point_digest, "points", scene.points)
        _hash_array(point_digest, "points_rgb", scene.points_rgb)
        _hash_array(point_digest, "points_err", scene.points_err)
        result["point_arrays_sha256"] = point_digest.hexdigest()
    return result


@dataclass
class SceneTransformConfig:
    """
    Configure how an SfmScene is transformed before optimization.
    """

    # Downsample images by this factor
    image_downsample_factor: int = 4
    # JPEG quality to use when resaving images after downsampling
    rescale_jpeg_quality: int = 95
    # Percentile of points to filter out based on their distance from the median point
    points_percentile_filter: float = 0.0
    # Type of normalization to apply to the scene
    normalization_type: Literal["none", "pca", "ecef2enu", "similarity"] = "pca"
    # Whether to crop the scene to the bounding box or not
    crop_to_points: bool = False
    # Minimum number of 3D points that must be visible in an image for it to be included in the optimization
    min_points_per_image: int = 5
    # Optional bounding box in transformed space: (xmin, ymin, zmin, xmax, ymax, zmax)
    crop_bbox: tuple[float, float, float, float, float, float] | None = None

    @property
    def scene_transform(self) -> BaseTransform:
        # Dataset transform
        transforms = [
            NormalizeScene(normalization_type=self.normalization_type),
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
        ]
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))
        return Compose(*transforms)


@dataclass
class WriterConfig(GaussianSplatReconstructionWriterConfig):
    """
    Configuration for saving and logging metrics, images, and checkpoints.
    """

    # Path to save logs, checkpoints, and other output to.
    # Defaults to `frgs_logs` in the current working directory.
    log_path: pathlib.Path | None = pathlib.Path("frgs_logs")

    # How frequently to log metrics during reconstruction.
    log_every: int = 10


@dataclass
class Reconstruct(BaseCommand):
    """
    Reconstruct a Gaussian Splat Radiance Field from a dataset of posed images, and save the result as a PLY or USD file.


    Example usage:

        # Reconstruct a Gaussian splat radiance field from a Colmap dataset
        frgs reconstruct ./colmap_dataset -o ./output.ply

        # Reconstruct a Gaussian splat radiance field from a dataset of e57 files
        frgs reconstruct ./simple_directory_dataset --dataset-type e57 --out-path ./output.usdc
    """

    # Path to the dataset. For "colmap" datasets, this should be the
    # directory containing the `images` and `sparse` subdirectories. For "simple_directory" datasets,
    # this should be the directory containing the images and a `cameras.txt` file.
    dataset_path: Positional[Path]

    # Path to save the output PLY file.
    # Defaults to `out.ply` in the current working directory.
    # Path must end in .ply, .usdc, or .usdz.
    out_path: Annotated[Path, arg(aliases=["-o"])] = Path("out.ply")

    # Name of the run. If None, a name will be generated based on the current date and time.
    run_name: Annotated[str | None, arg(aliases=["-n"])] = None

    # Type of dataset to load.
    dataset_type: Annotated[DatasetType, arg(aliases=["-dt"])] = "colmap"

    # Use every n-th image as a validation image. If -1, do not use a validation set.
    use_every_n_as_val: Annotated[int, arg(aliases=["-vn"])] = -1

    # How frequently (in epochs) to update the viewer during reconstruction.
    # An epoch is one full pass through the dataset. If -1, do not visualize.
    update_viz_every: Annotated[float, arg(aliases=["-uv"])] = -1.0

    # The port to expose the viewer server on if update_viz_every > 0.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080

    # The IP address to expose the viewer server on if update_viz_every > 0.
    viewer_ip_address: Annotated[str, arg(aliases=["-ip"])] = "127.0.0.1"

    # Which device to use for reconstruction. Must be a cuda device. You can pass in a specific device index via
    # cuda:N where N is the device index, or "cuda" to use the default cuda device.
    # CPU is not supported. Default is "cuda".
    device: Annotated[str | torch.device, arg(aliases=["-d"])] = "cuda"

    # If set, show verbose debug messages.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Configuration parameters for the Gaussian splat reconstruction.
    cfg: GaussianSplatReconstructionConfig = field(default_factory=GaussianSplatReconstructionConfig)

    # Configuration for the transforms to apply to the scene before reconstruction.
    tx: SceneTransformConfig = field(default_factory=SceneTransformConfig)

    # Configuration for the optimizer used to reconstruct the Gaussian splat radiance field.
    opt: GaussianSplatOptimizerConfig = field(default_factory=GaussianSplatOptimizerConfig)

    # Configure saving and logging metrics, images, and checkpoints.
    io: WriterConfig = field(default_factory=WriterConfig)

    # Configuration to split the dataset into chunks for reconstruction.
    # If set to (1, 1, 1), the dataset will not be chunked.
    nchunks: Annotated[tuple[int, int, int], arg(aliases=["-nc"])] = (1, 1, 1)

    # Percentage of overlap between chunks if reconstructing in chunks. Must be in [0, 1).
    # Only used if nchunks is not (1, 1, 1).
    # Default is 0.1 (10% overlap).
    chunk_overlap_pct: Annotated[float, arg(aliases=["-nco"])] = 0.1

    # Persistent directory for chunk PLYs and the resume manifest. Defaults to <out_path>.chunks.
    chunk_work_dir: Path | None = None

    # Reuse completed chunk artifacts from a compatible manifest.
    resume_chunks: bool = True

    # Resolve the chunk domain and count initialization points without training or merging.
    chunk_plan_only: bool = False

    # Select the bounded-memory binary COLMAP initialization path for chunked reconstruction.
    chunked_colmap_load: Literal["auto", "require", "off"] = "auto"

    @property
    def resolved_chunk_work_dir(self) -> Path:
        if self.chunk_work_dir is not None:
            return self.chunk_work_dir.expanduser().resolve()
        return Path(f"{self.out_path}.chunks").expanduser().resolve()

    @property
    def resolved_chunk_run_name(self) -> str:
        if self.run_name is not None:
            return self.run_name
        work_dir = self.resolved_chunk_work_dir
        work_dir_digest = hashlib.sha256(str(work_dir).encode("utf-8")).hexdigest()[:12]
        return f"{work_dir.name}_{work_dir_digest}"

    @property
    def resolved_chunk_log_dir(self) -> Path | None:
        if self.io.log_path is None:
            return None
        return (self.io.log_path.expanduser().resolve() / self.resolved_chunk_run_name).resolve()

    def _validate_chunk_paths(self) -> None:
        output_path = self.out_path.expanduser().resolve()
        work_dir = self.resolved_chunk_work_dir
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"Chunked output directory does not exist: {output_path.parent}")
        if _paths_overlap(output_path, work_dir):
            raise ValueError("out_path and chunk_work_dir must not contain or replace one another")
        log_dir = self.resolved_chunk_log_dir
        if log_dir is None:
            return
        if _paths_overlap(log_dir, work_dir) or _paths_overlap(log_dir, output_path):
            raise ValueError("Chunk log, work, and output paths must not contain or replace one another")
        if not log_dir.exists():
            return
        if not log_dir.is_dir():
            raise FileExistsError(f"Chunk log path exists and is not a directory: {log_dir}")
        owner_path = log_dir / _WRITER_OWNER_FILENAME
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FileExistsError(
                f"Chunk log directory is not owned by a resumable chunk run: {log_dir}. "
                "Choose a different --run-name or remove the unrelated directory."
            ) from error
        if not isinstance(owner, dict) or owner.get("work_dir") != str(work_dir):
            raise FileExistsError(
                f"Chunk log directory belongs to a different chunk work directory: {log_dir}. "
                "Choose a different --run-name."
            )

    def get_crop_bboxes(self, sfm_scene: SfmScene) -> list[tuple[float, float, float, float, float, float]]:
        """Return clipped training-halo bounds for backward compatibility."""

        return [chunk.train_bbox for chunk in self._chunk_specs_for_scene(sfm_scene)]

    def _chunk_specs_for_scene(self, sfm_scene: SfmScene) -> tuple[ChunkSpec, ...]:
        point_bbox = None
        if len(sfm_scene.points) > 0:
            point_bbox = np.concatenate((sfm_scene.points.min(axis=0), sfm_scene.points.max(axis=0)))
        domain_bbox = resolve_chunk_domain(scene_bbox=sfm_scene.scene_bbox, point_bbox=point_bbox)
        return plan_spatial_chunks(domain_bbox, self.nchunks, self.chunk_overlap_pct)

    def _merged_reconstruction_metadata(self, sfm_scene: SfmScene) -> dict[str, torch.Tensor | float | int | str]:
        """Build final metadata from the global transformed scene, not an arbitrary last chunk."""

        training_indices, _ = GaussianSplatReconstruction._make_index_splits(sfm_scene, self.use_every_n_as_val)
        training_images = [sfm_scene.images[int(index)] for index in training_indices]
        camera_models = np.asarray(
            [int(image.camera_metadata.camera_model) for image in training_images], dtype=np.int32
        )
        distortion_coeffs = np.zeros((len(training_images), 12), dtype=np.float32)
        for image_index, image in enumerate(training_images):
            image_distortion = np.asarray(image.camera_metadata.distortion_coeffs, dtype=np.float32).reshape(-1)
            if image_distortion.size > distortion_coeffs.shape[1]:
                raise ValueError(
                    f"Image {image.image_id} has {image_distortion.size} distortion coefficients; "
                    f"at most {distortion_coeffs.shape[1]} are supported"
                )
            distortion_coeffs[image_index, : image_distortion.size] = image_distortion

        camera_to_world_matrices = np.asarray(sfm_scene.camera_to_world_matrices, dtype=np.float32)[training_indices]
        projection_matrices = np.asarray(sfm_scene.projection_matrices, dtype=np.float32)[training_indices]
        image_sizes = np.asarray(sfm_scene.image_sizes, dtype=np.int32)[training_indices]
        median_depths = np.asarray(sfm_scene.median_depth_per_image, dtype=np.float32)[training_indices]

        return {
            "normalization_transform": torch.from_numpy(np.asarray(sfm_scene.transformation_matrix, dtype=np.float32)),
            "camera_to_world_matrices": torch.from_numpy(camera_to_world_matrices),
            "projection_matrices": torch.from_numpy(projection_matrices),
            "image_sizes": torch.from_numpy(image_sizes),
            "camera_models": torch.from_numpy(camera_models),
            "distortion_coeffs": torch.from_numpy(distortion_coeffs),
            "median_depths": torch.from_numpy(median_depths),
            "eps2d": self.cfg.eps_2d,
            "near_plane": self.cfg.near_plane,
            "far_plane": self.cfg.far_plane,
            "min_radius_2d": self.cfg.min_radius_2d,
            "antialias": int(self.cfg.antialias),
            "tile_size": self.cfg.tile_size,
            "projection_method": self.cfg.projection_method,
            "render_backend": self.cfg.render_backend,
        }

    def _chunk_run_signature(self, source_signature: Mapping[str, object]) -> dict[str, object]:
        """Capture every model-affecting command setting used to validate completed chunk reuse."""

        return {
            "dataset_path": str(self.dataset_path.expanduser().resolve()),
            "source": dict(source_signature),
            "nchunks": list(self.nchunks),
            "chunk_overlap_pct": self.chunk_overlap_pct,
            "use_every_n_as_val": self.use_every_n_as_val,
            "device": str(self.device),
            "reconstruction_config": repr(self.cfg),
            "optimizer_config": repr(self.opt),
            "transform_config": repr(self.tx),
            "reconstruction_version": GaussianSplatReconstruction.version,
            "implementation": _training_implementation_identity(),
            "writer": {
                "run_name": self.resolved_chunk_run_name,
                "log_path": (str(self.io.log_path.expanduser().resolve()) if self.io.log_path is not None else None),
                "config": repr(self.io),
            },
        }

    @staticmethod
    def _writer_owner_payload(manifest: ChunkRunManifest) -> dict[str, object]:
        signature_bytes = json.dumps(
            manifest.run_signature,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            "kind": "fvdb-reality-capture-chunk-writer-owner",
            "version": 1,
            "manifest_path": str(manifest.path),
            "work_dir": str(manifest.work_dir),
            "run_signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
        }

    def _make_chunk_writer(self, manifest: ChunkRunManifest) -> GaussianSplatReconstructionWriter:
        log_dir = self.resolved_chunk_log_dir
        if log_dir is None:
            return GaussianSplatReconstructionWriter(
                run_name=self.resolved_chunk_run_name,
                save_path=None,
                config=self.io,
                exist_ok=False,
            )

        expected_owner = self._writer_owner_payload(manifest)
        owner_path = log_dir / _WRITER_OWNER_FILENAME
        if log_dir.exists():
            if not log_dir.is_dir():
                raise FileExistsError(f"Chunk log path exists and is not a directory: {log_dir}")
            try:
                existing_owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise FileExistsError(
                    f"Chunk log directory is not owned by this manifest: {log_dir}. "
                    "Choose a different --run-name or remove the unrelated directory."
                ) from error
            if existing_owner != expected_owner:
                raise FileExistsError(
                    f"Chunk log directory belongs to a different chunk run: {log_dir}. Choose a different --run-name."
                )
        else:
            log_dir.mkdir(parents=True, exist_ok=False)
            try:
                with owner_path.open("x", encoding="utf-8") as owner_file:
                    json.dump(expected_owner, owner_file, indent=2, sort_keys=True, allow_nan=False)
                    owner_file.write("\n")
                    owner_file.flush()
                    os.fsync(owner_file.fileno())
                _fsync_directory(log_dir)
                _fsync_directory(log_dir.parent)
            except BaseException:
                owner_path.unlink(missing_ok=True)
                try:
                    log_dir.rmdir()
                except OSError:
                    pass
                raise

        assert self.io.log_path is not None
        return GaussianSplatReconstructionWriter(
            run_name=self.resolved_chunk_run_name,
            save_path=self.io.log_path,
            config=self.io,
            exist_ok=True,
        )

    def _count_chunk_initializations(
        self,
        chunks: tuple[ChunkSpec, ...],
        count_chunk: Callable[[ChunkSpec], int],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        over_limit: list[tuple[str, int]] = []
        for chunk in chunks:
            count = int(count_chunk(chunk))
            if count < 0:
                raise RuntimeError(f"Point counter returned a negative value for {chunk.id}: {count}")
            counts[chunk.id] = count
            self.logger.info(
                "Chunk %s grid=%s initialization=%s core=%s train=%s",
                chunk.id,
                chunk.grid_index,
                f"{count:,}",
                chunk.core_bbox,
                chunk.train_bbox,
            )
            if self.opt.max_gaussians > 0 and count > self.opt.max_gaussians:
                over_limit.append((chunk.id, count))

        self.logger.info(
            "Chunk initialization total (including halo duplication): %s; max_gaussians is per chunk%s",
            f"{sum(counts.values()):,}",
            f" ({self.opt.max_gaussians:,})" if self.opt.max_gaussians > 0 else " (unlimited)",
        )
        if over_limit:
            details = ", ".join(f"{chunk_id}={count:,}" for chunk_id, count in over_limit)
            raise ValueError(
                f"Chunk initialization exceeds the per-chunk max_gaussians={self.opt.max_gaussians:,}: {details}. "
                "Increase nchunks or max_gaussians; dense initialization points are never silently discarded."
            )
        return counts

    @staticmethod
    def _save_chunk_model(model: GaussianSplat3d, artifact_path: Path, expected_vertex_count: int) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{artifact_path.name}.", suffix=".tmp", dir=artifact_path.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            model.save_ply(temporary_path, {})
            validate_gaussian_ply_file(temporary_path, expected_vertex_count=expected_vertex_count)
            with temporary_path.open("rb") as artifact_file:
                os.fsync(artifact_file.fileno())
            os.replace(temporary_path, artifact_path)
            temporary_path = None
            _fsync_directory(artifact_path.parent)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _save_empty_chunk_marker(artifact_path: Path, chunk: ChunkSpec) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{artifact_path.name}.",
                suffix=".tmp",
                dir=artifact_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(f"{chunk.id}: empty Gaussian chunk\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, artifact_path)
            temporary_path = None
            _fsync_directory(artifact_path.parent)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _run_chunk_plan(
        self,
        chunks: tuple[ChunkSpec, ...],
        metadata_scene: SfmScene,
        source_signature: Mapping[str, object],
        count_chunk: Callable[[ChunkSpec], int],
        load_chunk: Callable[[ChunkSpec, int], SfmScene],
        viz_scene: fviz.Scene | None,
    ) -> None:
        if self.chunk_plan_only:
            self._count_chunk_initializations(chunks, count_chunk)
            return
        if self.out_path.suffix.lower() != ".ply":
            raise ValueError("Bounded-memory chunk merge currently requires a .ply output path")

        with ChunkRunLock(self.resolved_chunk_work_dir):
            self._run_locked_chunk_plan(
                chunks,
                metadata_scene=metadata_scene,
                source_signature=source_signature,
                count_chunk=count_chunk,
                load_chunk=load_chunk,
                viz_scene=viz_scene,
            )

    def _run_locked_chunk_plan(
        self,
        chunks: tuple[ChunkSpec, ...],
        metadata_scene: SfmScene,
        source_signature: Mapping[str, object],
        count_chunk: Callable[[ChunkSpec], int],
        load_chunk: Callable[[ChunkSpec, int], SfmScene],
        viz_scene: fviz.Scene | None,
    ) -> None:
        """Train and merge a chunk plan while the caller holds its work-directory lock."""

        work_dir = self.resolved_chunk_work_dir
        work_dir_existed = work_dir.exists()
        if work_dir_existed and not self.resume_chunks:
            raise FileExistsError(f"Chunk work directory already exists and resume_chunks is disabled: {work_dir}")
        fresh_counts = None
        if not work_dir_existed:
            # A bad grid or too-small per-chunk cap must not leave a manifest that
            # prevents the user from correcting nchunks/max_gaussians and retrying.
            fresh_counts = self._count_chunk_initializations(chunks, count_chunk)

        manifest = ChunkRunManifest.open_or_create(
            work_dir,
            self._chunk_run_signature(source_signature),
            chunks,
        )

        for state in manifest.states:
            if state.status != "complete" or state.final_count is None or state.final_count == 0:
                continue
            assert state.artifact_path is not None
            try:
                validate_gaussian_ply_file(
                    state.artifact_path,
                    expected_vertex_count=state.final_count,
                )
            except (OSError, RuntimeError, ValueError) as error:
                self.logger.warning(
                    "Demoting corrupt completed artifact for %s so it can be retrained: %s",
                    state.spec.id,
                    error,
                )
                manifest.mark_pending(state.spec.id)

        pending_chunks = manifest.pending_chunks
        if fresh_counts is None:
            counts = self._count_chunk_initializations(pending_chunks, count_chunk)
        else:
            counts = {chunk.id: fresh_counts[chunk.id] for chunk in pending_chunks}
        if manifest.completed_chunks:
            self.logger.info(
                "Resuming %s: reusing %d completed chunks and training %d pending chunks",
                manifest.path,
                len(manifest.completed_chunks),
                len(pending_chunks),
            )

        writer: GaussianSplatReconstructionWriter | None = None
        num_chunks = len(chunks)
        for chunk in pending_chunks:
            initial_count = counts[chunk.id]
            self.logger.info(
                "Reconstructing %s (%d/%d) with %s initialization points",
                chunk.id,
                chunk.index + 1,
                num_chunks,
                f"{initial_count:,}",
            )

            runner: GaussianSplatReconstruction | None = None
            scene_chunk: SfmScene | None = None
            try:
                if initial_count == 0:
                    artifact_path = work_dir / f"{chunk.id}.empty"
                    self._save_empty_chunk_marker(artifact_path, chunk)
                    manifest.mark_complete(chunk.id, artifact_path, initial_count=0, final_count=0)
                    self.logger.warning(
                        "Skipping %s because its training halo contains no initialization points", chunk.id
                    )
                    continue

                if writer is None:
                    writer = self._make_chunk_writer(manifest)
                scene_chunk = load_chunk(chunk, initial_count)
                if len(scene_chunk.points) != initial_count:
                    raise RuntimeError(
                        f"{chunk.id} count changed between preflight and materialization: "
                        f"expected {initial_count:,}, loaded {len(scene_chunk.points):,}"
                    )
                runner = GaussianSplatReconstruction.from_sfm_scene(
                    sfm_scene=scene_chunk,
                    config=self.cfg,
                    optimizer_config=self.opt,
                    writer=writer,
                    viz_scene=viz_scene,
                    use_every_n_as_val=self.use_every_n_as_val,
                    log_interval_steps=self.io.log_every,
                    viz_update_interval_epochs=self.update_viz_every,
                    device=self.device,
                )
                runner.optimize(True, f"reconstruct_{chunk.id}")
                final_count = runner.model.num_gaussians
                if self.opt.max_gaussians > 0 and final_count > self.opt.max_gaussians:
                    raise RuntimeError(
                        f"{chunk.id} finished with {final_count:,} Gaussians, exceeding its "
                        f"per-chunk max_gaussians={self.opt.max_gaussians:,}"
                    )

                if final_count == 0:
                    artifact_path = work_dir / f"{chunk.id}.empty"
                    self._save_empty_chunk_marker(artifact_path, chunk)
                else:
                    artifact_path = work_dir / f"{chunk.id}.ply"
                    self._save_chunk_model(runner.model, artifact_path, final_count)
                manifest.mark_complete(
                    chunk.id,
                    artifact_path,
                    initial_count=initial_count,
                    final_count=final_count,
                )
                self.logger.info("Completed %s with %s Gaussians", chunk.id, f"{final_count:,}")
            finally:
                runner = None
                scene_chunk = None
                gc.collect()
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()

        if writer is not None:
            writer.close()
        writer = None
        gc.collect()
        merge_sources = [
            GaussianPlyMergeSource(
                state.artifact_path,
                core_bbox=state.spec.core_bbox,
                inclusive_max=state.spec.inclusive_max,
            )
            for state in manifest.states
            if state.status == "complete" and state.final_count is not None and state.final_count > 0
        ]
        if not merge_sources:
            raise ValueError("All chunks completed with zero Gaussians; no merged PLY can be written")

        self.logger.info("Streaming %d completed chunk PLYs into %s", len(merge_sources), self.out_path)
        merge_result = merge_gaussian_ply_files(
            merge_sources,
            self.out_path,
            metadata=self._merged_reconstruction_metadata(metadata_scene),
        )
        self.logger.info(
            "Merged %s core-owned Gaussians from %s trained Gaussians; filtered %s halo Gaussians",
            f"{merge_result.output_gaussians:,}",
            f"{merge_result.input_gaussians:,}",
            f"{merge_result.filtered_gaussians:,}",
        )

    def _single_colmap_bbox_fast_path_incompatibilities(self) -> list[str]:
        """Return settings that would change results if points were selected before transforms."""

        incompatibilities = []
        if self.nchunks != (1, 1, 1):
            incompatibilities.append("nchunks must be (1, 1, 1)")
        if self.dataset_type != "colmap":
            incompatibilities.append("dataset_type must be 'colmap'")
        if self.tx.crop_bbox is None:
            incompatibilities.append("tx.crop_bbox must be set")
        if self.tx.normalization_type != "none":
            incompatibilities.append("tx.normalization_type must be 'none'")
        if self.tx.points_percentile_filter != 0.0:
            incompatibilities.append("tx.points_percentile_filter must be 0")
        return incompatibilities

    def _can_use_single_colmap_bbox_fast_path(self) -> tuple[bool, str]:
        incompatibilities = self._single_colmap_bbox_fast_path_incompatibilities()
        if incompatibilities:
            return False, "; ".join(incompatibilities)
        return probe_trackless_colmap_binary(self.dataset_path)

    def _load_single_colmap_bbox_scene(self) -> SfmScene | None:
        """Load only points that can survive the configured single-scene bbox crop."""

        assert self.tx.crop_bbox is not None
        # CropScene converts its bbox and the full loader's points to float32. Use the same bbox
        # representation here, then run CropScene normally below to remove any source float64
        # coordinates which round onto an open boundary during point materialization.
        crop_bbox = np.asarray(self.tx.crop_bbox, dtype=np.float32)
        source = TracklessColmapSceneSource(self.dataset_path)
        self.logger.info(
            "COLMAP bbox fast path: scanning %s trackless initialization points inside %s and validating point order",
            f"{source.point_count:,}",
            crop_bbox,
        )
        selected_count, strictly_increasing_ids = source.count_points_and_check_strict_id_order(
            crop_bbox, bounds_mode="open"
        )
        if not strictly_increasing_ids:
            self.logger.warning(
                "COLMAP bbox fast path requires points3D.bin records in strictly increasing point-ID order to "
                "match the standard loader; falling back to the standard dataset loader"
            )
            return None
        self.logger.info(
            "COLMAP bbox fast path: selected %s of %s initialization points; materializing selected columns",
            f"{selected_count:,}",
            f"{source.point_count:,}",
        )
        sfm_scene = source.scene_for_bbox(
            crop_bbox,
            bounds_mode="open",
            expected_selected_count=selected_count,
        )
        self.logger.info(
            "COLMAP bbox fast path: loaded %s of %s initialization points",
            f"{len(sfm_scene.points):,}",
            f"{source.point_count:,}",
        )
        return sfm_scene

    def _partial_colmap_incompatibilities(self) -> list[str]:
        incompatibilities = []
        if self.dataset_type != "colmap":
            incompatibilities.append("dataset_type must be 'colmap'")
        if self.tx.normalization_type != "none":
            incompatibilities.append("tx.normalization_type must be 'none'")
        if self.tx.points_percentile_filter != 0.0:
            incompatibilities.append("tx.points_percentile_filter must be 0")
        if self.tx.crop_to_points:
            incompatibilities.append("tx.crop_to_points must be disabled")
        if self.tx.min_points_per_image >= 0:
            incompatibilities.append(
                "tx.min_points_per_image must be negative because trackless points have no per-image observations"
            )
        if self.opt.spatial_scale_mode != SpatialScaleMode.ABSOLUTE_UNITS:
            incompatibilities.append("opt.spatial_scale_mode must be ABSOLUTE_UNITS")
        return incompatibilities

    def _can_use_partial_colmap(self) -> tuple[bool, str]:
        if self.chunked_colmap_load == "off":
            return False, "disabled by chunked_colmap_load=off"
        incompatibilities = self._partial_colmap_incompatibilities()
        if incompatibilities:
            return False, "; ".join(incompatibilities)
        supported, reason = probe_trackless_colmap_binary(self.dataset_path)
        return supported, reason

    def _run_partial_colmap_reconstruction(self, viz_scene: fviz.Scene | None) -> None:
        """Train chunked trackless binary COLMAP without ever materializing the global point model."""

        source = TracklessColmapSceneSource(self.dataset_path)
        if self.tx.crop_bbox is not None:
            domain_bbox = resolve_chunk_domain(scene_bbox=self.tx.crop_bbox)
        else:
            point_bbox = source.bounds()
            if point_bbox is None:
                raise ValueError("The COLMAP model contains no points and no explicit crop_bbox was provided")
            domain_bbox = resolve_chunk_domain(point_bbox=point_bbox)
        chunks = plan_spatial_chunks(domain_bbox, self.nchunks, self.chunk_overlap_pct)

        def count_chunk(chunk: ChunkSpec) -> int:
            # Closed training bounds avoid gaps when overlap is zero; the final half-open core merge
            # still assigns every retained Gaussian center to exactly one chunk.
            return source.count_points(np.asarray(chunk.train_bbox), bounds_mode="closed")

        if self.chunk_plan_only:
            self._count_chunk_initializations(chunks, count_chunk)
            return

        source_metadata_scene = source.metadata_scene(np.asarray(domain_bbox))
        source_scene_fingerprint = _scene_resume_fingerprint(source_metadata_scene, include_points=False)
        metadata_scene = DownsampleImages(
            image_downsample_factor=self.tx.image_downsample_factor,
            rescaled_jpeg_quality=self.tx.rescale_jpeg_quality,
        )(source_metadata_scene)
        transformed_scene_fingerprint = _scene_resume_fingerprint(
            metadata_scene, include_points=False, include_image_contents=True
        )

        def load_chunk(chunk: ChunkSpec, expected_count: int) -> SfmScene:
            # Generate/cache per-image training masks on a zero-point scene, then attach only the
            # spatially selected point columns. This avoids a second 65M-point scene and avoids
            # copying a chunk's point arrays merely to set its bbox/masks.
            chunk_metadata = CropScene(bbox=chunk.train_bbox)(metadata_scene)
            selected_scene = source.scene_for_bbox(
                np.asarray(chunk.train_bbox),
                bounds_mode="closed",
                expected_selected_count=expected_count,
            )
            return chunk_metadata.replace(
                points=selected_scene.points,
                points_err=selected_scene.points_err,
                points_rgb=selected_scene.points_rgb,
                scene_bbox=np.asarray(chunk.train_bbox, dtype=np.float32),
            )

        source_signature: dict[str, object] = {
            "loader": "trackless_binary_colmap_partial",
            "point_source_fingerprint": source.fingerprint,
            "point_source_full_sha256": source.full_fingerprint,
            "point_count": source.point_count,
            "bounds_mode": "closed",
            "scene_input": source_scene_fingerprint,
            "transformed_scene": transformed_scene_fingerprint,
        }
        self.logger.info(
            "Using bounded-memory COLMAP initialization for %s points; no full point model will be loaded",
            f"{source.point_count:,}",
        )
        self._run_chunk_plan(
            chunks,
            metadata_scene=metadata_scene,
            source_signature=source_signature,
            count_chunk=count_chunk,
            load_chunk=load_chunk,
            viz_scene=viz_scene,
        )

    def _run_chunked_reconstruction(
        self,
        sfm_scene: SfmScene,
        viz_scene: fviz.Scene | None,
    ) -> None:
        """Chunk an already materialized/transformed scene using the scalable artifact pipeline."""

        chunks = self._chunk_specs_for_scene(sfm_scene)

        def closed_train_bbox(chunk: ChunkSpec) -> tuple[float, float, float, float, float, float]:
            bbox = np.asarray(chunk.train_bbox, dtype=np.float32).copy()
            bbox[:3] = np.nextafter(bbox[:3], np.float32(-np.inf))
            bbox[3:] = np.nextafter(bbox[3:], np.float32(np.inf))
            return tuple(float(value) for value in bbox)  # type: ignore[return-value]

        def point_mask(chunk: ChunkSpec) -> np.ndarray:
            bbox = np.asarray(closed_train_bbox(chunk), dtype=np.float32)
            mask = np.ones((len(sfm_scene.points),), dtype=np.bool_)
            for axis in range(3):
                mask &= sfm_scene.points[:, axis] > bbox[axis]
                mask &= sfm_scene.points[:, axis] < bbox[axis + 3]
            return mask

        def count_chunk(chunk: ChunkSpec) -> int:
            return int(np.count_nonzero(point_mask(chunk)))

        def load_chunk(chunk: ChunkSpec, expected_count: int) -> SfmScene:
            del expected_count
            return CropScene(bbox=closed_train_bbox(chunk))(sfm_scene)

        if self.chunk_plan_only:
            self._count_chunk_initializations(chunks, count_chunk)
            return

        source_signature: dict[str, object] = {
            "loader": "materialized",
            "point_count": len(sfm_scene.points),
            "bounds_mode": "closed_via_float32_nextafter",
            "scene_input": _scene_resume_fingerprint(sfm_scene, include_points=True, include_image_contents=True),
        }
        self._run_chunk_plan(
            chunks,
            metadata_scene=sfm_scene,
            source_signature=source_signature,
            count_chunk=count_chunk,
            load_chunk=load_chunk,
            viz_scene=viz_scene,
        )

    def _run_single_reconstruction(
        self,
        sfm_scene: SfmScene,
        writer: GaussianSplatReconstructionWriter,
        viz_scene: fviz.Scene | None,
    ):
        """
        Reconstruct a single scene and save as a PLY or USDZ file.

        Args:
            sfm_scene (SfmScene): The SfM scene to reconstruct.
            writer (GaussianSplatReconstructionWriter): Writer to use for logging and saving metrics.
            viz_scene (fviz.Scene | None): :class:`fviz.Scene` to use for visualization. If ``None``, no visualization will be done.
        """
        runner = GaussianSplatReconstruction.from_sfm_scene(
            sfm_scene,
            config=self.cfg,
            optimizer_config=self.opt,
            writer=writer,
            viz_scene=viz_scene,
            use_every_n_as_val=self.use_every_n_as_val,
            log_interval_steps=self.io.log_every,
            viz_update_interval_epochs=self.update_viz_every,
            device=self.device,
        )

        runner.optimize()

        self.logger.info(f"Saving final model to {self.out_path}")
        save_model_from_runner(self.out_path, runner)

    def execute(self) -> None:
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        self.logger = logging.getLogger(__name__)

        if self.out_path.suffix.lower() not in [".ply", ".usdc", ".usdz"]:
            raise ValueError("Output path must end in .ply, .usdc, or .usdz")
        if self.out_path.exists() and not self.chunk_plan_only:
            raise ValueError(f"Output path {self.out_path} already exists")
        if len(self.nchunks) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.nchunks
        ):
            raise ValueError("nchunks must be a tuple of 3 positive integers")
        if not np.isfinite(self.chunk_overlap_pct) or not 0.0 <= self.chunk_overlap_pct < 1.0:
            raise ValueError("chunk_overlap_pct must be finite and in the range [0, 1)")
        if self.chunked_colmap_load not in ("auto", "require", "off"):
            raise ValueError("chunked_colmap_load must be one of: auto, require, off")

        chunked = self.nchunks != (1, 1, 1)
        if self.chunk_plan_only and not chunked:
            raise ValueError("chunk_plan_only requires nchunks other than (1, 1, 1)")
        if chunked and self.cfg.optimize_camera_poses:
            raise ValueError(
                "Chunked reconstruction requires fixed camera poses; independently optimized chunk poses "
                "cannot be merged into one consistent coordinate system"
            )
        if chunked and not self.chunk_plan_only and self.out_path.suffix.lower() != ".ply":
            raise ValueError("Chunked reconstruction currently supports bounded-memory merge only to .ply")
        if chunked and not self.chunk_plan_only:
            self._validate_chunk_paths()
            if not self.cfg.remove_gaussians_outside_scene_bbox:
                self.logger.warning("Enabling remove_gaussians_outside_scene_bbox for bounded chunk training")
                self.cfg.remove_gaussians_outside_scene_bbox = True
            if self.cfg.save_at_percent and (self.io.save_checkpoints or self.io.save_plys):
                self.logger.warning(
                    "Chunk checkpoints and intermediate PLYs can consume hundreds of gigabytes at this scale, "
                    "and completed-chunk resume does not restore mid-chunk checkpoints. Use "
                    "--io.no-save-checkpoints --io.no-save-plys when only final chunk artifacts and merge output "
                    "are required."
                )

        if self.update_viz_every > 0 and not self.chunk_plan_only:
            self.logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
            fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
            viz_scene = fviz.get_scene("Gaussian Splat Reconstruction Visualization")
        else:
            viz_scene = None

        if chunked:
            use_partial_colmap, partial_reason = self._can_use_partial_colmap()
            if use_partial_colmap:
                self.logger.info("Partial COLMAP fast path selected: %s", partial_reason)
                self._run_partial_colmap_reconstruction(viz_scene)
                return
            if self.chunked_colmap_load == "require":
                raise ValueError(
                    f"Bounded-memory COLMAP chunk loading was required but is unavailable: {partial_reason}"
                )
            if self.dataset_type == "colmap" and self.chunked_colmap_load == "auto":
                self.logger.warning(
                    "Bounded-memory COLMAP chunk loading is unavailable (%s); falling back to full model loading",
                    partial_reason,
                )

        sfm_scene: SfmScene | None = None
        if not chunked:
            use_single_colmap_bbox, single_colmap_bbox_reason = self._can_use_single_colmap_bbox_fast_path()
            if use_single_colmap_bbox:
                self.logger.info("Single-scene COLMAP bbox fast path selected: %s", single_colmap_bbox_reason)
                sfm_scene = self._load_single_colmap_bbox_scene()
            else:
                self.logger.debug(
                    "Single-scene COLMAP bbox fast path unavailable (%s); using the standard dataset loader",
                    single_colmap_bbox_reason,
                )

        if sfm_scene is None:
            self.logger.info("Loading dataset from %s with the standard dataset loader", self.dataset_path)
            sfm_scene = load_sfm_scene(self.dataset_path, self.dataset_type)

        self.logger.info(
            "Applying configured scene transforms to %s initialization points", f"{len(sfm_scene.points):,}"
        )
        scene_transform: BaseTransform = self.tx.scene_transform
        sfm_scene = scene_transform(sfm_scene)
        self.logger.info(
            "Scene transforms complete: %s initialization points across %s images",
            f"{len(sfm_scene.points):,}",
            f"{len(sfm_scene.images):,}",
        )

        if not chunked:
            writer = GaussianSplatReconstructionWriter(
                run_name=self.run_name,
                save_path=self.io.log_path,
                config=self.io,
                exist_ok=False,
            )
            self._run_single_reconstruction(sfm_scene, writer, viz_scene)
        else:
            self._run_chunked_reconstruction(sfm_scene, viz_scene)


@dataclass
class ReconstructMCMC(Reconstruct):
    """
    Reconstruct a Gaussian Splat Radiance Field using the MCMC optimizer strategy.

    This command is identical to :class:`Reconstruct`, but exposes the
    :class:`~fvdb_reality_capture.radiance_fields.GaussianSplatOptimizerMCMCConfig`
    configuration under ``--opt.*``.
    """

    opt: GaussianSplatOptimizerMCMCConfig = field(default_factory=GaussianSplatOptimizerMCMCConfig)
