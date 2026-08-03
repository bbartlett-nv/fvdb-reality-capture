# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import logging
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from fvdb_reality_capture import CameraModel
from fvdb_reality_capture.chunk_manifest import ChunkRunLock, ChunkRunLockError
from fvdb_reality_capture.cli.frgs._reconstruct import Reconstruct, _scene_image_contents_sha256
from fvdb_reality_capture.sfm_scene import (
    SfmCameraMetadata,
    SfmPosedImageMetadata,
    SfmScene,
    SpatialScaleMode,
)
from fvdb_reality_capture.spatial_chunking import plan_spatial_chunks


_RECONSTRUCT_MODULE = "fvdb_reality_capture.cli.frgs._reconstruct"


class ReconstructChunkOrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _command(self, **kwargs) -> Reconstruct:
        command = Reconstruct(
            dataset_path=self.root / "dataset",
            out_path=self.root / "merged.ply",
            **kwargs,
        )
        command.io.log_path = self.root / "logs"
        command.logger = mock.Mock(spec=logging.Logger)
        return command

    def _compatible_partial_command(self, **kwargs) -> Reconstruct:
        command = self._command(**kwargs)
        command.dataset_type = "colmap"
        command.tx.normalization_type = "none"
        command.tx.points_percentile_filter = 0.0
        command.tx.crop_to_points = False
        command.tx.min_points_per_image = -1
        command.opt.spatial_scale_mode = SpatialScaleMode.ABSOLUTE_UNITS
        return command

    def _compatible_single_bbox_command(self, **kwargs) -> Reconstruct:
        command = self._compatible_partial_command(**kwargs)
        command.tx.crop_bbox = (-4.0, -3.0, -2.0, 4.0, 3.0, 2.0)
        return command

    def _metadata_scene(self) -> SfmScene:
        pinhole = SfmCameraMetadata(
            img_width=640,
            img_height=480,
            fx=500.0,
            fy=501.0,
            cx=320.0,
            cy=240.0,
            camera_model=CameraModel.PINHOLE,
            distortion_coeffs=np.empty((0,), dtype=np.float32),
        )
        distortion = np.linspace(0.01, 0.12, 12, dtype=np.float32)
        opencv = SfmCameraMetadata(
            img_width=800,
            img_height=600,
            fx=700.0,
            fy=701.0,
            cx=400.0,
            cy=300.0,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=distortion,
        )
        camera_to_world_0 = np.eye(4, dtype=np.float64)
        camera_to_world_1 = np.eye(4, dtype=np.float64)
        camera_to_world_1[:3, 3] = [10.0, 20.0, 30.0]
        images = [
            SfmPosedImageMetadata(
                world_to_camera_matrix=np.linalg.inv(camera_to_world_0),
                camera_to_world_matrix=camera_to_world_0,
                camera_metadata=pinhole,
                camera_id=1,
                image_path="first.png",
                mask_path="",
                point_indices=np.array([0, 1], dtype=np.int64),
                image_id=10,
            ),
            SfmPosedImageMetadata(
                world_to_camera_matrix=np.linalg.inv(camera_to_world_1),
                camera_to_world_matrix=camera_to_world_1,
                camera_metadata=opencv,
                camera_id=2,
                image_path="second.png",
                mask_path="",
                point_indices=np.array([1, 2], dtype=np.int64),
                image_id=20,
            ),
        ]
        return SfmScene(
            cameras={1: pinhole, 2: opencv},
            images=images,
            points=np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]], dtype=np.float32),
            points_err=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            points_rgb=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
            scene_bbox=np.array([-5.0, -6.0, -7.0, 15.0, 16.0, 17.0], dtype=np.float32),
            transformation_matrix=np.arange(16, dtype=np.float64).reshape(4, 4),
            cache=mock.Mock(),
        )

    def test_explicit_scene_bbox_takes_precedence_over_point_bounds(self):
        command = self._command(nchunks=(2, 1, 1), chunk_overlap_pct=0.0)
        scene = SimpleNamespace(
            points=np.array([[40.0, 5.0, 2.0], [60.0, 15.0, 8.0]], dtype=np.float32),
            scene_bbox=np.array([0.0, 0.0, 0.0, 100.0, 20.0, 10.0], dtype=np.float32),
        )

        chunks = command._chunk_specs_for_scene(scene)

        self.assertEqual(chunks[0].core_bbox, (0.0, 0.0, 0.0, 50.0, 20.0, 10.0))
        self.assertEqual(chunks[1].core_bbox, (50.0, 0.0, 0.0, 100.0, 20.0, 10.0))
        self.assertEqual(tuple(chunk.train_bbox for chunk in chunks), tuple(chunk.core_bbox for chunk in chunks))

    def test_max_gaussians_preflight_is_applied_independently_per_chunk(self):
        command = self._command()
        command.opt.max_gaussians = 10
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 2.0, 1.0, 1.0), (2, 1, 1), 0.0)

        counts = command._count_chunk_initializations(chunks, mock.Mock(side_effect=[8, 8]))

        self.assertEqual(counts, {"chunk_0000": 8, "chunk_0001": 8})
        over_limit_counter = mock.Mock(side_effect=[11, 12])
        with self.assertRaisesRegex(
            ValueError,
            r"per-chunk max_gaussians=10: chunk_0000=11, chunk_0001=12",
        ):
            command._count_chunk_initializations(chunks, over_limit_counter)
        self.assertEqual(over_limit_counter.call_count, 2)

    def test_merged_metadata_uses_all_global_scene_images_with_stable_shapes(self):
        command = self._command()
        scene = self._metadata_scene()

        metadata = command._merged_reconstruction_metadata(scene)

        self.assertEqual(tuple(metadata["normalization_transform"].shape), (4, 4))
        self.assertEqual(tuple(metadata["camera_to_world_matrices"].shape), (2, 4, 4))
        self.assertEqual(tuple(metadata["projection_matrices"].shape), (2, 3, 3))
        self.assertEqual(tuple(metadata["image_sizes"].shape), (2, 2))
        self.assertEqual(tuple(metadata["camera_models"].shape), (2,))
        self.assertEqual(tuple(metadata["distortion_coeffs"].shape), (2, 12))
        self.assertEqual(tuple(metadata["median_depths"].shape), (2,))
        torch.testing.assert_close(
            metadata["camera_to_world_matrices"],
            torch.from_numpy(scene.camera_to_world_matrices.astype(np.float32)),
        )
        torch.testing.assert_close(
            metadata["camera_models"],
            torch.tensor([int(CameraModel.PINHOLE), int(CameraModel.OPENCV_RADTAN_5)], dtype=torch.int32),
        )
        expected_distortion = torch.from_numpy(
            np.stack(
                [
                    np.zeros((12,), dtype=np.float32),
                    scene.images[1].camera_metadata.distortion_coeffs,
                ]
            )
        )
        torch.testing.assert_close(metadata["distortion_coeffs"], expected_distortion)
        torch.testing.assert_close(
            metadata["median_depths"],
            torch.from_numpy(scene.median_depth_per_image.astype(np.float32)),
        )

    def test_merged_metadata_excludes_validation_images(self):
        command = self._command()
        command.use_every_n_as_val = 2
        scene = self._metadata_scene()

        metadata = command._merged_reconstruction_metadata(scene)

        self.assertEqual(tuple(metadata["camera_to_world_matrices"].shape), (1, 4, 4))
        torch.testing.assert_close(
            metadata["camera_to_world_matrices"][0],
            torch.from_numpy(scene.camera_to_world_matrices[1].astype(np.float32)),
        )
        torch.testing.assert_close(
            metadata["camera_models"],
            torch.tensor([int(CameraModel.OPENCV_RADTAN_5)], dtype=torch.int32),
        )

    def test_failed_fresh_preflight_does_not_create_manifest_or_work_directory(self):
        command = self._command()
        command.opt.max_gaussians = 1
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 2.0, 1.0, 1.0), (2, 1, 1), 0.0)

        with mock.patch(f"{_RECONSTRUCT_MODULE}.ChunkRunManifest.open_or_create") as manifest_open:
            with self.assertRaisesRegex(ValueError, "per-chunk max_gaussians=1"):
                command._run_chunk_plan(
                    chunks,
                    metadata_scene=object(),
                    source_signature={"loader": "synthetic"},
                    count_chunk=lambda chunk: 2,
                    load_chunk=mock.Mock(),
                    viz_scene=None,
                )

        manifest_open.assert_not_called()
        self.assertFalse(command.resolved_chunk_work_dir.exists())
        self.assertFalse(ChunkRunLock(command.resolved_chunk_work_dir).path.exists())

    def test_training_signature_allows_a_new_merge_destination(self):
        command = self._command(
            run_name="stable-run",
            chunk_work_dir=self.root / "persistent-chunks",
        )
        with mock.patch(f"{_RECONSTRUCT_MODULE}._training_implementation_identity", return_value={"id": 1}):
            first_signature = command._chunk_run_signature({"source": "same"})
            command.out_path = self.root / "other-filesystem.ply"
            second_signature = command._chunk_run_signature({"source": "same"})

        self.assertEqual(first_signature, second_signature)

    def test_training_image_fingerprint_detects_same_size_content_changes(self):
        image_path = self.root / "cached.jpg"
        image_path.write_bytes(b"first")
        scene = SimpleNamespace(images=(SimpleNamespace(image_id=3, image_path=str(image_path)),))
        first_fingerprint = _scene_image_contents_sha256(scene)
        image_path.write_bytes(b"other")
        second_fingerprint = _scene_image_contents_sha256(scene)
        self.assertNotEqual(first_fingerprint, second_fingerprint)

    def test_chunk_path_preflight_rejects_conflicts_and_unowned_logs(self):
        command = self._command(chunk_work_dir=self.root / "merged.ply")
        with self.assertRaisesRegex(ValueError, "out_path and chunk_work_dir"):
            command._validate_chunk_paths()

        missing_parent = self._command(chunk_work_dir=self.root / "safe-work")
        missing_parent.out_path = self.root / "missing" / "merged.ply"
        with self.assertRaisesRegex(FileNotFoundError, "output directory"):
            missing_parent._validate_chunk_paths()

        unrelated_log = self._command(run_name="occupied")
        assert unrelated_log.io.log_path is not None
        unrelated_log.resolved_chunk_log_dir.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "not owned"):
            unrelated_log._validate_chunk_paths()

    def test_zero_overlap_fallback_includes_internal_split_plane_points(self):
        command = self._command(nchunks=(2, 1, 1), chunk_overlap_pct=0.0, chunk_plan_only=True)
        scene = SimpleNamespace(
            points=np.asarray([[1.0, 0.5, 0.5]], dtype=np.float32),
            scene_bbox=np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 1.0], dtype=np.float32),
        )
        observed_counts = []

        def collect_counts(chunks, count_chunk):
            observed_counts.extend(count_chunk(chunk) for chunk in chunks)
            return {chunk.id: observed_counts[index] for index, chunk in enumerate(chunks)}

        with mock.patch.object(command, "_count_chunk_initializations", side_effect=collect_counts):
            command._run_chunked_reconstruction(scene, viz_scene=None)

        self.assertEqual(observed_counts, [1, 1])

    def test_merge_receives_metadata_from_global_scene(self):
        command = self._command()
        command.resolved_chunk_work_dir.mkdir()
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1, 1, 1), 0.0)
        global_scene = object()
        global_metadata = {"source": "global-scene"}
        state = SimpleNamespace(
            spec=chunks[0],
            status="complete",
            artifact_path=self.root / "chunk_0000.ply",
            initial_count=4,
            final_count=3,
        )
        manifest = SimpleNamespace(
            path=self.root / "manifest.json",
            pending_chunks=(),
            completed_chunks=chunks,
            states=(state,),
        )
        merge_result = SimpleNamespace(input_gaussians=3, output_gaussians=3, filtered_gaussians=0)
        count_chunk = mock.Mock(side_effect=AssertionError("completed chunks must not be recounted"))
        load_chunk = mock.Mock(side_effect=AssertionError("completed chunks must not be loaded"))

        with (
            mock.patch(f"{_RECONSTRUCT_MODULE}.ChunkRunManifest.open_or_create", return_value=manifest),
            mock.patch(f"{_RECONSTRUCT_MODULE}.GaussianSplatReconstructionWriter"),
            mock.patch(f"{_RECONSTRUCT_MODULE}.validate_gaussian_ply_file", return_value=3),
            mock.patch(f"{_RECONSTRUCT_MODULE}.merge_gaussian_ply_files", return_value=merge_result) as merge,
            mock.patch.object(command, "_merged_reconstruction_metadata", return_value=global_metadata) as metadata,
        ):
            command._run_chunk_plan(
                chunks,
                metadata_scene=global_scene,
                source_signature={"loader": "synthetic"},
                count_chunk=count_chunk,
                load_chunk=load_chunk,
                viz_scene=None,
            )

        metadata.assert_called_once_with(global_scene)
        self.assertIs(merge.call_args.kwargs["metadata"], global_metadata)
        self.assertEqual(merge.call_args.args[1], command.out_path)
        self.assertEqual(merge.call_args.args[0][0].core_bbox, chunks[0].core_bbox)
        count_chunk.assert_not_called()
        load_chunk.assert_not_called()

    def test_work_directory_lock_spans_manifest_training_and_merge_then_releases(self):
        command = self._command()
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1, 1, 1), 0.0)
        chunk = chunks[0]
        lock_path = ChunkRunLock(command.resolved_chunk_work_dir).path
        state = SimpleNamespace(
            spec=chunk,
            status="complete",
            artifact_path=command.resolved_chunk_work_dir / "chunk_0000.ply",
            initial_count=1,
            final_count=1,
        )
        manifest = SimpleNamespace(
            path=command.resolved_chunk_work_dir / "chunk_manifest.json",
            pending_chunks=chunks,
            completed_chunks=(),
            states=(state,),
            mark_complete=mock.Mock(),
        )
        runner = SimpleNamespace(
            model=SimpleNamespace(num_gaussians=1),
            optimize=mock.Mock(side_effect=lambda *args: self.assertTrue(lock_path.is_file())),
        )
        scene_chunk = SimpleNamespace(points=np.zeros((1, 3), dtype=np.float32))
        writer = mock.Mock()

        def open_manifest(*args, **kwargs):
            del args, kwargs
            self.assertTrue(lock_path.is_file())
            return manifest

        def save_model(model, artifact_path, expected_vertex_count):
            del model
            self.assertTrue(lock_path.is_file())
            self.assertEqual(artifact_path, state.artifact_path)
            self.assertEqual(expected_vertex_count, 1)

        def fail_merge(*args, **kwargs):
            del args, kwargs
            self.assertTrue(lock_path.is_file())
            raise RuntimeError("merge failed")

        with (
            mock.patch(
                f"{_RECONSTRUCT_MODULE}.ChunkRunManifest.open_or_create",
                side_effect=open_manifest,
            ) as manifest_open,
            mock.patch(f"{_RECONSTRUCT_MODULE}.validate_gaussian_ply_file", return_value=1),
            mock.patch.object(command, "_make_chunk_writer", return_value=writer),
            mock.patch(
                f"{_RECONSTRUCT_MODULE}.GaussianSplatReconstruction.from_sfm_scene",
                return_value=runner,
            ),
            mock.patch(f"{_RECONSTRUCT_MODULE}.merge_gaussian_ply_files", side_effect=fail_merge),
            mock.patch.object(command, "_save_chunk_model", side_effect=save_model),
            mock.patch.object(command, "_merged_reconstruction_metadata", return_value={}),
        ):
            with self.assertRaisesRegex(RuntimeError, "merge failed"):
                command._run_chunk_plan(
                    chunks,
                    metadata_scene=object(),
                    source_signature={"loader": "synthetic"},
                    count_chunk=lambda selected_chunk: 1,
                    load_chunk=lambda selected_chunk, expected_count: scene_chunk,
                    viz_scene=None,
                )

        manifest_open.assert_called_once()
        runner.optimize.assert_called_once_with(True, "reconstruct_chunk_0000")
        manifest.mark_complete.assert_called_once_with(
            "chunk_0000",
            state.artifact_path,
            initial_count=1,
            final_count=1,
        )
        self.assertFalse(lock_path.exists())

    def test_concurrent_work_directory_lock_fails_before_manifest_open(self):
        command = self._command()
        chunks = plan_spatial_chunks((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1, 1, 1), 0.0)
        active_lock = ChunkRunLock(command.resolved_chunk_work_dir)

        with active_lock:
            with mock.patch(f"{_RECONSTRUCT_MODULE}.ChunkRunManifest.open_or_create") as manifest_open:
                with self.assertRaises(ChunkRunLockError) as raised:
                    command._run_chunk_plan(
                        chunks,
                        metadata_scene=object(),
                        source_signature={"loader": "synthetic"},
                        count_chunk=mock.Mock(),
                        load_chunk=mock.Mock(),
                        viz_scene=None,
                    )
            manifest_open.assert_not_called()
            self.assertTrue(active_lock.path.is_file())
            self.assertIn("Another reconstruction may still be active", str(raised.exception))

        self.assertFalse(active_lock.path.exists())

    def test_partial_colmap_incompatibilities_prevent_probe(self):
        cases = (
            ("dataset type", lambda command: setattr(command, "dataset_type", "e57"), "dataset_type"),
            (
                "normalization",
                lambda command: setattr(command.tx, "normalization_type", "pca"),
                "normalization_type",
            ),
            (
                "point filter",
                lambda command: setattr(command.tx, "points_percentile_filter", 1.0),
                "points_percentile_filter",
            ),
            (
                "minimum points per image",
                lambda command: setattr(command.tx, "min_points_per_image", 0),
                "min_points_per_image",
            ),
            ("crop to points", lambda command: setattr(command.tx, "crop_to_points", True), "crop_to_points"),
            (
                "spatial scale",
                lambda command: setattr(command.opt, "spatial_scale_mode", SpatialScaleMode.MEDIAN_CAMERA_DEPTH),
                "spatial_scale_mode",
            ),
        )
        for label, make_incompatible, expected_reason in cases:
            with self.subTest(label=label):
                command = self._compatible_partial_command()
                make_incompatible(command)
                with mock.patch(f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary") as probe:
                    supported, reason = command._can_use_partial_colmap()
                self.assertFalse(supported)
                self.assertIn(expected_reason, reason)
                probe.assert_not_called()

    def test_partial_colmap_mode_off_bypasses_probe_and_compatible_mode_uses_it(self):
        disabled = self._compatible_partial_command(chunked_colmap_load="off")
        with mock.patch(f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary") as probe:
            self.assertEqual(disabled._can_use_partial_colmap(), (False, "disabled by chunked_colmap_load=off"))
        probe.assert_not_called()

        enabled = self._compatible_partial_command(chunked_colmap_load="auto")
        with mock.patch(
            f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary",
            return_value=(True, "trackless binary COLMAP"),
        ) as probe:
            self.assertEqual(enabled._can_use_partial_colmap(), (True, "trackless binary COLMAP"))
        probe.assert_called_once_with(enabled.dataset_path)

    def test_required_partial_mode_rejects_incompatibility_before_full_load(self):
        command = self._compatible_partial_command(
            nchunks=(2, 1, 1),
            chunked_colmap_load="require",
        )
        command.cfg.optimize_camera_poses = False
        command.tx.normalization_type = "pca"

        with mock.patch(f"{_RECONSTRUCT_MODULE}.load_sfm_scene") as full_loader:
            with self.assertRaisesRegex(ValueError, "required.*normalization_type"):
                command.execute()

        full_loader.assert_not_called()
        self.assertFalse(command.out_path.exists())
        self.assertFalse(command.resolved_chunk_work_dir.exists())

    def test_plan_only_counts_partial_chunks_without_metadata_or_artifacts(self):
        command = self._compatible_partial_command(
            nchunks=(2, 1, 1),
            chunk_overlap_pct=0.0,
            chunk_plan_only=True,
        )
        command.tx.crop_bbox = (0.0, 0.0, 0.0, 4.0, 2.0, 2.0)
        source = mock.Mock()
        source.count_points.side_effect = [3, 4]

        with (
            mock.patch(f"{_RECONSTRUCT_MODULE}.TracklessColmapSceneSource", return_value=source),
            mock.patch(f"{_RECONSTRUCT_MODULE}.ChunkRunManifest.open_or_create") as open_manifest,
            mock.patch(f"{_RECONSTRUCT_MODULE}.GaussianSplatReconstructionWriter") as writer,
            mock.patch(f"{_RECONSTRUCT_MODULE}.merge_gaussian_ply_files") as merge,
            mock.patch.object(command, "_save_empty_chunk_marker") as save_empty,
            mock.patch.object(command, "_save_chunk_model") as save_model,
        ):
            command._run_partial_colmap_reconstruction(viz_scene=None)

        source.bounds.assert_not_called()
        source.metadata_scene.assert_not_called()
        source.scene_for_bbox.assert_not_called()
        self.assertEqual(source.count_points.call_count, 2)
        first_bbox = source.count_points.call_args_list[0].args[0]
        second_bbox = source.count_points.call_args_list[1].args[0]
        np.testing.assert_allclose(first_bbox, [0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
        np.testing.assert_allclose(second_bbox, [2.0, 0.0, 0.0, 4.0, 2.0, 2.0])
        self.assertEqual(source.count_points.call_args_list[0].kwargs, {"bounds_mode": "closed"})
        self.assertEqual(source.count_points.call_args_list[1].kwargs, {"bounds_mode": "closed"})
        open_manifest.assert_not_called()
        writer.assert_not_called()
        merge.assert_not_called()
        save_empty.assert_not_called()
        save_model.assert_not_called()
        self.assertFalse(command.out_path.exists())
        self.assertFalse(command.resolved_chunk_work_dir.exists())

    def test_single_colmap_bbox_fast_path_checks_only_global_pre_crop_semantics(self):
        command = self._compatible_single_bbox_command()
        command.tx.crop_to_points = True
        command.tx.min_points_per_image = 100
        command.opt.spatial_scale_mode = SpatialScaleMode.MAX_CAMERA_TO_CENTROID

        with mock.patch(
            f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary",
            return_value=(True, "trackless binary COLMAP"),
        ) as probe:
            self.assertEqual(command._can_use_single_colmap_bbox_fast_path(), (True, "trackless binary COLMAP"))
        probe.assert_called_once_with(command.dataset_path)

        cases = (
            ("chunking", lambda candidate: setattr(candidate, "nchunks", (2, 1, 1)), "nchunks"),
            ("dataset type", lambda candidate: setattr(candidate, "dataset_type", "e57"), "dataset_type"),
            ("crop bbox", lambda candidate: setattr(candidate.tx, "crop_bbox", None), "crop_bbox"),
            (
                "normalization",
                lambda candidate: setattr(candidate.tx, "normalization_type", "pca"),
                "normalization_type",
            ),
            (
                "point filter",
                lambda candidate: setattr(candidate.tx, "points_percentile_filter", 1.0),
                "points_percentile_filter",
            ),
        )
        for label, make_incompatible, expected_reason in cases:
            with self.subTest(label=label):
                candidate = self._compatible_single_bbox_command()
                make_incompatible(candidate)
                with mock.patch(f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary") as candidate_probe:
                    supported, reason = candidate._can_use_single_colmap_bbox_fast_path()
                self.assertFalse(supported)
                self.assertIn(expected_reason, reason)
                candidate_probe.assert_not_called()

    def test_single_colmap_bbox_fast_path_selects_open_points_then_runs_regular_transforms(self):
        command = self._compatible_single_bbox_command()
        partial_scene = SimpleNamespace(
            points=np.zeros((7, 3), dtype=np.float32),
            images=(object(), object(), object()),
        )
        transformed_scene = SimpleNamespace(
            points=np.zeros((5, 3), dtype=np.float32),
            images=(object(), object()),
        )
        source = mock.Mock()
        source.point_count = 12_345
        source.count_points_and_check_strict_id_order.return_value = (7, True)
        source.scene_for_bbox.return_value = partial_scene
        scene_transform = mock.Mock(return_value=transformed_scene)
        writer = mock.Mock()

        with (
            mock.patch(
                f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary",
                return_value=(True, "trackless binary COLMAP"),
            ) as probe,
            mock.patch(f"{_RECONSTRUCT_MODULE}.TracklessColmapSceneSource", return_value=source) as source_type,
            mock.patch(f"{_RECONSTRUCT_MODULE}.load_sfm_scene") as full_loader,
            mock.patch.object(
                type(command.tx),
                "scene_transform",
                new_callable=mock.PropertyMock,
                return_value=scene_transform,
            ),
            mock.patch(f"{_RECONSTRUCT_MODULE}.GaussianSplatReconstructionWriter", return_value=writer),
            mock.patch.object(command, "_run_single_reconstruction") as run_single,
            self.assertLogs(_RECONSTRUCT_MODULE, level="INFO") as captured_logs,
        ):
            command.execute()

        probe.assert_called_once_with(command.dataset_path)
        source_type.assert_called_once_with(command.dataset_path)
        full_loader.assert_not_called()
        scanned_bbox = source.count_points_and_check_strict_id_order.call_args.args[0]
        self.assertEqual(scanned_bbox.dtype, np.float32)
        np.testing.assert_allclose(scanned_bbox, command.tx.crop_bbox)
        self.assertEqual(
            source.count_points_and_check_strict_id_order.call_args.kwargs,
            {"bounds_mode": "open"},
        )
        selected_bbox = source.scene_for_bbox.call_args.args[0]
        self.assertEqual(selected_bbox.dtype, np.float32)
        np.testing.assert_allclose(selected_bbox, command.tx.crop_bbox)
        self.assertEqual(
            source.scene_for_bbox.call_args.kwargs,
            {"bounds_mode": "open", "expected_selected_count": 7},
        )
        scene_transform.assert_called_once_with(partial_scene)
        run_single.assert_called_once_with(transformed_scene, writer, None)
        logs = "\n".join(captured_logs.output)
        self.assertIn("Single-scene COLMAP bbox fast path selected", logs)
        self.assertIn("scanning 12,345 trackless initialization points", logs)
        self.assertIn("selected 7 of 12,345 initialization points", logs)
        self.assertIn("loaded 7 of 12,345 initialization points", logs)
        self.assertIn("Applying configured scene transforms to 7 initialization points", logs)
        self.assertIn("Scene transforms complete: 5 initialization points across 2 images", logs)

    def test_single_colmap_bbox_fast_path_falls_back_for_unsorted_point_ids(self):
        command = self._compatible_single_bbox_command()
        loaded_scene = SimpleNamespace(points=np.zeros((9, 3), dtype=np.float32), images=(object(),))
        transformed_scene = SimpleNamespace(points=np.zeros((8, 3), dtype=np.float32), images=(object(),))
        source = mock.Mock()
        source.point_count = 12_345
        source.count_points_and_check_strict_id_order.return_value = (7, False)
        scene_transform = mock.Mock(return_value=transformed_scene)
        writer = mock.Mock()

        with (
            mock.patch(
                f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary",
                return_value=(True, "trackless binary COLMAP"),
            ) as probe,
            mock.patch(f"{_RECONSTRUCT_MODULE}.TracklessColmapSceneSource", return_value=source),
            mock.patch(f"{_RECONSTRUCT_MODULE}.load_sfm_scene", return_value=loaded_scene) as full_loader,
            mock.patch.object(
                type(command.tx),
                "scene_transform",
                new_callable=mock.PropertyMock,
                return_value=scene_transform,
            ),
            mock.patch(f"{_RECONSTRUCT_MODULE}.GaussianSplatReconstructionWriter", return_value=writer),
            mock.patch.object(command, "_run_single_reconstruction") as run_single,
            self.assertLogs(_RECONSTRUCT_MODULE, level="WARNING") as captured_logs,
        ):
            command.execute()

        probe.assert_called_once_with(command.dataset_path)
        source.count_points_and_check_strict_id_order.assert_called_once()
        source.scene_for_bbox.assert_not_called()
        full_loader.assert_called_once_with(command.dataset_path, command.dataset_type)
        scene_transform.assert_called_once_with(loaded_scene)
        run_single.assert_called_once_with(transformed_scene, writer, None)
        self.assertIn("strictly increasing point-ID order", "\n".join(captured_logs.output))

    def test_single_colmap_bbox_fast_path_falls_back_for_unsupported_models(self):
        command = self._compatible_single_bbox_command()
        loaded_scene = SimpleNamespace(points=np.zeros((9, 3), dtype=np.float32), images=(object(),))
        transformed_scene = SimpleNamespace(points=np.zeros((8, 3), dtype=np.float32), images=(object(),))
        scene_transform = mock.Mock(return_value=transformed_scene)
        writer = mock.Mock()

        with (
            mock.patch(
                f"{_RECONSTRUCT_MODULE}.probe_trackless_colmap_binary",
                return_value=(False, "visibility tracks or text model"),
            ) as probe,
            mock.patch(f"{_RECONSTRUCT_MODULE}.TracklessColmapSceneSource") as source_type,
            mock.patch(f"{_RECONSTRUCT_MODULE}.load_sfm_scene", return_value=loaded_scene) as full_loader,
            mock.patch.object(
                type(command.tx),
                "scene_transform",
                new_callable=mock.PropertyMock,
                return_value=scene_transform,
            ),
            mock.patch(f"{_RECONSTRUCT_MODULE}.GaussianSplatReconstructionWriter", return_value=writer),
            mock.patch.object(command, "_run_single_reconstruction") as run_single,
        ):
            command.execute()

        probe.assert_called_once_with(command.dataset_path)
        source_type.assert_not_called()
        full_loader.assert_called_once_with(command.dataset_path, command.dataset_type)
        scene_transform.assert_called_once_with(loaded_scene)
        run_single.assert_called_once_with(transformed_scene, writer, None)


if __name__ == "__main__":
    unittest.main()
