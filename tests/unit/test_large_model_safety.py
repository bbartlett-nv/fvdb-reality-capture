# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree as ScipyKDTree

import fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction as reconstruction_module
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
)
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import NormalizeScene


def _make_trackless_scene(root: pathlib.Path, points: np.ndarray) -> SfmScene:
    return SfmScene(
        cameras={},
        images=[],
        points=np.asarray(points, dtype=np.float32),
        points_err=np.arange(len(points), dtype=np.float32),
        points_rgb=np.arange(len(points) * 3, dtype=np.uint8).reshape((-1, 3)),
        scene_bbox=np.array([-1.0, -1.0, -1.0, 2.0, 2.0, 2.0], dtype=np.float32),
        transformation_matrix=np.eye(4, dtype=np.float32),
        cache=SfmCache.get_cache(root / "cache", "large_model_test", "Large model safety test cache"),
    )


def _make_training_dataset(points: np.ndarray):
    points = np.asarray(points, dtype=np.float32)
    return SimpleNamespace(
        points=points,
        points_rgb=np.full((len(points), 3), 127, dtype=np.uint8),
        scene_bbox=np.array([-1.0, -1.0, -1.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )


class LargeModelSafetyTest(unittest.TestCase):
    def test_none_normalization_is_identity_on_empty_metadata_scene(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene = _make_trackless_scene(pathlib.Path(temporary_directory), np.empty((0, 3), dtype=np.float32))
            transform = NormalizeScene(normalization_type="none")

            output = transform(scene)

            self.assertIs(output, scene)
            normalization = transform._compute_normalization_transform(scene)
            self.assertIsNotNone(normalization)
            assert normalization is not None
            self.assertEqual(normalization.dtype, np.float32)
            np.testing.assert_array_equal(normalization, np.eye(4, dtype=np.float32))

    def test_trackless_filter_avoids_visibility_remap_and_preserves_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            points = np.arange(18, dtype=np.float32).reshape((6, 3))
            scene = _make_trackless_scene(pathlib.Path(temporary_directory), points)
            mask = np.array([True, False, True, False, False, True])

            filtered = scene.filter_points(mask)

            np.testing.assert_array_equal(filtered.points, points[mask])
            np.testing.assert_array_equal(filtered.points_err, scene.points_err[mask])
            np.testing.assert_array_equal(filtered.points_rgb, scene.points_rgb[mask])
            self.assertFalse(filtered.has_visible_point_indices)

    def test_all_true_filter_returns_same_scene_without_copying_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene = _make_trackless_scene(
                pathlib.Path(temporary_directory), np.arange(12, dtype=np.float32).reshape((4, 3))
            )

            filtered = scene.filter_points(np.ones(4, dtype=np.bool_))

            self.assertIs(filtered, scene)

    def test_filter_rejects_non_boolean_or_wrong_shape_masks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene = _make_trackless_scene(
                pathlib.Path(temporary_directory), np.arange(12, dtype=np.float32).reshape((4, 3))
            )
            with self.assertRaisesRegex(TypeError, "boolean dtype"):
                scene.filter_points(np.ones(4, dtype=np.int32))
            with self.assertRaisesRegex(ValueError, "shape"):
                scene.filter_points(np.ones((4, 1), dtype=np.bool_))

    def test_initial_point_count_above_cap_fails_before_knn(self):
        dataset = _make_training_dataset(np.arange(15, dtype=np.float32).reshape((5, 3)))
        config = GaussianSplatReconstructionConfig(sh_degree=0)
        optimizer_config = GaussianSplatOptimizerConfig(max_gaussians=4)

        with mock.patch.object(reconstruction_module, "cKDTree", side_effect=AssertionError("KNN must not run")):
            with self.assertRaisesRegex(ValueError, "max_gaussians applies independently to each chunk"):
                GaussianSplatReconstruction._init_model(config, optimizer_config, "cpu", dataset)

    def test_knn_queries_are_batched_and_duplicate_scales_remain_finite(self):
        dataset = _make_training_dataset(np.zeros((5, 3), dtype=np.float32))
        config = GaussianSplatReconstructionConfig(sh_degree=0)
        optimizer_config = GaussianSplatOptimizerConfig(max_gaussians=5)
        query_sizes = []

        class RecordingKDTree:
            def __init__(self, points):
                self._tree = ScipyKDTree(points)

            def query(self, points, *args, **kwargs):
                query_sizes.append(len(points))
                return self._tree.query(points, *args, **kwargs)

        with (
            mock.patch.object(reconstruction_module, "_KNN_QUERY_BATCH_SIZE", 2),
            mock.patch.object(reconstruction_module, "cKDTree", RecordingKDTree),
        ):
            model = GaussianSplatReconstruction._init_model(config, optimizer_config, "cpu", dataset)

        self.assertEqual(query_sizes, [2, 2, 1])
        self.assertTrue(model.log_scales.isfinite().all().item())
        self.assertEqual(model.num_gaussians, 5)

    def test_singleton_initialization_uses_finite_bbox_fallback(self):
        dataset = _make_training_dataset(np.zeros((1, 3), dtype=np.float32))
        model = GaussianSplatReconstruction._init_model(
            GaussianSplatReconstructionConfig(sh_degree=0),
            GaussianSplatOptimizerConfig(max_gaussians=1),
            "cpu",
            dataset,
        )
        self.assertTrue(model.log_scales.isfinite().all().item())


if __name__ == "__main__":
    unittest.main()
