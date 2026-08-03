# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import hashlib
import pathlib
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fvdb_reality_capture.sfm_scene import load_colmap_metadata_scene
from fvdb_reality_capture.sfm_scene.colmap_partial import (
    ColmapBinaryPointSource,
    TracklessColmapSceneSource,
    UnsupportedColmapPartialLoadError,
    probe_trackless_colmap_binary,
)


def _write_camera_model(sparse_path: pathlib.Path) -> None:
    with (sparse_path / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<IiQQ4d", 1, 1, 64, 48, 50.0, 51.0, 32.0, 24.0))

    with (sparse_path / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<I7dI", 7, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1))
        file.write(b"frame.png\0")
        file.write(struct.pack("<Q", 0))


def _write_points(sparse_path: pathlib.Path, points, track: tuple[int, int] | None = None) -> None:
    with (sparse_path / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(points)))
        for point_id, xyz, rgb, error in points:
            track_length = 0 if track is None else 1
            file.write(struct.pack("<Q3d3BdQ", point_id, *xyz, *rgb, error, track_length))
            if track is not None:
                file.write(struct.pack("<II", *track))


class ColmapPartialTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.sparse_path = self.root / "sparse" / "0"
        self.sparse_path.mkdir(parents=True)
        (self.root / "images").mkdir()
        (self.root / "images" / "frame.png").write_bytes(b"")
        _write_camera_model(self.sparse_path)
        self.points = [
            (9, (-1.0, 0.0, 0.0), (10, 20, 30), 0.1),
            (2, (0.0, 0.0, 0.0), (40, 50, 60), 0.2),
            (5, (1.0, 1.0, 1.0), (70, 80, 90), 0.3),
            (4, (2.0, 0.0, 0.0), (100, 110, 120), 0.4),
        ]
        _write_points(self.sparse_path, self.points)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_probe_rejects_points_with_visibility_tracks(self):
        supported, reason = probe_trackless_colmap_binary(self.root)
        self.assertTrue(supported, reason)

        _write_points(self.sparse_path, self.points[:1], track=(7, 0))
        supported, reason = probe_trackless_colmap_binary(self.root)
        self.assertFalse(supported)
        self.assertIn("empty visibility track", reason)
        with self.assertRaisesRegex(UnsupportedColmapPartialLoadError, "empty visibility track"):
            ColmapBinaryPointSource(self.root)

    def test_bounded_queries_preserve_columns_and_boundary_modes(self):
        source = ColmapBinaryPointSource(self.root, block_size=2)
        self.assertEqual(source.point_count, 4)
        self.assertEqual(len(source.fingerprint), 64)
        np.testing.assert_allclose(source.bounds(), [-1.0, 0.0, 0.0, 2.0, 1.0, 1.0])

        bbox = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        self.assertEqual(source.count_points(bbox, bounds_mode="open"), 1)
        self.assertEqual(source.count_points(bbox, bounds_mode="closed"), 3)
        self.assertEqual(source.count_points(bbox, bounds_mode="half_open"), 2)

        selection = source.query_points(bbox, bounds_mode="closed")
        np.testing.assert_array_equal(selection.point_ids, [9, 2, 5])
        np.testing.assert_allclose(selection.points, [[-1, 0, 0], [0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(selection.points_rgb, [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        np.testing.assert_allclose(selection.points_err, [0.1, 0.2, 0.3])
        self.assertEqual(selection.points.dtype, np.float32)
        self.assertEqual(selection.points_err.dtype, np.float32)

    def test_count_and_order_check_detects_cross_block_point_id_inversion(self):
        cross_block_inversion = [self.points[index] for index in (1, 2, 3, 0)]
        _write_points(self.sparse_path, cross_block_inversion)
        bbox = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

        source = TracklessColmapSceneSource(self.root, block_size=2)
        self.assertEqual(
            source.count_points_and_check_strict_id_order(bbox, bounds_mode="closed"),
            (3, False),
        )
        self.assertFalse((self.root / "_cache").exists())

        del source
        _write_points(self.sparse_path, sorted(cross_block_inversion, key=lambda point: point[0]))
        ordered_source = TracklessColmapSceneSource(self.root, block_size=2)
        self.assertEqual(
            ordered_source.count_points_and_check_strict_id_order(bbox, bounds_mode="closed"),
            (3, True),
        )
        self.assertFalse((self.root / "_cache").exists())

    def test_point_only_access_is_metadata_and_cache_lazy_with_full_content_hash(self):
        metadata_loader = "fvdb_reality_capture.sfm_scene.colmap_partial._load_metadata_without_points"
        with patch(metadata_loader) as load_metadata:
            source = TracklessColmapSceneSource(self.root, block_size=2)
            self.assertEqual(source.count_points(), len(self.points))
            full_fingerprint = source.full_fingerprint

        load_metadata.assert_not_called()
        self.assertFalse((self.root / "_cache").exists())
        expected_fingerprint = hashlib.sha256((self.sparse_path / "points3D.bin").read_bytes()).hexdigest()
        self.assertEqual(full_fingerprint, expected_fingerprint)
        self.assertEqual(source.full_fingerprint, expected_fingerprint)

    def test_partial_scene_loads_metadata_without_visibility_or_outside_points(self):
        source = TracklessColmapSceneSource(self.root, block_size=2)
        bbox = np.array([-0.5, -0.5, -0.5, 1.5, 1.5, 1.5])
        selected_count = source.count_points(bbox)
        with patch.object(source.point_source, "count_points", side_effect=AssertionError("unexpected recount")):
            scene = source.scene_for_bbox(bbox, expected_selected_count=selected_count)

        self.assertEqual(scene.points.shape, (2, 3))
        np.testing.assert_allclose(scene.points, [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(scene.points_rgb, [[40, 50, 60], [70, 80, 90]])
        self.assertEqual(scene.num_images, 1)
        self.assertEqual(scene.num_cameras, 1)
        self.assertFalse(scene.has_visible_point_indices)
        np.testing.assert_array_equal(scene.images[0].point_indices, np.empty((0,), dtype=np.int32))
        self.assertEqual(pathlib.Path(scene.images[0].image_path).name, "frame.png")
        np.testing.assert_allclose(scene.images[0].world_to_camera_matrix[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(scene.transformation_matrix, np.eye(4))
        np.testing.assert_allclose(scene.scene_bbox, bbox)

    def test_expected_selected_count_skips_recount_and_validates_mismatches(self):
        source = ColmapBinaryPointSource(self.root, block_size=2)
        bbox = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

        with patch.object(source, "count_points", side_effect=AssertionError("unexpected recount")):
            selection = source.query_points(bbox, bounds_mode="closed", expected_selected_count=3)
        self.assertEqual(len(selection), 3)

        with self.assertRaisesRegex(ValueError, "expected_selected_count=2"):
            source.query_points(bbox, bounds_mode="closed", expected_selected_count=2)
        with self.assertRaisesRegex(ValueError, "expected_selected_count=4"):
            source.query_points(bbox, bounds_mode="closed", expected_selected_count=4)
        with self.assertRaises(TypeError):
            source.query_points(bbox, expected_selected_count=True)
        with self.assertRaises(ValueError):
            source.query_points(bbox, expected_selected_count=-1)

    def test_metadata_scene_has_finite_explicit_bbox_and_no_points(self):
        bbox = np.array([-3.0, -2.0, -1.0, 3.0, 2.0, 1.0])
        scene = load_colmap_metadata_scene(self.root, bbox, block_size=2)

        self.assertEqual(scene.points.shape, (0, 3))
        self.assertEqual(scene.points_err.shape, (0,))
        self.assertEqual(scene.points_rgb.shape, (0, 3))
        np.testing.assert_allclose(scene.scene_bbox, bbox)
        self.assertEqual(scene.num_images, 1)
        self.assertEqual(scene.num_cameras, 1)
        self.assertFalse(scene.has_visible_point_indices)
        np.testing.assert_allclose(scene.transformation_matrix, np.eye(4))

        with self.assertRaisesRegex(ValueError, "finite"):
            TracklessColmapSceneSource(self.root).metadata_scene([-np.inf, -1, -1, np.inf, 1, 1])
        with self.assertRaisesRegex(ValueError, "NaN"):
            TracklessColmapSceneSource(self.root).metadata_scene([np.nan, -1, -1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
