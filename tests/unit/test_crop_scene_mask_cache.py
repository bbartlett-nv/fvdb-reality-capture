# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from fvdb_reality_capture import CameraModel
from fvdb_reality_capture.sfm_scene import (
    PerImageValueAttribute,
    SfmCache,
    SfmCameraMetadata,
    SfmPosedImageMetadata,
    SfmScene,
)
from fvdb_reality_capture.sfm_scene.scene_attribute import CROP_MASK_BBOX_ATTRIBUTE
from fvdb_reality_capture.transforms.crop_scene import (
    CropScene,
    _MASK_BBOX_IMAGE_IDS_KEY,
    _MASK_BBOX_MANIFEST_VERSION_KEY,
    _mask_bbox_xyxy_count,
    _rasterize_convex_hull_mask,
)
from fvdb_reality_capture.transforms.downsample_images import DownsampleImages


class CropSceneMaskCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_scene(self) -> SfmScene:
        image_path = self.root / "image.png"
        self.assertTrue(cv2.imwrite(str(image_path), np.zeros((24, 32, 3), dtype=np.uint8)))

        camera_metadata = SfmCameraMetadata(
            img_width=32,
            img_height=24,
            fx=8.0,
            fy=8.0,
            cx=16.0,
            cy=12.0,
            camera_model=CameraModel.PINHOLE,
            distortion_coeffs=np.zeros((12,), dtype=np.float32),
        )
        images = [
            SfmPosedImageMetadata(
                world_to_camera_matrix=np.eye(4, dtype=np.float32),
                camera_to_world_matrix=np.eye(4, dtype=np.float32),
                camera_metadata=camera_metadata,
                camera_id=1,
                image_id=image_id,
                image_path=str(image_path),
                mask_path="",
                point_indices=np.array([0, 1], dtype=np.int64),
            )
            for image_id in (7, 42)
        ]
        cache = SfmCache.get_cache(self.root / "cache", "crop_scene_mask_cache", "CropScene mask-cache test")
        return SfmScene(
            cameras={1: camera_metadata},
            images=images,
            points=np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]], dtype=np.float32),
            points_err=np.zeros((2,), dtype=np.float32),
            points_rgb=np.zeros((2, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=np.eye(4, dtype=np.float32),
            cache=cache,
        )

    @staticmethod
    def _attribute_rows(scene: SfmScene) -> np.ndarray:
        attribute = scene.get_attribute(CROP_MASK_BBOX_ATTRIBUTE)
        if not isinstance(attribute, PerImageValueAttribute):
            raise AssertionError(f"Unexpected crop-mask attribute type: {type(attribute).__name__}")
        return np.stack(attribute.values)

    def test_mask_bbox_matches_bool_binary_normalized_and_byte_masks(self):
        expected = np.array([2, 1, 5, 4, 2], dtype=np.int64)
        mask = np.zeros((5, 6), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True
        normalized_mask = mask.astype(np.float32) * 0.75
        normalized_mask[1, 2] = 0.25

        np.testing.assert_array_equal(_mask_bbox_xyxy_count(mask), expected)
        np.testing.assert_array_equal(_mask_bbox_xyxy_count(mask.astype(np.uint8)), expected)
        np.testing.assert_array_equal(
            _mask_bbox_xyxy_count(normalized_mask), np.array([4, 3, 5, 4, 1], dtype=np.int64)
        )
        np.testing.assert_array_equal(_mask_bbox_xyxy_count(mask.astype(np.uint8) * 255), expected)
        np.testing.assert_array_equal(
            _mask_bbox_xyxy_count(np.zeros((5, 6), dtype=np.uint8)), np.zeros((5,), dtype=np.int64)
        )

        byte_mask = np.zeros((5, 6), dtype=np.uint8)
        byte_mask[0, 0] = 127
        byte_mask[4, 5] = 128
        np.testing.assert_array_equal(
            _mask_bbox_xyxy_count(byte_mask), np.array([5, 4, 6, 5, 1], dtype=np.int64)
        )

    def test_bounded_rasterizer_preserves_closed_half_space_semantics(self):
        hull = ConvexHull(
            np.array(
                [
                    [-4.2, 3.2],
                    [5.7, -2.1],
                    [22.4, 5.5],
                    [18.2, 18.7],
                    [2.3, 15.6],
                ]
            )
        )
        image_height, image_width = 17, 23
        pixel_u, pixel_v = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing="xy")
        pixel_coords = np.stack([pixel_u, pixel_v], axis=-1)
        expected = np.all(
            pixel_coords @ hull.equations[:, :-1].T + hull.equations[np.newaxis, np.newaxis, :, -1] <= 0.0,
            axis=-1,
        )

        actual = _rasterize_convex_hull_mask(hull, image_height, image_width)

        np.testing.assert_array_equal(actual, expected)

    def test_fresh_jpeg_metadata_matches_decoded_lossy_mask(self):
        scene = self._make_scene()
        output = CropScene(
            np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32),
            mask_format="jpg",
        )(scene)

        rows = self._attribute_rows(output)
        for row, image_metadata in zip(rows, output.images):
            decoded_mask = cv2.imread(image_metadata.mask_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(decoded_mask)
            np.testing.assert_array_equal(row, _mask_bbox_xyxy_count(decoded_mask))

        _, manifest = output.cache.read_file("transform")
        np.testing.assert_array_equal(manifest[CROP_MASK_BBOX_ATTRIBUTE], rows)

    def test_downsample_images_invalidates_full_resolution_crop_mask_bbox(self):
        scene = self._make_scene()
        cropped = CropScene(np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32))(scene)

        self.assertTrue(cropped.has_attribute(CROP_MASK_BBOX_ATTRIBUTE))
        downsampled = DownsampleImages(2, image_type="png")(cropped)

        self.assertFalse(downsampled.has_attribute(CROP_MASK_BBOX_ATTRIBUTE))
        for image_metadata in downsampled.images:
            resized_mask = cv2.imread(image_metadata.mask_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(resized_mask)
            self.assertEqual(resized_mask.shape, (12, 16))

    def test_fresh_cache_hit_and_legacy_manifest_upgrade(self):
        scene = self._make_scene()
        transform = CropScene(np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32))

        first_output = transform(scene)
        first_rows = self._attribute_rows(first_output)
        self.assertEqual(first_rows.shape, (2, 5))
        self.assertTrue(first_output.cache.has_file("mask_007"))
        self.assertTrue(first_output.cache.has_file("mask_042"))
        self.assertFalse(first_output.cache.has_file("mask_000"))
        for row, image_metadata in zip(first_rows, first_output.images):
            mask = cv2.imread(image_metadata.mask_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            np.testing.assert_array_equal(row, _mask_bbox_xyxy_count(mask))

        with mock.patch(
            "fvdb_reality_capture.transforms.crop_scene._rasterize_convex_hull_mask",
            side_effect=AssertionError("cache hit rerasterized masks"),
        ):
            cached_output = transform(scene)
        np.testing.assert_array_equal(self._attribute_rows(cached_output), first_rows)

        first_output.cache.write_file(
            "transform",
            {"transform": scene.transformation_matrix},
            data_type="pt",
        )
        with mock.patch(
            "fvdb_reality_capture.transforms.crop_scene._rasterize_convex_hull_mask",
            side_effect=AssertionError("legacy cache upgrade rerasterized masks"),
        ):
            upgraded_output = transform(scene)
        np.testing.assert_array_equal(self._attribute_rows(upgraded_output), first_rows)

        _, upgraded_manifest = upgraded_output.cache.read_file("transform")
        self.assertIn(_MASK_BBOX_MANIFEST_VERSION_KEY, upgraded_manifest)
        np.testing.assert_array_equal(upgraded_manifest[_MASK_BBOX_IMAGE_IDS_KEY], np.array([7, 42]))
        np.testing.assert_array_equal(upgraded_manifest[CROP_MASK_BBOX_ATTRIBUTE], first_rows)


if __name__ == "__main__":
    unittest.main()
