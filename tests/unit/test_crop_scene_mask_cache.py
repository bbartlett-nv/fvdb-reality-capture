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
    CropSceneToPoints,
    _MASK_BBOX_IMAGE_IDS_KEY,
    _MASK_BBOX_MANIFEST_VERSION_KEY,
    _mask_bbox_xyxy_count,
    _points_in_bbox_mask,
    _project_bbox_mask,
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

    @staticmethod
    def _replace_image_inputs(
        scene: SfmScene,
        *,
        camera_metadata: SfmCameraMetadata | None = None,
        mask_path: str | None = None,
    ) -> SfmScene:
        camera = scene.cameras[1] if camera_metadata is None else camera_metadata
        images = [
            SfmPosedImageMetadata(
                world_to_camera_matrix=image.world_to_camera_matrix,
                camera_to_world_matrix=image.camera_to_world_matrix,
                camera_metadata=camera,
                camera_id=image.camera_id,
                image_id=image.image_id,
                image_path=image.image_path,
                mask_path=image.mask_path if mask_path is None else mask_path,
                point_indices=image.point_indices,
            )
            for image in scene.images
        ]
        return scene.replace(cameras={1: camera}, images=images)

    def test_mask_bbox_matches_bool_binary_normalized_and_byte_masks(self):
        expected = np.array([2, 1, 5, 4, 2], dtype=np.int64)
        mask = np.zeros((5, 6), dtype=bool)
        mask[1, 2] = True
        mask[3, 4] = True
        normalized_mask = mask.astype(np.float32) * 0.75
        normalized_mask[1, 2] = 0.25

        np.testing.assert_array_equal(_mask_bbox_xyxy_count(mask), expected)
        np.testing.assert_array_equal(_mask_bbox_xyxy_count(mask.astype(np.uint8)), expected)
        np.testing.assert_array_equal(_mask_bbox_xyxy_count(normalized_mask), np.array([4, 3, 5, 4, 1], dtype=np.int64))
        np.testing.assert_array_equal(_mask_bbox_xyxy_count(mask.astype(np.uint8) * 255), expected)
        np.testing.assert_array_equal(
            _mask_bbox_xyxy_count(np.zeros((5, 6), dtype=np.uint8)), np.zeros((5,), dtype=np.int64)
        )

        byte_mask = np.zeros((5, 6), dtype=np.uint8)
        byte_mask[0, 0] = 127
        byte_mask[4, 5] = 128
        np.testing.assert_array_equal(_mask_bbox_xyxy_count(byte_mask), np.array([5, 4, 6, 5, 1], dtype=np.int64))

    def test_point_bbox_mask_uses_strict_bounds_across_blocks(self):
        bbox = np.array([-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, -3.0],
                [0.0, 0.0, 6.0],
                [3.999, 4.999, 5.999],
                [-1.001, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        expected = np.logical_and.reduce(
            [
                points[:, 0] > bbox[0],
                points[:, 0] < bbox[3],
                points[:, 1] > bbox[1],
                points[:, 1] < bbox[4],
                points[:, 2] > bbox[2],
                points[:, 2] < bbox[5],
            ]
        )

        actual = _points_in_bbox_mask(points, bbox, block_size=2)

        np.testing.assert_array_equal(actual, expected)

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

    def test_fresh_cache_hit_and_legacy_manifest_invalidation(self):
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
            "fvdb_reality_capture.transforms.crop_scene._project_bbox_mask",
            wraps=_project_bbox_mask,
        ) as project_bbox:
            regenerated_output = transform(scene)
        self.assertEqual(project_bbox.call_count, 2)
        np.testing.assert_array_equal(self._attribute_rows(regenerated_output), first_rows)

        _, regenerated_manifest = regenerated_output.cache.read_file("transform")
        self.assertIn(_MASK_BBOX_MANIFEST_VERSION_KEY, regenerated_manifest)
        np.testing.assert_array_equal(regenerated_manifest[_MASK_BBOX_IMAGE_IDS_KEY], np.array([7, 42]))
        np.testing.assert_array_equal(regenerated_manifest[CROP_MASK_BBOX_ATTRIBUTE], first_rows)

    def test_bbox_projection_handles_inside_behind_near_plane_and_unsupported_cameras(self):
        scene = self._make_scene()
        image = scene.images[0]

        inside_mask = _project_bbox_mask(
            image,
            np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        behind_mask = _project_bbox_mask(
            image,
            np.array([-1.0, -1.0, -3.0, 1.0, 1.0, -1.0], dtype=np.float32),
        )
        crossing_mask = _project_bbox_mask(
            image,
            np.array([0.5, -1.0, -1.0, 2.0, 1.0, 1.0], dtype=np.float32),
        )

        self.assertTrue(np.all(inside_mask))
        self.assertFalse(np.any(behind_mask))
        self.assertEqual(crossing_mask.dtype, np.bool_)
        self.assertEqual(crossing_mask.shape, (24, 32))
        self.assertTrue(np.any(crossing_mask))

        unsupported_camera = SfmCameraMetadata(
            img_width=32,
            img_height=24,
            fx=8.0,
            fy=8.0,
            cx=16.0,
            cy=12.0,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=np.zeros((12,), dtype=np.float32),
        )
        unsupported_image = self._replace_image_inputs(scene, camera_metadata=unsupported_camera).images[0]
        with self.assertRaisesRegex(NotImplementedError, "UndistortImages"):
            _project_bbox_mask(
                unsupported_image,
                np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32),
            )

    def test_crop_scene_state_restores_mask_settings_and_point_margin_is_total_fraction(self):
        bbox = np.array([-1.0, -2.0, 1.0, 3.0, 4.0, 8.0], dtype=np.float32)
        restored = CropScene.from_state_dict(
            CropScene(bbox, mask_format="npy", composite_with_existing_masks=False).state_dict()
        )
        np.testing.assert_array_equal(restored._bbox, bbox)
        self.assertEqual(restored._mask_format, "npy")
        self.assertFalse(restored._composite_with_existing_masks)

        scene = self._make_scene().replace(
            points=np.array([[0.0, 0.0, 1.0], [2.0, 4.0, 7.0]], dtype=np.float32),
            points_err=np.zeros((2,), dtype=np.float32),
            points_rgb=np.zeros((2, 3), dtype=np.uint8),
        )
        with mock.patch(
            "fvdb_reality_capture.transforms.crop_scene._crop_scene_to_bbox",
            return_value=scene,
        ) as crop_to_bbox:
            CropSceneToPoints(margin=0.1)(scene)

        passed_bbox = crop_to_bbox.call_args.kwargs["bbox"]
        np.testing.assert_allclose(
            passed_bbox,
            np.array([-0.1, -0.2, 0.7, 2.1, 4.2, 7.3], dtype=np.float32),
        )

    def test_crop_cache_fingerprints_camera_state_and_validates_decoded_bbox(self):
        scene = self._make_scene()
        bbox = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32)
        transform = CropScene(bbox)
        output = transform(scene)

        _, manifest = output.cache.read_file("transform")
        manifest[CROP_MASK_BBOX_ATTRIBUTE] = np.zeros((2, 5), dtype=np.int64)
        output.cache.write_file("transform", manifest, data_type="pt")
        with mock.patch(
            "fvdb_reality_capture.transforms.crop_scene._project_bbox_mask",
            wraps=_project_bbox_mask,
        ) as project_bbox:
            transform(scene)
        self.assertEqual(project_bbox.call_count, 2)

        changed_camera = SfmCameraMetadata(
            img_width=32,
            img_height=24,
            fx=9.0,
            fy=8.0,
            cx=16.0,
            cy=12.0,
            camera_model=CameraModel.PINHOLE,
            distortion_coeffs=np.zeros((12,), dtype=np.float32),
        )
        changed_scene = self._replace_image_inputs(scene, camera_metadata=changed_camera)
        with mock.patch(
            "fvdb_reality_capture.transforms.crop_scene._project_bbox_mask",
            wraps=_project_bbox_mask,
        ) as project_bbox:
            transform(changed_scene)
        self.assertEqual(project_bbox.call_count, 2)

    def test_crop_composites_bool_npy_and_invalidates_when_source_mask_changes(self):
        scene = self._make_scene()
        source_mask_path = self.root / "source_mask.npy"
        source_mask = np.zeros((24, 32), dtype=np.bool_)
        source_mask[4:20, 6:26] = True
        np.save(source_mask_path, source_mask)
        masked_scene = self._replace_image_inputs(scene, mask_path=str(source_mask_path))
        transform = CropScene(np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32))

        first_output = transform(masked_scene)
        for image in first_output.images:
            persisted = cv2.imread(image.mask_path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(persisted)
            self.assertTrue(set(np.unique(persisted)).issubset({0, 255}))
            self.assertFalse(np.any((persisted > 0) & ~source_mask))

        np.save(source_mask_path, np.zeros_like(source_mask))
        with mock.patch(
            "fvdb_reality_capture.transforms.crop_scene._project_bbox_mask",
            wraps=_project_bbox_mask,
        ) as project_bbox:
            second_output = transform(masked_scene)
        self.assertEqual(project_bbox.call_count, 2)
        for image in second_output.images:
            persisted = cv2.imread(image.mask_path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(persisted)
            self.assertFalse(np.any(persisted))

        missing_scene = self._replace_image_inputs(scene, mask_path=str(self.root / "missing.npy"))
        with self.assertRaisesRegex(FileNotFoundError, "Declared mask file does not exist"):
            transform(missing_scene)

    def test_downsample_masks_are_canonical_and_source_content_invalidates_cache(self):
        scene = self._make_scene()
        source_mask_path = self.root / "downsample_source.npy"
        source_mask = np.zeros((24, 32), dtype=np.bool_)
        source_mask[3:21, 5:27] = True
        np.save(source_mask_path, source_mask)
        masked_scene = self._replace_image_inputs(scene, mask_path=str(source_mask_path))
        transform = DownsampleImages(2, image_type="png")

        first_output = transform(masked_scene)
        for image in first_output.images:
            persisted = cv2.imread(image.mask_path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(persisted)
            self.assertTrue(set(np.unique(persisted)).issubset({0, 255}))
            self.assertTrue(np.any(persisted))

        np.save(source_mask_path, np.zeros_like(source_mask))
        second_output = transform(masked_scene)
        for image in second_output.images:
            persisted = cv2.imread(image.mask_path, cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(persisted)
            self.assertFalse(np.any(persisted))

        missing_scene = self._replace_image_inputs(scene, mask_path=str(self.root / "missing_downsample.npy"))
        with self.assertRaisesRegex(FileNotFoundError, "Declared mask file does not exist"):
            transform(missing_scene)


if __name__ == "__main__":
    unittest.main()
