# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from fvdb_reality_capture import CameraModel
from fvdb_reality_capture.radiance_fields.gaussian_splat_dataset import (
    SfmDataset,
    _aligned_mask_crop,
    _dilated_mask_to_tiles,
    _load_binary_mask,
)
from fvdb_reality_capture.sfm_scene import (
    PerImageValueAttribute,
    SfmCache,
    SfmCameraMetadata,
    SfmPosedImageMetadata,
    SfmScene,
)
from fvdb_reality_capture.sfm_scene.scene_attribute import CROP_MASK_BBOX_ATTRIBUTE


def _packed_radtan5_coeffs() -> np.ndarray:
    coeffs = np.zeros((12,), dtype=np.float32)
    coeffs[0] = 0.1
    coeffs[1] = -0.05
    coeffs[2] = 0.01
    coeffs[6] = 0.002
    coeffs[7] = -0.003
    return coeffs


class GaussianSplatDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_scene(self) -> tuple[SfmScene, SfmCameraMetadata]:
        image_path = self.root / "image.png"
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))

        camera_metadata = SfmCameraMetadata(
            img_width=10,
            img_height=8,
            fx=6.0,
            fy=6.5,
            cx=5.0,
            cy=4.0,
            camera_model=CameraModel.OPENCV_RADTAN_5,
            distortion_coeffs=_packed_radtan5_coeffs(),
        )
        image_metadata = SfmPosedImageMetadata(
            world_to_camera_matrix=np.eye(4, dtype=np.float32),
            camera_to_world_matrix=np.eye(4, dtype=np.float32),
            camera_metadata=camera_metadata,
            camera_id=1,
            image_path=str(image_path),
            mask_path="",
            point_indices=np.array([], dtype=np.int64),
            image_id=0,
        )
        cache = SfmCache.get_cache(self.root / "cache_root", "dataset_unit_test_cache", "Dataset unit test cache")
        scene = SfmScene(
            cameras={1: camera_metadata},
            images=[image_metadata],
            points=np.zeros((0, 3), dtype=np.float32),
            points_err=np.zeros((0,), dtype=np.float32),
            points_rgb=np.zeros((0, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=np.eye(4, dtype=np.float32),
            cache=cache,
        )
        return scene, camera_metadata

    def _make_masked_scene(self, mask: np.ndarray) -> tuple[SfmScene, SfmCameraMetadata, np.ndarray]:
        height, width = mask.shape
        image_path = self.root / "masked_image.png"
        mask_path = self.root / "mask.png"
        pixels = (np.arange(height * width, dtype=np.uint32).reshape(height, width) % 251).astype(np.uint8)
        image = np.repeat(pixels[:, :, None], 3, axis=2)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        self.assertTrue(cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255))

        camera_metadata = SfmCameraMetadata(
            img_width=width,
            img_height=height,
            fx=80.0,
            fy=81.0,
            cx=width / 2.0,
            cy=height / 2.0,
            camera_model=CameraModel.PINHOLE,
            distortion_coeffs=np.zeros((12,), dtype=np.float32),
        )
        image_metadata = SfmPosedImageMetadata(
            world_to_camera_matrix=np.eye(4, dtype=np.float32),
            camera_to_world_matrix=np.eye(4, dtype=np.float32),
            camera_metadata=camera_metadata,
            camera_id=1,
            image_path=str(image_path),
            mask_path=str(mask_path),
            point_indices=np.array([], dtype=np.int64),
            image_id=0,
        )
        cache = SfmCache.get_cache(self.root / "masked_cache", "masked_dataset", "Masked dataset unit test cache")
        scene = SfmScene(
            cameras={1: camera_metadata},
            images=[image_metadata],
            points=np.zeros((0, 3), dtype=np.float32),
            points_err=np.zeros((0,), dtype=np.float32),
            points_rgb=np.zeros((0, 3), dtype=np.uint8),
            scene_bbox=None,
            transformation_matrix=np.eye(4, dtype=np.float32),
            cache=cache,
        )
        return scene, camera_metadata, image

    def test_dataset_returns_camera_model_and_distortion_coeffs(self):
        scene, _ = self._make_scene()

        datum = SfmDataset(scene)[0]

        self.assertEqual(int(datum["camera_model"]), int(CameraModel.OPENCV_RADTAN_5))
        np.testing.assert_allclose(datum["distortion_coeffs"].numpy(), _packed_radtan5_coeffs())

    def test_dataset_distortion_coeffs_do_not_alias_camera_metadata(self):
        scene, camera_metadata = self._make_scene()

        datum = SfmDataset(scene)[0]
        datum["distortion_coeffs"][0] = 999.0

        self.assertAlmostEqual(float(camera_metadata.distortion_coeffs[0]), 0.1)

    def test_all_modes_share_zero_one_mask_threshold(self):
        mask = np.ones((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)
        self.assertTrue(cv2.imwrite(scene.images[0].mask_path, np.ones(mask.shape, dtype=np.uint8)))

        off_datum = SfmDataset(scene, mask_rasterization_mode="off")[0]
        bbox_datum = SfmDataset(scene, mask_rasterization_mode="bbox")[0]

        self.assertTrue(np.all(off_datum["mask"]))
        self.assertTrue(np.all(bbox_datum["mask"]))

    def test_optimized_mask_loader_uses_dtype_appropriate_thresholds(self):
        float_path = self.root / "float_mask.npy"
        integer_path = self.root / "integer_mask.npy"
        byte_path = self.root / "byte_mask.npy"
        np.save(float_path, np.array([[0.0, 0.5, 0.5001, 1.0]], dtype=np.float32))
        np.save(integer_path, np.array([[0, 1]], dtype=np.uint8))
        np.save(byte_path, np.array([[0, 127, 128, 255]], dtype=np.uint8))

        np.testing.assert_array_equal(_load_binary_mask(str(float_path)), np.array([[False, False, True, True]]))
        np.testing.assert_array_equal(_load_binary_mask(str(integer_path)), np.array([[False, True]]))
        np.testing.assert_array_equal(_load_binary_mask(str(byte_path)), np.array([[False, False, True, True]]))

    def test_canonical_mask_loader_rejects_ambiguous_multichannel_masks(self):
        ambiguous_path = self.root / "ambiguous.png"
        ambiguous = np.zeros((4, 5, 3), dtype=np.uint8)
        ambiguous[1, 2, 0] = 255
        self.assertTrue(cv2.imwrite(str(ambiguous_path), ambiguous))

        with self.assertRaisesRegex(ValueError, "Ambiguous multi-channel mask"):
            _load_binary_mask(str(ambiguous_path))

    def test_all_modes_support_bool_npy_masks(self):
        mask = np.ones((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)
        npy_mask_path = self.root / "mask.npy"
        np.save(npy_mask_path, mask)
        old_metadata = scene.images[0]
        npy_metadata = SfmPosedImageMetadata(
            world_to_camera_matrix=old_metadata.world_to_camera_matrix,
            camera_to_world_matrix=old_metadata.camera_to_world_matrix,
            camera_metadata=old_metadata.camera_metadata,
            camera_id=old_metadata.camera_id,
            image_path=old_metadata.image_path,
            mask_path=str(npy_mask_path),
            point_indices=old_metadata.point_indices,
            image_id=old_metadata.image_id,
        )
        scene = scene.replace(images=[npy_metadata])

        off_datum = SfmDataset(scene, mask_rasterization_mode="off")[0]
        bbox_datum = SfmDataset(scene, mask_rasterization_mode="bbox")[0]
        self.assertTrue(np.all(off_datum["mask"]))
        self.assertTrue(np.all(bbox_datum["mask"]))

    def test_aligned_mask_crop_adds_ten_pixel_context_and_aligns_outward(self):
        mask = np.zeros((100, 120), dtype=np.bool_)
        mask[30:37, 35:42] = True

        crop = _aligned_mask_crop(mask, context_pixels=10, tile_size=16)

        self.assertEqual(crop, (16, 16, 48, 32))

    def test_aligned_mask_crop_clips_at_non_aligned_sensor_edges(self):
        mask = np.zeros((100, 120), dtype=np.bool_)
        mask[95:100, 110:120] = True

        crop = _aligned_mask_crop(mask, context_pixels=10, tile_size=16)

        self.assertEqual(crop, (96, 80, 24, 20))

    def test_aligned_mask_crop_returns_minimal_origin_crop_for_empty_mask(self):
        crop = _aligned_mask_crop(np.zeros((7, 9), dtype=np.bool_), context_pixels=10, tile_size=16)

        self.assertEqual(crop, (0, 0, 9, 7))

    def test_dilated_mask_to_tiles_preserves_ten_pixel_ssim_context(self):
        mask = np.zeros((64, 64), dtype=np.bool_)
        mask[31, 31] = True

        tile_mask = _dilated_mask_to_tiles(mask, context_pixels=10, tile_size=16)

        expected = np.zeros((4, 4), dtype=np.bool_)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(tile_mask, expected)

    def test_bbox_mode_crops_image_and_residual_mask_without_shifting_projection(self):
        mask = np.zeros((100, 120), dtype=np.bool_)
        mask[30:37, 35:42] = True
        scene, camera_metadata, source_image = self._make_masked_scene(mask)

        datum = SfmDataset(scene, mask_rasterization_mode="bbox")[0]

        np.testing.assert_array_equal(datum["raster_crop"], np.array([16, 16, 48, 32]))
        np.testing.assert_array_equal(datum["full_image_size"], np.array([100, 120]))
        self.assertAlmostEqual(float(datum["image_loss_scale"]), (48 * 32) / (120 * 100))
        self.assertEqual(datum["image"].shape, (32, 48, 3))
        self.assertEqual(datum["mask"].shape, (32, 48))
        np.testing.assert_array_equal(datum["image"], source_image[16:48, 16:64])
        np.testing.assert_array_equal(datum["mask"], mask[16:48, 16:64])
        np.testing.assert_allclose(datum["projection"].numpy(), camera_metadata.projection_matrix)
        self.assertNotIn("raster_tile_mask", datum)

    def test_bbox_mode_validates_persistent_crop_mask_bbox_attribute(self):
        mask = np.zeros((100, 120), dtype=np.bool_)
        mask[30:37, 35:42] = True
        scene, _, _ = self._make_masked_scene(mask)
        scene = scene.with_attributes(
            **{
                CROP_MASK_BBOX_ATTRIBUTE: PerImageValueAttribute(
                    [np.array([35, 30, 42, 37, int(mask.sum())], dtype=np.int64)]
                )
            }
        )

        with mock.patch(
            "fvdb_reality_capture.radiance_fields.gaussian_splat_dataset.load_binary_mask",
            wraps=_load_binary_mask,
        ) as load_mask:
            dataset = SfmDataset(scene, mask_rasterization_mode="bbox")
        load_mask.assert_called_once_with(scene.images[0].mask_path)
        datum = dataset[0]

        np.testing.assert_array_equal(datum["raster_crop"], np.array([16, 16, 48, 32]))

    def test_bbox_mode_rejects_all_empty_persistent_masks(self):
        mask = np.zeros((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)
        scene = scene.with_attributes(
            **{CROP_MASK_BBOX_ATTRIBUTE: PerImageValueAttribute([np.array([0, 0, 0, 0, 0], dtype=np.int64)])}
        )

        with self.assertRaisesRegex(ValueError, "no images with a valid mask"):
            SfmDataset(scene, mask_rasterization_mode="bbox")

    def test_bbox_mode_rejects_all_empty_masks_without_bbox_metadata(self):
        mask = np.zeros((7, 9), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)

        with self.assertRaisesRegex(ValueError, "no images with a valid mask"):
            SfmDataset(scene, mask_rasterization_mode="bbox")

    def test_bbox_mode_validates_persistent_bbox_bounds_and_count(self):
        mask = np.ones((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)

        invalid_bounds = scene.with_attributes(
            **{
                CROP_MASK_BBOX_ATTRIBUTE: PerImageValueAttribute(
                    [np.array([0, 0, 49, 32, int(mask.sum())], dtype=np.int64)]
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "Invalid raw mask bbox"):
            SfmDataset(invalid_bounds, mask_rasterization_mode="bbox")

        invalid_count = scene.with_attributes(
            **{CROP_MASK_BBOX_ATTRIBUTE: PerImageValueAttribute([np.array([10, 10, 12, 12, 5], dtype=np.int64)])}
        )
        with self.assertRaisesRegex(ValueError, "Stale or invalid"):
            SfmDataset(invalid_count, mask_rasterization_mode="bbox")

        invalid_empty_bounds = scene.with_attributes(
            **{CROP_MASK_BBOX_ATTRIBUTE: PerImageValueAttribute([np.array([0, 0, 1, 1, 0], dtype=np.int64)])}
        )
        with self.assertRaisesRegex(ValueError, "Stale or invalid"):
            SfmDataset(invalid_empty_bounds, mask_rasterization_mode="bbox")

    def test_bbox_mode_metadata_collates_for_batch_size_one(self):
        mask = np.zeros((100, 120), dtype=np.bool_)
        mask[30:37, 35:42] = True
        scene, _, _ = self._make_masked_scene(mask)
        loader = torch.utils.data.DataLoader(
            SfmDataset(scene, mask_rasterization_mode="bbox"), batch_size=1, num_workers=0
        )

        batch = next(iter(loader))

        self.assertEqual(tuple(batch["image"].shape), (1, 32, 48, 3))
        self.assertEqual(tuple(batch["mask"].shape), (1, 32, 48))
        self.assertEqual(tuple(batch["raster_crop"].shape), (1, 4))
        self.assertEqual(tuple(batch["full_image_size"].shape), (1, 2))
        self.assertEqual(tuple(batch["image_loss_scale"].shape), (1,))

    def test_tiles_mode_retains_full_frame_and_emits_compact_context_mask(self):
        mask = np.zeros((65, 70), dtype=np.bool_)
        mask[31, 31] = True
        scene, camera_metadata, source_image = self._make_masked_scene(mask)

        datum = SfmDataset(scene, mask_rasterization_mode="tiles")[0]

        np.testing.assert_array_equal(datum["image"], source_image)
        np.testing.assert_array_equal(datum["mask"], mask)
        np.testing.assert_array_equal(datum["raster_crop"], np.array([0, 0, 70, 65]))
        np.testing.assert_array_equal(datum["full_image_size"], np.array([65, 70]))
        self.assertEqual(float(datum["image_loss_scale"]), 1.0)
        self.assertEqual(datum["raster_tile_mask"].shape, (5, 5))
        expected = np.zeros((5, 5), dtype=np.bool_)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(datum["raster_tile_mask"], expected)
        np.testing.assert_allclose(datum["projection"].numpy(), camera_metadata.projection_matrix)

    def test_mask_rasterization_mode_requires_mask_at_dataset_construction(self):
        scene, _ = self._make_scene()

        with self.assertRaisesRegex(ValueError, "requires a mask"):
            SfmDataset(scene, mask_rasterization_mode="bbox")

    def test_tiles_mode_rejects_all_empty_masks(self):
        mask = np.zeros((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)

        with self.assertRaisesRegex(ValueError, "no images with a valid mask"):
            SfmDataset(scene, mask_rasterization_mode="tiles")

    def test_mask_aware_dataset_filters_empty_and_optional_small_masks(self):
        valid_mask = np.zeros((32, 48), dtype=np.bool_)
        valid_mask[4:20, 5:25] = True
        scene, _, _ = self._make_masked_scene(valid_mask)
        source_metadata = scene.images[0]

        empty_path = self.root / "empty_mask.png"
        tiny_path = self.root / "tiny_mask.png"
        self.assertTrue(cv2.imwrite(str(empty_path), np.zeros(valid_mask.shape, dtype=np.uint8)))
        tiny_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
        tiny_mask[0, 0] = 255
        self.assertTrue(cv2.imwrite(str(tiny_path), tiny_mask))

        def metadata(mask_path: str | pathlib.Path, image_id: int) -> SfmPosedImageMetadata:
            return SfmPosedImageMetadata(
                world_to_camera_matrix=source_metadata.world_to_camera_matrix,
                camera_to_world_matrix=source_metadata.camera_to_world_matrix,
                camera_metadata=source_metadata.camera_metadata,
                camera_id=source_metadata.camera_id,
                image_path=source_metadata.image_path,
                mask_path=str(mask_path),
                point_indices=source_metadata.point_indices,
                image_id=image_id,
            )

        scene = scene.replace(
            images=[
                metadata(empty_path, 10),
                metadata(tiny_path, 11),
                metadata(pathlib.Path(source_metadata.mask_path), 12),
            ]
        )

        default_dataset = SfmDataset(scene, mask_rasterization_mode="tiles")
        threshold_dataset = SfmDataset(
            scene,
            mask_rasterization_mode="bbox",
            minimum_valid_mask_area=0.01,
        )

        np.testing.assert_array_equal(default_dataset.indices, np.array([1, 2]))
        np.testing.assert_array_equal(threshold_dataset.indices, np.array([2]))

        off_scene = scene.replace(images=[metadata("", 9), *scene.images])
        off_dataset = SfmDataset(
            off_scene,
            filter_empty_masks=True,
            minimum_valid_mask_area=0.01,
        )
        np.testing.assert_array_equal(off_dataset.indices, np.array([0, 3]))
        self.assertNotIn("mask", off_dataset[0])
        self.assertTrue(np.any(off_dataset[1]["mask"]))

    def test_off_mode_empty_mask_filtering_is_opt_in(self):
        mask = np.zeros((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)

        unfiltered_dataset = SfmDataset(scene)
        self.assertEqual(len(unfiltered_dataset), 1)
        self.assertFalse(np.any(unfiltered_dataset[0]["mask"]))
        with self.assertRaisesRegex(ValueError, "no images with a valid mask"):
            SfmDataset(scene, filter_empty_masks=True)

        empty_validation_dataset = SfmDataset(
            scene,
            filter_empty_masks=True,
            allow_empty_after_filtering=True,
        )
        self.assertEqual(len(empty_validation_dataset), 0)
        np.testing.assert_array_equal(empty_validation_dataset.indices, np.array([], dtype=np.int64))

    def test_minimum_valid_mask_area_can_filter_every_image(self):
        mask = np.zeros((32, 48), dtype=np.bool_)
        mask[0, 0] = True
        scene, _, _ = self._make_masked_scene(mask)

        with self.assertRaisesRegex(ValueError, "no images with a valid mask"):
            SfmDataset(
                scene,
                filter_empty_masks=True,
                minimum_valid_mask_area=0.01,
            )
        with self.assertRaisesRegex(ValueError, "no images with a valid mask"):
            SfmDataset(
                scene,
                mask_rasterization_mode="bbox",
                minimum_valid_mask_area=0.01,
            )

    def test_mask_rasterization_mode_rejects_mask_shape_mismatch(self):
        mask = np.ones((32, 48), dtype=np.bool_)
        scene, _, _ = self._make_masked_scene(mask)
        wrong_shape = np.ones((31, 48), dtype=np.uint8) * 255
        self.assertTrue(cv2.imwrite(scene.images[0].mask_path, wrong_shape))

        with self.assertRaisesRegex(ValueError, "does not match camera image size"):
            SfmDataset(scene, mask_rasterization_mode="bbox")

    def test_mask_rasterization_argument_validation(self):
        scene, _ = self._make_scene()

        with self.assertRaisesRegex(ValueError, "must be one of"):
            SfmDataset(scene, mask_rasterization_mode="invalid")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "raster_tile_size must be positive"):
            SfmDataset(scene, raster_tile_size=0)
        with self.assertRaisesRegex(ValueError, "raster_context_pixels must be non-negative"):
            SfmDataset(scene, raster_context_pixels=-1)
        with self.assertRaisesRegex(ValueError, "patch_size is not supported"):
            SfmDataset(scene, patch_size=4, mask_rasterization_mode="bbox")
        with self.assertRaisesRegex(ValueError, "return_visible_points is not supported"):
            SfmDataset(scene, return_visible_points=True, mask_rasterization_mode="bbox")
        with self.assertRaisesRegex(ValueError, "requires filter_empty_masks"):
            SfmDataset(scene, minimum_valid_mask_area=0.1)
        with self.assertRaisesRegex(ValueError, "filter_empty_masks must be a bool"):
            SfmDataset(scene, filter_empty_masks=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "allow_empty_after_filtering must be a bool"):
            SfmDataset(scene, allow_empty_after_filtering=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "finite fraction"):
            SfmDataset(scene, mask_rasterization_mode="tiles", minimum_valid_mask_area=1.1)
