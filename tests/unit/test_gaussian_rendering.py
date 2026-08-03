# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from fvdb_reality_capture.enums import CameraModel
from fvdb_reality_capture.radiance_fields._gaussian_rendering import ImageSpaceRenderBackend


class TestImageSpaceMaskAwareTrainingRender(unittest.TestCase):
    @staticmethod
    def _config(mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            sparse_depth_reg=0.0,
            dense_depth_reg=0.0,
            projection_method="analytic",
            near_plane=0.01,
            far_plane=1000.0,
            min_radius_2d=0.0,
            eps_2d=0.3,
            antialias=False,
            tile_size=16,
            mask_rasterization_mode=mode,
        )

    @staticmethod
    def _model(height: int, width: int) -> Mock:
        model = Mock()
        model.device = torch.device("cpu")
        model.num_channels = 3
        projected = object()
        model.project_gaussians_for_images.return_value = projected
        model.render_from_projected_gaussians.return_value = (
            torch.zeros(1, height, width, 3),
            torch.zeros(1, height, width, 1),
        )
        return model

    @staticmethod
    def _camera_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        world_to_camera = torch.eye(4).unsqueeze(0).contiguous()
        intrinsics = torch.tensor([[[100.0, 0.0, 64.0], [0.0, 120.0, 48.0], [0.0, 0.0, 1.0]]]).contiguous()
        camera_models = torch.tensor([int(CameraModel.PINHOLE)], dtype=torch.int32)
        distortion = torch.zeros(1, 12)
        return world_to_camera, intrinsics, camera_models, distortion

    def test_bbox_preserves_full_projection_and_gates_statistics_to_crop_tiles(self):
        crop = (16, 32, 48, 64)
        model = self._model(crop[3], crop[2])
        world_to_camera, intrinsics, camera_models, distortion = self._camera_inputs()

        ImageSpaceRenderBackend().forward_train(
            model=model,
            config=self._config("bbox"),
            world_to_camera_matrices=world_to_camera,
            projection_matrices=intrinsics,
            camera_models=camera_models,
            distortion_coeffs=distortion,
            image_width=128,
            image_height=96,
            sh_degree_to_use=0,
            crop=crop,
        )

        projection_args = model.project_gaussians_for_images.call_args.kwargs
        torch.testing.assert_close(projection_args["projection_matrices"], intrinsics)
        self.assertEqual(projection_args["image_width"], 128)
        self.assertEqual(projection_args["image_height"], 96)
        expected_tile_mask = torch.zeros((1, 6, 8), dtype=torch.bool)
        expected_tile_mask[:, 2:6, 1:4] = True
        self.assertTrue(torch.equal(projection_args["gradient_accumulation_tile_mask"], expected_tile_mask))
        self.assertEqual(projection_args["gradient_accumulation_tile_size"], 16)

        raster_args = model.render_from_projected_gaussians.call_args.kwargs
        self.assertEqual(raster_args["crop_origin_w"], crop[0])
        self.assertEqual(raster_args["crop_origin_h"], crop[1])
        self.assertEqual(raster_args["crop_width"], crop[2])
        self.assertEqual(raster_args["crop_height"], crop[3])
        self.assertIsNone(raster_args["tile_masks"])

    def test_tiles_pass_the_same_mask_to_rasterization_and_statistics(self):
        height, width = 32, 48
        model = self._model(height, width)
        world_to_camera, intrinsics, camera_models, distortion = self._camera_inputs()
        tile_mask = torch.tensor(
            [[[True, False, False], [False, True, False]]],
            dtype=torch.bool,
        )

        ImageSpaceRenderBackend().forward_train(
            model=model,
            config=self._config("tiles"),
            world_to_camera_matrices=world_to_camera,
            projection_matrices=intrinsics,
            camera_models=camera_models,
            distortion_coeffs=distortion,
            image_width=width,
            image_height=height,
            sh_degree_to_use=0,
            crop=(0, 0, width, height),
            raster_tile_mask=tile_mask,
        )

        projection_args = model.project_gaussians_for_images.call_args.kwargs
        torch.testing.assert_close(projection_args["projection_matrices"], intrinsics)
        self.assertEqual(projection_args["image_width"], width)
        self.assertEqual(projection_args["image_height"], height)
        self.assertTrue(torch.equal(projection_args["gradient_accumulation_tile_mask"], tile_mask))
        self.assertEqual(projection_args["gradient_accumulation_tile_size"], 16)

        raster_args = model.render_from_projected_gaussians.call_args.kwargs
        self.assertTrue(torch.equal(raster_args["tile_masks"], tile_mask))


if __name__ == "__main__":
    unittest.main()
