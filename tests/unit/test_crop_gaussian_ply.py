# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import sys
import tempfile
import unittest

import torch
from fvdb_reality_capture import GaussianSplat3d

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from scripts.crop_gaussian_ply import (
    _default_output_path,
    _validate_paths,
    compute_percentile_crop,
    crop_ply,
    validate_percentile_ranges,
)


class TestCropGaussianPly(unittest.TestCase):
    def test_compute_percentile_crop_uses_inclusive_axis_bounds(self):
        coordinates = torch.arange(5, dtype=torch.float32)
        means = coordinates[:, None].repeat(1, 3)

        result = compute_percentile_crop(
            means,
            x_percentiles=(25.0, 75.0),
            y_percentiles=(25.0, 75.0),
            z_percentiles=(25.0, 75.0),
        )

        torch.testing.assert_close(result.lower_bounds, torch.ones(3))
        torch.testing.assert_close(result.upper_bounds, torch.full((3,), 3.0))
        torch.testing.assert_close(result.mask, torch.tensor([False, True, True, True, False]))
        self.assertEqual(result.retained_count, 3)
        self.assertEqual(result.removed_count, 2)

    def test_non_finite_centers_are_removed_and_excluded_from_quantiles(self):
        means = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [float("nan"), 3.0, 3.0],
                [float("inf"), 4.0, 4.0],
            ]
        )

        result = compute_percentile_crop(means)

        torch.testing.assert_close(result.lower_bounds, torch.zeros(3))
        torch.testing.assert_close(result.upper_bounds, torch.full((3,), 2.0))
        torch.testing.assert_close(result.mask, torch.tensor([True, True, True, False, False]))
        self.assertEqual(result.finite_count, 3)

    def test_sample_stride_only_changes_bound_estimation(self):
        coordinates = torch.arange(7, dtype=torch.float32)
        means = coordinates[:, None].repeat(1, 3)

        result = compute_percentile_crop(means, sample_stride=2)

        self.assertEqual(result.sample_stride, 2)
        self.assertEqual(result.sampled_count, 4)
        self.assertEqual(result.input_count, 7)
        self.assertEqual(result.retained_count, 7)

    def test_quantile_samples_are_automatically_capped_without_cropping_default_axes(self):
        coordinates = torch.arange(10, dtype=torch.float32)
        means = coordinates[:, None].repeat(1, 3)

        result = compute_percentile_crop(means, max_quantile_samples=3)

        self.assertEqual(result.sample_stride, 4)
        self.assertEqual(result.sampled_count, 3)
        torch.testing.assert_close(result.lower_bounds, torch.zeros(3))
        torch.testing.assert_close(result.upper_bounds, torch.full((3,), 9.0))
        self.assertEqual(result.retained_count, 10)

    def test_max_quantile_samples_must_be_positive(self):
        with self.assertRaises(ValueError):
            compute_percentile_crop(torch.zeros(1, 3), max_quantile_samples=0)

    def test_mixed_endpoint_bounds_use_full_extrema(self):
        means = torch.tensor(
            [
                [5.0, 5.0, 0.0],
                [-10.0, 6.0, 1.0],
                [6.0, 7.0, 2.0],
                [7.0, 100.0, 3.0],
                [8.0, 8.0, 4.0],
                [9.0, -10.0, 5.0],
            ]
        )

        result = compute_percentile_crop(
            means,
            x_percentiles=(0.0, 50.0),
            y_percentiles=(50.0, 100.0),
            z_percentiles=(0.0, 100.0),
            sample_stride=2,
        )

        torch.testing.assert_close(result.lower_bounds, torch.tensor([-10.0, 7.0, 0.0]))
        torch.testing.assert_close(result.upper_bounds, torch.tensor([6.0, 100.0, 5.0]))

    def test_percentile_validation_rejects_reversed_and_out_of_range_values(self):
        invalid_ranges = [
            ((50.0, 49.0), (0.0, 100.0), (0.0, 100.0)),
            ((-1.0, 99.0), (0.0, 100.0), (0.0, 100.0)),
            ((0.0, 101.0), (0.0, 100.0), (0.0, 100.0)),
        ]
        for ranges in invalid_ranges:
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                validate_percentile_ranges(*ranges)

    def test_path_validation_is_non_destructive_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            input_path = directory_path / "scene.ply"
            input_path.touch()

            with self.assertRaises(ValueError):
                _validate_paths(input_path, input_path, force=True)

            output_path = directory_path / "scene_cropped.ply"
            output_path.touch()
            with self.assertRaises(FileExistsError):
                _validate_paths(input_path, output_path, force=False)

            _validate_paths(input_path, output_path, force=True)
            _validate_paths(input_path, output_path, force=False, will_write=False)

    def test_crop_ply_round_trip_preserves_attributes_and_metadata(self):
        device = "cuda:0"
        means = torch.arange(5, dtype=torch.float32, device=device)[:, None].repeat(1, 3)
        num_gaussians = means.shape[0]
        model = GaussianSplat3d.from_tensors(
            means=means,
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_gaussians, 1),
            log_scales=torch.arange(num_gaussians * 3, dtype=torch.float32, device=device).reshape(num_gaussians, 3),
            logit_opacities=torch.arange(num_gaussians, dtype=torch.float32, device=device),
            sh0=torch.arange(num_gaussians * 3, dtype=torch.float32, device=device).reshape(num_gaussians, 1, 3),
            shN=torch.empty(num_gaussians, 0, 3, device=device),
        )
        metadata = {"scene_name": "synthetic crop test", "camera_count": 7}

        with tempfile.TemporaryDirectory() as directory:
            directory_path = pathlib.Path(directory)
            input_path = directory_path / "scene.ply"
            output_path = directory_path / "scene_cropped.ply"
            model.save_ply(input_path, metadata)

            result = crop_ply(
                input_path,
                output_path,
                x_percentiles=(25.0, 75.0),
                device=device,
            )
            cropped, cropped_metadata = GaussianSplat3d.from_ply(output_path, device=device)

        self.assertEqual(result.retained_count, 3)
        torch.testing.assert_close(cropped.means, model.means[1:4])
        torch.testing.assert_close(cropped.quats, model.quats[1:4])
        torch.testing.assert_close(cropped.log_scales, model.log_scales[1:4])
        torch.testing.assert_close(cropped.logit_opacities, model.logit_opacities[1:4])
        torch.testing.assert_close(cropped.sh0, model.sh0[1:4])
        torch.testing.assert_close(cropped.shN, model.shN[1:4])
        self.assertEqual(cropped_metadata, metadata)

    def test_default_output_is_alongside_input(self):
        input_path = pathlib.Path("/datasets/example.scene.ply")
        self.assertEqual(
            _default_output_path(input_path),
            pathlib.Path("/datasets/example.scene_cropped.ply"),
        )


if __name__ == "__main__":
    unittest.main()
