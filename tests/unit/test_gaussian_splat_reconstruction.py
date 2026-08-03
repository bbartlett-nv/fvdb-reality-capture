# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import torch

from fvdb_reality_capture.radiance_fields.gaussian_splat_optimizer import GaussianSplatOptimizer
from fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionConfig,
    _masked_ground_truth,
    _masked_l1_losses,
    _masked_ssim_losses,
    _validate_mask_rasterization_config,
)


class TestMaskedGroundTruth(unittest.TestCase):
    def test_replaces_invalid_pixels_without_mutating_ground_truth(self):
        ground_truth = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
        prediction = ground_truth + 100.0
        valid_mask = torch.tensor([[[True, False, True, False], [False, True, False, True]]])
        original_ground_truth = ground_truth.clone()

        result = _masked_ground_truth(ground_truth, prediction, valid_mask)
        expected = torch.where(valid_mask.unsqueeze(-1), ground_truth, prediction)

        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(ground_truth, original_ground_truth)

    def test_detaches_prediction_at_invalid_pixels(self):
        ground_truth = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3).requires_grad_(True)
        prediction = (ground_truth.detach() + 10.0).requires_grad_(True)
        valid_mask = torch.tensor([[[True, False], [False, True]]])

        result = _masked_ground_truth(ground_truth, prediction, valid_mask)
        result.square().sum().backward()

        expected_gradient = 2.0 * ground_truth.detach() * valid_mask.unsqueeze(-1)
        torch.testing.assert_close(ground_truth.grad, expected_gradient)
        self.assertIsNone(prediction.grad)

    def test_rejects_incompatible_mask_shape(self):
        image = torch.zeros(1, 2, 3, 3)

        with self.assertRaisesRegex(ValueError, "Mask shape"):
            _masked_ground_truth(image, image, torch.ones(2, 3, dtype=torch.bool))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_runs_on_cuda_with_cpu_mask(self):
        ground_truth = torch.arange(12, dtype=torch.float32, device="cuda").reshape(1, 2, 2, 3)
        prediction = ground_truth + 100.0
        valid_mask = torch.tensor([[[True, False], [False, True]]])

        result = _masked_ground_truth(ground_truth, prediction, valid_mask)
        expected = torch.where(valid_mask.cuda().unsqueeze(-1), ground_truth, prediction)

        self.assertEqual(result.device.type, "cuda")
        torch.testing.assert_close(result, expected)


class TestStrictMaskedL1(unittest.TestCase):
    def test_invalid_values_do_not_affect_loss_or_receive_gradients(self):
        target = torch.arange(36, dtype=torch.float32).reshape(1, 3, 4, 3) / 36.0
        target[:, 0, 1] = torch.nan
        target[:, 2, 3] = torch.nan
        valid_mask = torch.tensor([[[True, False, True, True], [False, True, True, False], [True, True, False, False]]])
        prediction = torch.zeros_like(target, requires_grad=True)

        full_loss, valid_loss = _masked_l1_losses(prediction, target, valid_mask)
        self.assertTrue(torch.isfinite(full_loss))
        self.assertTrue(torch.isfinite(valid_loss))
        full_loss.backward()

        invalid_gradients = prediction.grad[~valid_mask.unsqueeze(-1).expand_as(prediction)]
        valid_gradients = prediction.grad[valid_mask.unsqueeze(-1).expand_as(prediction)]
        self.assertTrue(torch.equal(invalid_gradients, torch.zeros_like(invalid_gradients)))
        self.assertGreater(torch.abs(valid_gradients).sum().item(), 0.0)

    def test_empty_mask_returns_connected_zero_loss(self):
        prediction = torch.randn(1, 3, 4, 3, requires_grad=True)
        target = torch.randn_like(prediction)
        mask = torch.zeros(1, 3, 4, dtype=torch.bool)

        full_loss, valid_loss = _masked_l1_losses(prediction, target, mask)
        self.assertEqual(full_loss.item(), 0.0)
        self.assertEqual(valid_loss.item(), 0.0)
        (full_loss + valid_loss).backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))


class TestSceneBoundingBoxClipping(unittest.TestCase):
    def test_uses_integer_keep_indices_and_preserves_boundary_points(self):
        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig()
        runner._training_dataset = SimpleNamespace(
            scene_bbox=np.array([-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        )
        runner._model = SimpleNamespace(
            means=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, -2.0, -3.0],
                    [4.0, 5.0, 6.0],
                    [-1.1, 0.0, 0.0],
                    [0.0, 5.1, 0.0],
                    [0.0, 0.0, -3.1],
                ]
            ),
            num_gaussians=6,
        )

        def filter_gaussians(indices: torch.Tensor) -> None:
            runner._model.means = runner._model.means[indices]
            runner._model.num_gaussians = indices.numel()

        runner._optimizer = SimpleNamespace(filter_gaussians=Mock(side_effect=filter_gaussians))
        runner._logger = Mock()

        runner._clip_gaussians_to_scene_bbox()

        keep_indices = runner._optimizer.filter_gaussians.call_args.args[0]
        self.assertEqual(keep_indices.dtype, torch.long)
        self.assertEqual(keep_indices.device, runner._model.means.device)
        torch.testing.assert_close(keep_indices, torch.tensor([0, 1, 2]))
        self.assertEqual(runner._model.num_gaussians, 3)

    def test_skips_filter_when_all_gaussians_are_inside(self):
        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig()
        runner._training_dataset = SimpleNamespace(
            scene_bbox=np.array([-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        )
        runner._model = SimpleNamespace(means=torch.tensor([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]), num_gaussians=2)
        runner._optimizer = SimpleNamespace(filter_gaussians=Mock())
        runner._logger = Mock()

        runner._clip_gaussians_to_scene_bbox()

        runner._optimizer.filter_gaussians.assert_not_called()

    def test_positive_sigma_clips_rotated_ellipsoid_support(self):
        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(scene_bbox_support_sigma=1.0)
        runner._training_dataset = SimpleNamespace(
            scene_bbox=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        sqrt_half = 2.0**-0.5
        runner._model = SimpleNamespace(
            means=torch.tensor([[0.5, 0.0, 0.0], [0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]]),
            quats=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [sqrt_half, 0.0, 0.0, sqrt_half],
                    [sqrt_half, 0.0, 0.0, sqrt_half],
                ]
            ),
            scales=torch.tensor([[0.4, 0.1, 0.1]]).repeat(4, 1),
            num_gaussians=4,
        )

        def filter_gaussians(indices: torch.Tensor) -> None:
            runner._model.means = runner._model.means[indices]
            runner._model.num_gaussians = indices.numel()

        runner._optimizer = SimpleNamespace(filter_gaussians=Mock(side_effect=filter_gaussians))
        runner._logger = Mock()

        runner._clip_gaussians_to_scene_bbox()

        keep_indices = runner._optimizer.filter_gaussians.call_args.args[0]
        torch.testing.assert_close(keep_indices, torch.tensor([0, 2]))
        self.assertEqual(runner._model.num_gaussians, 2)


class TestPostRefinementSceneBoundingBoxClipping(unittest.TestCase):
    def test_continues_at_refinement_cadence_after_refinement_stops(self):
        minibatch = {
            "camera_to_world": torch.eye(4).unsqueeze(0),
            "world_to_camera": torch.eye(4).unsqueeze(0),
            "projection": torch.eye(3).unsqueeze(0),
            "camera_model": torch.zeros(1, dtype=torch.int32),
            "distortion_coeffs": torch.zeros(1, 12),
            "image": torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
            "image_id": torch.zeros(1, dtype=torch.long),
        }

        class OneBatchTrainloader:
            sampler = (0,)

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter((minibatch,))

        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(
            max_epochs=3,
            max_steps=3,
            refine_start_epoch=0,
            refine_stop_epoch=1,
            refine_every_epoch=1,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,
            remove_gaussians_outside_scene_bbox=True,
        )
        runner._training_dataset = [None]
        runner._validation_dataset = []
        runner._model = SimpleNamespace(num_gaussians=1)
        runner._optimizer = Mock()
        runner._optimizer.regularization_loss.return_value = torch.zeros(())
        runner._pose_adjust_model = None
        runner._pose_adjust_optimizer = None
        runner._pose_adjust_scheduler = None
        runner._start_step = 2
        runner._global_step = 2
        runner._viz_update_interval_epochs = 1
        runner._log_interval_steps = 1_000
        runner._dense_depth_is_relative = False
        runner._render_backend = Mock()
        runner._render_backend.forward_train.return_value = SimpleNamespace(
            image=torch.zeros(1, 16, 16, 3, requires_grad=True)
        )
        runner._writer = Mock()
        runner._viz_scene = None
        runner._logger = Mock()
        runner.device = torch.device("cpu")

        def mock_ssim(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return 1.0 - torch.mean((image - target) ** 2)

        events = []
        runner._optimizer.step.side_effect = lambda: events.append("step")
        with (
            patch("torch.utils.data.DataLoader", return_value=OneBatchTrainloader()),
            patch("fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction.ssim", side_effect=mock_ssim),
            patch.object(runner, "_clip_gaussians_to_scene_bbox", side_effect=lambda: events.append("clip")),
        ):
            runner.optimize(show_progress=False)

        self.assertEqual(events, ["clip", "step", "clip"])
        runner._optimizer.refine.assert_not_called()


class TestPostRefinementPruningAndScaleFreeze(unittest.TestCase):
    @staticmethod
    def _make_runner(
        *,
        start_step: int,
        global_step: int,
        prune_after_stop: bool = True,
        freeze_after_stop: bool = True,
    ) -> tuple[
        GaussianSplatReconstruction,
        list[tuple[int, bool]],
        list[tuple[int, bool]],
        list[int],
    ]:
        minibatch = {
            "camera_to_world": torch.eye(4).unsqueeze(0),
            "world_to_camera": torch.eye(4).unsqueeze(0),
            "projection": torch.eye(3).unsqueeze(0),
            "camera_model": torch.zeros(1, dtype=torch.int32),
            "distortion_coeffs": torch.zeros(1, 12),
            "image": torch.zeros(1, 16, 16, 3, dtype=torch.uint8),
            "image_id": torch.zeros(1, dtype=torch.long),
        }

        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(
            max_epochs=4,
            max_steps=4,
            refine_start_epoch=0,
            refine_stop_epoch=2,
            refine_every_epoch=1,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,
            prune_after_refinement_stop=prune_after_stop,
            freeze_scales_after_refinement_stop=freeze_after_stop,
        )
        runner._training_dataset = [None]
        runner._validation_dataset = []
        runner._model = SimpleNamespace(
            num_gaussians=1,
            means=torch.tensor([[0.4, 0.5, 0.6]], requires_grad=True),
            log_scales=torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True),
        )
        runner._optimizer = Mock(spec=GaussianSplatOptimizer)
        runner._optimizer.regularization_loss.return_value = torch.zeros(())
        runner._pose_adjust_model = None
        runner._pose_adjust_optimizer = None
        runner._pose_adjust_scheduler = None
        runner._start_step = start_step
        runner._global_step = global_step
        runner._viz_update_interval_epochs = 1
        runner._log_interval_steps = 1_000
        runner._dense_depth_is_relative = False
        runner._render_backend = Mock()
        runner._writer = Mock()
        runner._viz_scene = None
        runner._logger = Mock()
        runner.device = torch.device("cpu")

        step_observations: list[tuple[int, bool]] = []
        render_observations: list[tuple[int, bool]] = []
        prune_steps: list[int] = []

        def render(**_kwargs):
            render_observations.append((runner._global_step, runner._model.log_scales.requires_grad))
            value = runner._model.means.sum() + runner._model.log_scales.sum()
            return SimpleNamespace(
                image=value.reshape(1, 1, 1, 1).expand(1, 16, 16, 3),
                alpha=torch.ones((1, 16, 16, 1)),
                depth=None,
            )

        def prune(use_scale_threshold: bool = True):
            assert use_scale_threshold
            prune_steps.append(runner._global_step)

            # Emulate filter_gaussians replacing parameter tensors while preserving their gradients.
            old_scales = runner._model.log_scales
            replacement = old_scales.detach().clone().requires_grad_(True)
            if old_scales.grad is not None:
                replacement.grad = old_scales.grad.detach().clone()
            runner._model.log_scales = replacement
            return {"num_deleted": 0}

        def step():
            step_observations.append((runner._global_step, runner._model.log_scales.grad is None))

        def zero_grad(*_args, **_kwargs):
            runner._model.means.grad = None
            runner._model.log_scales.grad = None

        runner._render_backend.forward_train.side_effect = render
        runner._optimizer.prune.side_effect = prune
        runner._optimizer.step.side_effect = step
        runner._optimizer.zero_grad.side_effect = zero_grad
        runner._test_minibatch = minibatch
        return runner, step_observations, render_observations, prune_steps

    @staticmethod
    def _run(runner: GaussianSplatReconstruction) -> None:
        class OneBatchTrainloader:
            sampler = (0,)

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter((runner._test_minibatch,))

        def mock_ssim(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return 1.0 - torch.mean((image - target) ** 2)

        with (
            patch("torch.utils.data.DataLoader", return_value=OneBatchTrainloader()),
            patch("fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction.ssim", side_effect=mock_ssim),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            runner.optimize(show_progress=False)

    def test_prunes_on_cadence_and_freezes_from_first_stop_step(self):
        runner, step_observations, render_observations, prune_steps = self._make_runner(start_step=0, global_step=0)

        self._run(runner)

        self.assertEqual(step_observations, [(0, False), (1, False), (2, True), (3, True)])
        self.assertEqual(render_observations, [(0, True), (1, True), (2, False), (3, False)])
        self.assertEqual(prune_steps, [2, 3, 4])
        runner._optimizer.refine.assert_called_once_with()
        self.assertEqual(
            [call.kwargs for call in runner._optimizer.prune.call_args_list],
            [{"use_scale_threshold": True}] * 3,
        )

    def test_resume_past_stop_prunes_and_freezes_immediately(self):
        # A real resumed runner starts global_step at zero and advances it while skipping to start_step.
        runner, step_observations, render_observations, prune_steps = self._make_runner(start_step=3, global_step=0)

        self._run(runner)

        self.assertEqual(step_observations, [(3, True)])
        self.assertEqual(render_observations, [(3, False)])
        self.assertEqual(prune_steps, [3, 4])
        runner._optimizer.refine.assert_not_called()

    def test_defaults_leave_post_stop_scale_gradient_and_pruning_unchanged(self):
        runner, step_observations, render_observations, prune_steps = self._make_runner(
            start_step=3,
            global_step=0,
            prune_after_stop=False,
            freeze_after_stop=False,
        )

        self._run(runner)

        self.assertEqual(step_observations, [(3, False)])
        self.assertEqual(render_observations, [(3, True)])
        self.assertEqual(prune_steps, [])
        runner._optimizer.prune.assert_not_called()

    def test_prune_only_cadence_does_not_wait_for_refinement_start(self):
        runner, step_observations, render_observations, prune_steps = self._make_runner(
            start_step=0,
            global_step=0,
        )
        runner._cfg.max_epochs = 1
        runner._cfg.max_steps = 1
        runner._cfg.refine_start_epoch = 10
        runner._cfg.refine_stop_epoch = 0

        self._run(runner)

        self.assertEqual(step_observations, [(0, True)])
        self.assertEqual(render_observations, [(0, False)])
        self.assertEqual(prune_steps, [0, 1])
        runner._optimizer.refine.assert_not_called()

    def test_prunes_before_checkpoint_and_deduplicates_final_same_step_scan(self):
        class EmptyTrainloader:
            sampler: tuple[()] = ()

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter(())

        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_stop_epoch=0,
            eval_at_percent=[],
            save_at_percent=[100],
            optimize_camera_poses=False,
            prune_after_refinement_stop=True,
        )
        runner._training_dataset = [None]
        runner._validation_dataset = []
        runner._model = SimpleNamespace(num_gaussians=1)
        runner._optimizer = Mock(spec=GaussianSplatOptimizer)
        runner._pose_adjust_optimizer = None
        runner._start_step = 0
        runner._global_step = 1
        runner._viz_update_interval_epochs = 1
        runner._logger = Mock()
        runner._writer = Mock()

        events = []
        runner._optimizer.prune.side_effect = lambda **_kwargs: events.append("prune")
        runner._writer.save_checkpoint.side_effect = lambda *args: events.append("checkpoint")
        runner._writer.save_ply.side_effect = lambda *args: events.append("ply")
        with (
            patch("torch.utils.data.DataLoader", return_value=EmptyTrainloader()),
            patch.object(runner, "state_dict", return_value={}),
            patch.object(GaussianSplatReconstruction, "reconstruction_metadata", new_callable=PropertyMock) as metadata,
        ):
            metadata.return_value = {}
            runner.optimize(show_progress=False)

        self.assertEqual(events, ["prune", "checkpoint", "ply"])
        runner._optimizer.prune.assert_called_once_with(use_scale_threshold=True)


class TestFinalSceneBoundingBoxClipping(unittest.TestCase):
    def test_clips_at_end_when_refinement_never_runs(self):
        class EmptyTrainloader:
            sampler: tuple[()] = ()

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter(())

        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(
            max_epochs=1,
            refine_start_epoch=10_000,
            refine_stop_epoch=10_000,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,
            remove_gaussians_outside_scene_bbox=True,
        )
        runner._training_dataset = [None]
        runner._optimizer = Mock()
        runner._pose_adjust_optimizer = None
        runner._start_step = 0
        runner._global_step = 0
        runner._viz_update_interval_epochs = 1
        runner._logger = Mock()

        with (
            patch("torch.utils.data.DataLoader", return_value=EmptyTrainloader()),
            patch.object(runner, "_clip_gaussians_to_scene_bbox") as clip,
        ):
            runner.optimize(show_progress=False)

        clip.assert_called_once_with()

    def test_clips_before_post_refinement_checkpoint_serialization(self):
        class EmptyTrainloader:
            sampler: tuple[()] = ()

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter(())

        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(
            max_epochs=1,
            eval_at_percent=[],
            save_at_percent=[100],
            refine_stop_epoch=0,
            optimize_camera_poses=False,
            remove_gaussians_outside_scene_bbox=True,
        )
        runner._training_dataset = [None]
        runner._model = SimpleNamespace(num_gaussians=1)
        runner._optimizer = Mock()
        runner._pose_adjust_optimizer = None
        runner._start_step = 0
        runner._global_step = 1
        runner._viz_update_interval_epochs = 1
        runner._logger = Mock()
        runner._writer = Mock()

        events = []
        runner._writer.save_checkpoint.side_effect = lambda *args: events.append("checkpoint")
        runner._writer.save_ply.side_effect = lambda *args: events.append("ply")
        with (
            patch("torch.utils.data.DataLoader", return_value=EmptyTrainloader()),
            patch.object(runner, "state_dict", return_value={}),
            patch.object(GaussianSplatReconstruction, "reconstruction_metadata", new_callable=PropertyMock) as metadata,
            patch.object(runner, "_clip_gaussians_to_scene_bbox", side_effect=lambda: events.append("clip")),
        ):
            metadata.return_value = {}
            runner.optimize(show_progress=False)

        self.assertEqual(events, ["clip", "checkpoint", "ply", "clip"])


class TestMaskRasterizationConfig(unittest.TestCase):
    def test_off_preserves_existing_configuration_freedom(self):
        config = GaussianSplatReconstructionConfig(
            mask_rasterization_mode="off",
            render_backend="world_space",
            batch_size=2,
            crops_per_image=3,
            ignore_masks=True,
            sparse_depth_reg=1.0,
        )

        _validate_mask_rasterization_config(config, num_channels=7)

    def test_rejects_unsupported_combinations(self):
        cases = (
            ({"render_backend": "world_space"}, "image_space"),
            ({"optimize_camera_poses": True}, "optimize_camera_poses=False"),
            ({"batch_size": 2}, "batch_size=1"),
            ({"crops_per_image": 2}, "crops_per_image=1"),
            ({"ignore_masks": True}, "ignore_masks"),
            ({"sparse_depth_reg": 1.0}, "without depth"),
            ({"dense_depth_reg": 1.0}, "without depth"),
            ({"tile_size": 0}, "tile_size > 0"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                values = {"mask_rasterization_mode": "bbox", "optimize_camera_poses": False, **overrides}
                config = GaussianSplatReconstructionConfig(**values)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_mask_rasterization_config(config, num_channels=3)

    def test_rejects_non_rgb_model_and_unknown_mode(self):
        config = GaussianSplatReconstructionConfig(mask_rasterization_mode="tiles", optimize_camera_poses=False)
        with self.assertRaisesRegex(ValueError, "RGB models only"):
            _validate_mask_rasterization_config(config, num_channels=4)

        config.mask_rasterization_mode = "unknown"  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "Unknown mask_rasterization_mode"):
            _validate_mask_rasterization_config(config, num_channels=3)

    def test_rejects_nonfinite_or_negative_support_sigma(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                config = GaussianSplatReconstructionConfig(scene_bbox_support_sigma=value)
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    _validate_mask_rasterization_config(config, num_channels=3)

    def test_off_allows_minimum_mask_area_when_masks_are_honored(self):
        config = GaussianSplatReconstructionConfig(
            mask_rasterization_mode="off",
            minimum_valid_mask_area=0.25,
            ignore_masks=False,
        )

        _validate_mask_rasterization_config(config, num_channels=3)

    def test_minimum_mask_area_rejects_ignored_masks(self):
        config = GaussianSplatReconstructionConfig(
            mask_rasterization_mode="off",
            minimum_valid_mask_area=0.25,
            ignore_masks=True,
        )

        with self.assertRaisesRegex(ValueError, "cannot be used when ignore_masks=True"):
            _validate_mask_rasterization_config(config, num_channels=3)


class TestFilteredDatasetConstruction(unittest.TestCase):
    def test_filtered_indices_drive_schedules_and_validation_may_become_empty(self):
        class FakeDataset:
            def __init__(self, scene, indices):
                self.sfm_scene = scene
                self.indices = np.asarray(indices, dtype=np.int64)

            def __len__(self):
                return len(self.indices)

        scene = SimpleNamespace(num_images=4)
        train_dataset = FakeDataset(scene, [3])
        validation_dataset = FakeDataset(scene, [])
        model = SimpleNamespace(num_gaussians=123)
        optimizer = Mock()
        optimizer_config = Mock()
        optimizer_config.make_optimizer.return_value = optimizer
        config = GaussianSplatReconstructionConfig(
            max_epochs=7,
            optimize_camera_poses=False,
            minimum_valid_mask_area=0.25,
        )

        module = "fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction"
        with (
            patch(f"{module}.SfmDataset", side_effect=[train_dataset, validation_dataset]) as dataset_type,
            patch.object(GaussianSplatReconstruction, "_init_model", return_value=model),
            patch(f"{module}.make_render_backend", return_value=Mock()),
            patch.object(GaussianSplatReconstruction, "__init__", return_value=None) as init,
        ):
            GaussianSplatReconstruction.from_sfm_scene(
                scene,
                writer=Mock(),
                config=config,
                optimizer_config=optimizer_config,
                use_every_n_as_val=2,
                device="cpu",
            )

        first_dataset_call, second_dataset_call = dataset_type.call_args_list
        self.assertTrue(first_dataset_call.kwargs["filter_empty_masks"])
        self.assertEqual(first_dataset_call.kwargs["minimum_valid_mask_area"], 0.25)
        self.assertTrue(second_dataset_call.kwargs["filter_empty_masks"])
        self.assertTrue(second_dataset_call.kwargs["allow_empty_after_filtering"])
        optimizer.reset_learning_rates_and_decay.assert_called_once_with(batch_size=1, expected_steps=7)
        np.testing.assert_array_equal(init.call_args.kwargs["train_indices"], np.array([3]))
        np.testing.assert_array_equal(init.call_args.kwargs["val_indices"], np.array([], dtype=np.int64))

    def test_legacy_resume_uses_filtered_schedule_but_original_pose_table_indices(self):
        class FakeDataset:
            sfm_scene = None
            indices = np.array([1, 2], dtype=np.int64)

            def __len__(self):
                return len(self.indices)

        scene = SimpleNamespace(num_images=5)
        config = GaussianSplatReconstructionConfig(
            optimize_camera_poses=True,
            minimum_valid_mask_area=0.25,
        )
        source_pose_weights = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        state_dict = {
            "magic": GaussianSplatReconstruction._magic,
            "version": GaussianSplatReconstruction.version,
            "step": 4,
            "config": config.__dict__.copy(),
            "sfm_scene": {},
            "model": {},
            "optimizer": {"config": {}},
            "train_indices": [0, 1, 2],
            "val_indices": [],
            "num_training_poses": 3,
            "pose_adjust_model": {"pose_embeddings.weight": source_pose_weights},
            "pose_adjust_optimizer": {},
            "pose_adjust_scheduler": {},
        }
        model = SimpleNamespace()
        optimizer = Mock()
        pose_adjust_model = SimpleNamespace(
            pose_embeddings=SimpleNamespace(weight=torch.zeros(5, 6)),
            load_state_dict=Mock(),
        )
        pose_adjust_optimizer = Mock()
        pose_adjust_scheduler = Mock()

        module = "fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction"
        with (
            patch(f"{module}.SfmDataset", return_value=FakeDataset()) as dataset_type,
            patch.object(
                GaussianSplatReconstruction, "_move_state_tensors_to_device", side_effect=lambda value, _: value
            ),
            patch(f"{module}.GaussianSplat3d.from_state_dict", return_value=model),
            patch(f"{module}.BaseGaussianSplatOptimizer.from_state_dict", return_value=optimizer),
            patch.object(
                GaussianSplatReconstruction,
                "_make_pose_optimizer",
                return_value=(pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler),
            ) as make_pose_optimizer,
            patch(f"{module}.make_render_backend", return_value=Mock()),
            patch.object(GaussianSplatReconstruction, "__init__", return_value=None) as init,
        ):
            GaussianSplatReconstruction.from_state_dict(
                state_dict,
                override_sfm_scene=scene,
                device="cpu",
            )

        self.assertTrue(dataset_type.call_args.kwargs["filter_empty_masks"])
        self.assertEqual(dataset_type.call_args.kwargs["minimum_valid_mask_area"], 0.25)
        make_pose_optimizer.assert_called_once()
        self.assertEqual(make_pose_optimizer.call_args.args[3], 2)
        np.testing.assert_array_equal(init.call_args.kwargs["train_indices"], np.array([1, 2]))
        torch.testing.assert_close(pose_adjust_model.pose_embeddings.weight[:3], source_pose_weights)
        pose_adjust_model.load_state_dict.assert_not_called()
        pose_adjust_optimizer.load_state_dict.assert_not_called()
        pose_adjust_scheduler.load_state_dict.assert_not_called()


class TestMaskRasterizationTrainingLoopWiring(unittest.TestCase):
    def test_bbox_uses_full_sensor_crop_and_area_scaled_image_objective(self):
        crop = (16, 16, 48, 32)
        full_height, full_width = 100, 120
        image_loss_scale = np.float32((crop[2] * crop[3]) / (full_height * full_width))
        minibatch = {
            "camera_to_world": torch.eye(4).unsqueeze(0),
            "world_to_camera": torch.eye(4).unsqueeze(0),
            "projection": torch.eye(3).unsqueeze(0),
            "camera_model": torch.zeros(1, dtype=torch.int32),
            "distortion_coeffs": torch.zeros(1, 12),
            "image": torch.full((1, crop[3], crop[2], 3), 255, dtype=torch.uint8),
            "mask": torch.ones((1, crop[3], crop[2]), dtype=torch.bool),
            "image_id": torch.zeros(1, dtype=torch.long),
            "full_image_size": torch.tensor([[full_height, full_width]], dtype=torch.int64),
            "raster_crop": torch.tensor([crop], dtype=torch.int64),
            "image_loss_scale": torch.tensor([image_loss_scale]),
        }

        class OneBatchTrainloader:
            sampler = (0,)

            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter((minibatch,))

        prediction = torch.zeros((1, crop[3], crop[2], 3), requires_grad=True)
        runner = GaussianSplatReconstruction.__new__(GaussianSplatReconstruction)
        runner._cfg = GaussianSplatReconstructionConfig(
            max_epochs=1,
            max_steps=1,
            mask_rasterization_mode="bbox",
            ssim_lambda=0.25,
            refine_start_epoch=10,
            refine_stop_epoch=10,
            refine_every_epoch=1,
            eval_at_percent=[],
            save_at_percent=[],
            optimize_camera_poses=False,
        )
        runner._training_dataset = [None]
        runner._validation_dataset = []
        runner._model = SimpleNamespace(num_gaussians=1)
        runner._optimizer = Mock()
        runner._optimizer.regularization_loss.return_value = torch.zeros(())
        runner._pose_adjust_model = None
        runner._pose_adjust_optimizer = None
        runner._pose_adjust_scheduler = None
        runner._start_step = 0
        runner._global_step = 0
        runner._viz_update_interval_epochs = 1
        runner._log_interval_steps = 1
        runner._dense_depth_is_relative = False
        runner._render_backend = Mock()
        runner._render_backend.forward_train.return_value = SimpleNamespace(
            image=prediction,
            alpha=torch.ones((1, crop[3], crop[2], 1)),
            depth=None,
        )
        runner._writer = Mock()
        runner._viz_scene = None
        runner._logger = Mock()
        runner.device = torch.device("cpu")

        def mock_masked_ssim_losses(
            image: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            loss = torch.mean((image - target) ** 2)
            return loss, loss

        with (
            patch("torch.utils.data.DataLoader", return_value=OneBatchTrainloader()),
            patch(
                "fvdb_reality_capture.radiance_fields.gaussian_splat_reconstruction._masked_ssim_losses",
                side_effect=mock_masked_ssim_losses,
            ),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            runner.optimize(show_progress=False)

        render_kwargs = runner._render_backend.forward_train.call_args.kwargs
        self.assertEqual(render_kwargs["image_width"], full_width)
        self.assertEqual(render_kwargs["image_height"], full_height)
        self.assertEqual(render_kwargs["crop"], crop)
        self.assertIsNone(render_kwargs["raster_tile_mask"])

        expected_scaled_loss = float(image_loss_scale)
        logged_metrics = {
            call.args[1]: call.args[2] for call in runner._writer.log_metric.call_args_list if len(call.args) == 3
        }
        self.assertAlmostEqual(logged_metrics["reconstruct/l1loss"], expected_scaled_loss, places=7)
        self.assertAlmostEqual(logged_metrics["reconstruct/ssimloss"], expected_scaled_loss, places=7)
        self.assertAlmostEqual(logged_metrics["reconstruct/loss"], expected_scaled_loss, places=7)
        self.assertAlmostEqual(logged_metrics["reconstruct/l1loss_valid"], 1.0, places=7)
        self.assertAlmostEqual(logged_metrics["reconstruct/ssimloss_valid"], 1.0, places=7)
        self.assertAlmostEqual(logged_metrics["reconstruct/image_loss_valid"], 1.0, places=7)
        self.assertIsNotNone(prediction.grad)


class TestMaskRasterizationLossParity(unittest.TestCase):
    @staticmethod
    def _image_objective(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        area_scale: float = 1.0,
    ) -> torch.Tensor:
        l1loss, _ = _masked_l1_losses(prediction, target, valid_mask)
        ssimloss, _ = _masked_ssim_losses(prediction, target, valid_mask)
        return torch.lerp(l1loss * area_scale, ssimloss * area_scale, 0.2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by the production SSIM kernel")
    def test_ten_pixel_context_and_area_scale_match_full_frame_loss_and_gradient(self):
        generator = torch.Generator().manual_seed(1234)
        base_prediction = torch.rand((1, 96, 128, 3), generator=generator)
        target = torch.rand((1, 96, 128, 3), generator=generator)
        valid_mask = torch.zeros((1, 96, 128), dtype=torch.bool)
        valid_mask[:, 35:55, 43:61] = True
        base_prediction = base_prediction.cuda()
        target = target.cuda()
        valid_mask = valid_mask.cuda()

        full_prediction = base_prediction.clone().requires_grad_(True)
        full_loss = self._image_objective(full_prediction, target, valid_mask)
        full_loss.backward()

        # The valid-mask bounds expanded by 10 SSIM pixels are aligned outward
        # to 16-pixel tiles: x=[32, 80), y=[16, 80).
        x0, y0, x1, y1 = 32, 16, 80, 80
        crop_prediction = base_prediction[:, y0:y1, x0:x1].clone().requires_grad_(True)
        crop_target = target[:, y0:y1, x0:x1]
        crop_mask = valid_mask[:, y0:y1, x0:x1]
        area_scale = ((y1 - y0) * (x1 - x0)) / (96 * 128)
        crop_loss = self._image_objective(crop_prediction, crop_target, crop_mask, area_scale)
        crop_loss.backward()

        torch.testing.assert_close(crop_loss, full_loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            crop_prediction.grad,
            full_prediction.grad[:, y0:y1, x0:x1],
            atol=2e-7,
            rtol=2e-5,
        )
        outside_gradient = full_prediction.grad.clone()
        outside_gradient[:, y0:y1, x0:x1] = 0
        torch.testing.assert_close(outside_gradient, torch.zeros_like(outside_gradient), atol=2e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
