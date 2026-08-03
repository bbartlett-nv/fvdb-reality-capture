# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import logging
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal

import numpy as np
import torch
import torch.nn.functional as nnf
import torch.utils.data
from torch.utils import _pytree
import tqdm
from fvdb.utils.metrics import masked_ssim, psnr, ssim
from fvdb.viz import Scene
from scipy.spatial import cKDTree  # type: ignore

from fvdb_reality_capture.sfm_scene import SfmScene
from fvdb_reality_capture.tools import export_splats_to_usd

from ._gaussian_rendering import RenderBackend, make_render_backend
from ._gaussian_splat_viz import gaussian_splat_to_view_data
from ._private.lpips import LPIPSLoss
from ._private.utils import crop_image_batch
from .camera_pose_adjust import CameraPoseAdjustment
from .gaussian_splat_dataset import SfmDataset
from .gaussian_splatting import GaussianSplat3d
from .gaussian_splat_optimizer import (
    BaseGaussianSplatOptimizer,
    GaussianSplatOptimizer,
    GaussianSplatOptimizerConfig,
)
from .gaussian_splat_reconstruction_writer import (
    GaussianSplatReconstructionBaseWriter,
    GaussianSplatReconstructionWriter,
)

MASK_RASTERIZATION_CONTEXT_PIXELS = 10
_KNN_QUERY_BATCH_SIZE = 262_144


def _validate_mask_rasterization_config(
    config: "GaussianSplatReconstructionConfig", num_channels: int | None = None
) -> None:
    """Reject combinations whose loss/raster semantics are not yet supported."""
    mode = config.mask_rasterization_mode
    if mode not in ("off", "tiles", "bbox"):
        raise ValueError(f"Unknown mask_rasterization_mode {mode!r}; expected one of: off, tiles, bbox.")
    if not np.isfinite(config.scene_bbox_support_sigma) or config.scene_bbox_support_sigma < 0.0:
        raise ValueError("scene_bbox_support_sigma must be finite and nonnegative.")
    if config.minimum_valid_mask_area is not None and not 0.0 <= config.minimum_valid_mask_area <= 1.0:
        raise ValueError("minimum_valid_mask_area must be a fraction in [0, 1] or None.")
    if config.minimum_valid_mask_area is not None and config.ignore_masks:
        raise ValueError("minimum_valid_mask_area cannot be used when ignore_masks=True.")
    if mode == "off":
        return
    if config.optimize_camera_poses:
        raise ValueError(
            "Mask-aware rasterization requires optimize_camera_poses=False because masks are tied to fixed poses."
        )

    if config.render_backend != "image_space":
        raise ValueError("Mask-aware rasterization currently requires render_backend='image_space'.")
    if config.batch_size != 1:
        raise ValueError("Mask-aware rasterization currently requires batch_size=1.")
    if config.crops_per_image != 1:
        raise ValueError("Mask-aware rasterization currently requires crops_per_image=1.")
    if config.ignore_masks:
        raise ValueError("Mask-aware rasterization cannot be combined with ignore_masks=True.")
    if config.sparse_depth_reg > 0.0 or config.dense_depth_reg > 0.0:
        raise ValueError("Mask-aware rasterization currently supports RGB image losses only, without depth losses.")
    if config.tile_size <= 0:
        raise ValueError("Mask-aware rasterization requires tile_size > 0.")
    if num_channels is not None and num_channels != 3:
        raise ValueError(f"Mask-aware rasterization currently supports RGB models only; got {num_channels} channels.")


def _composite_background(image: torch.Tensor, alphas: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Composite a BHWC image without broadcasting the alpha channel."""
    alpha_without_channel = alphas[..., 0]
    return torch.stack(
        [
            image[..., channel] + background[:, channel].reshape(-1, 1, 1) * (1.0 - alpha_without_channel)
            for channel in range(image.shape[-1])
        ],
        dim=-1,
    )


def _masked_ground_truth(
    ground_truth: torch.Tensor, prediction: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Replace invalid ground-truth pixels with a detached prediction.

    The fixed-shape ``torch.where`` avoids boolean advanced indexing, which is not
    reliable on every PrivateUse1 backend (including Torch-DGX).
    """
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground truth and prediction must have the same shape; got {ground_truth.shape} and {prediction.shape}."
        )
    if valid_mask.shape != ground_truth.shape[:-1]:
        raise ValueError(
            f"Mask shape must match the non-channel image dimensions; got {valid_mask.shape} for "
            f"image shape {ground_truth.shape}."
        )

    valid_rgb = (
        valid_mask.to(device=ground_truth.device, dtype=torch.bool).unsqueeze(-1).expand_as(ground_truth).contiguous()
    )
    return torch.where(valid_rgb, ground_truth, prediction.detach())


def _validate_masked_image_inputs(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Validate image/mask shapes and move the mask to the image device."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {prediction.shape} and {target.shape}.")
    if prediction.ndim != 4:
        raise ValueError(f"Expected BHWC images, got shape {prediction.shape}.")
    if valid_mask.shape != prediction.shape[:-1]:
        raise ValueError(f"Mask shape {valid_mask.shape} does not match image shape {prediction.shape}.")
    return valid_mask.to(device=prediction.device, dtype=torch.bool)


def _masked_l1_losses(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-image- and valid-normalized strict masked L1 losses."""
    valid = _validate_masked_image_inputs(prediction, target, valid_mask)
    valid_channels = valid.unsqueeze(-1).expand_as(prediction).contiguous()
    zero = prediction.new_zeros(())
    safe_prediction = torch.where(valid_channels, prediction, zero)
    safe_target = torch.where(valid_channels, target, zero)
    absolute_error = torch.abs(safe_prediction - safe_target)
    full_image_loss = absolute_error.mean()
    valid_loss = absolute_error.sum() / valid_channels.sum().clamp_min(1)
    return full_image_loss, valid_loss


def _masked_ssim_losses(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-image- and valid-normalized strict masked SSIM losses."""
    valid = _validate_masked_image_inputs(prediction, target, valid_mask)
    ssim_map = masked_ssim(
        prediction.permute(0, 3, 1, 2).contiguous(),
        target.permute(0, 3, 1, 2).contiguous(),
        valid.unsqueeze(1),
        reduction="none",
    )
    valid_centers = valid.unsqueeze(1).expand_as(ssim_map).contiguous()
    loss_map = torch.where(valid_centers, 1.0 - ssim_map, torch.zeros_like(ssim_map))
    full_image_loss = loss_map.mean()
    valid_loss = loss_map.sum() / valid_centers.sum().clamp_min(1)
    return full_image_loss, valid_loss


def _masked_psnr(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute valid-pixel PSNR for BHWC images."""
    valid = _validate_masked_image_inputs(prediction, target, valid_mask)
    valid_channels = valid.unsqueeze(-1).expand_as(prediction).contiguous()
    zero = prediction.new_zeros(())
    safe_prediction = torch.where(valid_channels, prediction, zero)
    safe_target = torch.where(valid_channels, target, zero)
    squared_error = torch.square(safe_prediction - safe_target)
    mse = squared_error.sum() / valid_channels.sum().clamp_min(1)
    return -10.0 * torch.log10(mse)


def _scale_shift_invariant_l1(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Scale-and-shift invariant L1 loss between ``pred`` and ``target``, per image.

    Each image in the batch is independently normalized by its own median and
    mean-absolute-deviation over its valid pixels (Ranftl et al., MiDaS); the L1 of the
    residual is then averaged over valid pixels and over the batch.

    Because the valid-pixel count differs per batch element, normalization is computed
    per element (a short Python loop over the batch dimension rather than a single
    vectorized reduction). Batch elements with no valid pixels contribute zero.

    Args:
        pred (torch.Tensor): Predicted depth, shape ``(B, H, W)``.
        target (torch.Tensor): Target depth, shape ``(B, H, W)``.
        valid (torch.Tensor): Boolean validity mask, shape ``(B, H, W)``.
        eps (float): Lower bound on the MAD denominator to avoid division by zero on
            constant-depth (or near-empty) crops.

    Returns:
        loss (torch.Tensor): Scalar L1 loss in normalized depth units, averaged over the
            batch. Returns zero when no pixels are valid anywhere.
    """
    per_image_losses = []
    for b in range(pred.shape[0]):
        vb = valid[b]
        if not vb.any():
            per_image_losses.append(pred.new_zeros(()))
            continue
        p = pred[b][vb]
        t = target[b][vb]
        p_med = torch.median(p)
        t_med = torch.median(t)
        p_mad = (p - p_med).abs().mean().clamp_min(eps)
        t_mad = (t - t_med).abs().mean().clamp_min(eps)
        p_hat = (p - p_med) / p_mad
        t_hat = (t - t_med) / t_mad
        per_image_losses.append((p_hat - t_hat).abs().mean())
    return torch.stack(per_image_losses).mean()


@dataclass
class GaussianSplatReconstructionConfig:
    """
    Configuration parameters for reconstructing a Gaussian splat radiance field from posed images.

    See :class:`GaussianSplatReconstruction` for details on how these parameters are used.
    """

    seed: int = 42
    """
    A random seed for reproducibility.

    Default: 42 (the meaning of life, the universe, and everything).
    """

    #
    # Training duration and evaluation parameters
    #

    max_epochs: int = 200
    """
    The maximum number of optimization epochs, *i.e.*, the number of times each image in the dataset will be visited.

    An epoch is defined as one full pass through the dataset. If you have a dataset with 100 images and a batch
    size of 10, then one epoch corresponds to 10 steps.

    Default: 200
    """

    max_steps: int | None = None
    """
    The maximum number of optimization steps. If set, this overrides the number of steps calculated from `max_epochs` and the dataset size.

    You shouldn't use this parameter unless you have a specific reason to do so.

    Default: None
    """

    eval_at_percent: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    """
    Percentage of the total optimization epochs at which to perform evaluation on the validation set.

    For example, if `eval_at_percent` is set to `[10, 50, 100]` and `max_epochs` is set to `200`, then evaluation will be
    performed after 20, 100, and 200 epochs.

    Default: [10, 20, 30, 40, 50, 75, 100]
    """

    save_at_percent: List[int] = field(default_factory=lambda: [20, 100])
    """
    Percentage of the total optimization epochs at which to save model checkpoints.

    For example, if `save_at_percent` is set to `[50, 100]` and `max_epochs` is set to `200`, then checkpoints will be saved after 100 and 200 epochs.

    Default: [20, 100]
    """

    #
    # Gaussian Optimization Parameters
    #

    batch_size: int = 1
    """
    Batch size for optimization. Each step of optimization will compute losses on :obj:`batch_size` images. Note that
    learning rates are scaled automatically based on the batch size.

    Default: ``1``
    """

    crops_per_image: int = 1
    """
    Number of crops to use per image during reconstruction. If you're using very large images, you can set this to a value greater than 1
    to run the forward pass on crops and accumulate gradients. This can help reduce memory usage.

    Default: ``1`` (no cropping, use full images).
    """

    sh_degree: int = 3
    """
    Maximum degree of spherical harmonics to use for each Gaussian's view-dependent color.
    Higher degrees allow for more complex view-dependent effects, but increase memory usage and computation time.

    Default: ``3``
    """

    increase_sh_degree_every_epoch: int = 5
    """
    When reconstructing a Gaussian splat radiance field, we start by only optimizing the diffuse (degree 0) spherical harmonics coefficients
    per Gaussian, and progressively increase the degree of spherical harmonics used every :obj:`increase_sh_degree_every_epoch` epochs
    until we reach :obj:`sh_degree`. This helps stabilize optimization in the early stages of optimization.

    Default: ``5``
    """

    ssim_lambda: float = 0.2
    """
    Weight for SSIM loss. Reconstruction aims to minimize
    the `Structural Similarity Index Measure (SSIM) <https://en.wikipedia.org/wiki/Structural_similarity_index_measure>`_
    between rendered images with the radiance field and ground truth images. This weight applies to the SSIM loss term.

    Default: ``0.2``
    """

    lpips_net: Literal["vgg", "alex"] = "alex"
    """
    During evaluation, we compute the `Learned Perceptual Image Patch Similarity (LPIPS) <https://arxiv.org/abs/1801.03924>`_ metric
    as a measure of quality of the reconstruction. This parameter controls which network architecture is used for the LPIPS metric.

    Default: ``"alex"`` meaning the `AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ architecture.
    """

    sparse_depth_reg: float = 0.0
    """
    Weight for sparse depth loss. If you have sparse depth measurements for some pixels in the images, you can set this weight
    to a value greater than 0 to encourage the reconstruction to match those depth measurements.
    Default: ``0.0`` (no sparse depth loss).
    """

    dense_depth_attribute: str | None = None
    """
    Name of a :class:`~fvdb_reality_capture.sfm_scene.DepthMapAttribute` registered on the scene to use as a
    per-pixel ground-truth depth target. If ``None`` and :attr:`dense_depth_reg` is non-zero, a ``ValueError`` is raised.

    Default: ``None`` (no dense depth supervision).
    """

    dense_depth_reg: float = 0.0
    """
    Weight on the dense depth loss term. The dense loss coexists with :attr:`sparse_depth_reg`; both may be enabled
    simultaneously. The form of the loss is selected by the :attr:`~DepthMapAttribute.scale` of the referenced attribute:

    - :class:`DepthScale.METRIC` --- direct masked L1 between rendered depth (alpha-corrected) and target depth,
      both in scene units.
    - :class:`DepthScale.RELATIVE` --- scale-and-shift invariant L1 (each side normalized by its own per-image
      median and mean-absolute-deviation, then L1; cf. Ranftl et al., MiDaS). Use this for monocular relative-depth
      predictions whose absolute scale and offset are arbitrary.

    .. note::
        For :class:`DepthScale.METRIC` the loss is computed in **scene units**, so a good value for
        ``dense_depth_reg`` depends on the scale of the scene and must be retuned when the scene units change.
        For :class:`DepthScale.RELATIVE` the loss is unitless (each image is median/MAD-normalized), so a value
        chosen for one scene transfers across scenes. This differs from :attr:`sparse_depth_reg`, whose loss is
        always median-normalized and therefore scene-scale independent.

    Default: ``0.0`` (no dense depth loss).
    """

    random_bkgd: bool = False
    """
    Whether to render images with the radiance field against a background of random values during optimization.
    This discourages the model from using transparency to minimize loss.

    Default: ``False``
    """

    refine_start_epoch: int = 3
    """
    At which epoch to start refining the Gaussians by inserting and deleting Gaussians based on their contribution to the optimization.
    *e.g.* If this value is 3, the first refinement will occur at the start of epoch 3.

    Default: ``3``
    """

    refine_stop_epoch: int = 100
    """
    At which epoch to stop refining the Gaussians by inserting and deleting Gaussians based on their contribution to the optimization.

    Default: ``100``
    """

    refine_every_epoch: float = 0.65
    """
    How often to refine Gaussians during optimization, in terms of epochs.
    For example, a value of 0.65 means refinement occurs approximately every 0.65 epochs.

    Default: ``0.65``
    """

    prune_after_refinement_stop: bool = False
    """
    If set to ``True``, continue deleting low-opacity and oversized Gaussians at the configured refinement cadence
    after insertion and splitting stop. Pruning is independent of the maximum-Gaussian budget and also runs before
    post-refinement checkpoints and the final export.

    This option is supported by the classic :class:`GaussianSplatOptimizer` only.

    Default: ``False``
    """

    freeze_scales_after_refinement_stop: bool = False
    """
    If set to ``True``, stop updating Gaussian scales once insertion and splitting stop. Other Gaussian parameters
    continue to optimize normally, and post-refinement pruning can still remove oversized Gaussians.

    Default: ``False``
    """

    ignore_masks: bool = False
    """
    If set to ``True``, then ignore any masks in the data and treat all pixels as valid during optimization.

    Default: ``False``
    """

    mask_rasterization_mode: Literal["off", "tiles", "bbox"] = "off"
    """
    Reduce rasterization work using training-image masks.

    ``"bbox"`` decodes each image and transfers/rasterizes only a tile-aligned bounding box around its valid mask,
    including the context required to preserve the existing 11-by-11 SSIM objective. ``"tiles"`` keeps full-size
    images but skips raster tiles that cannot affect the masked loss. ``"off"`` preserves the existing behavior.

    The optimized modes currently require image-space RGB rendering, batch size 1, one crop per image, masks enabled,
    and no sparse or dense depth loss.

    Default: ``"off"``
    """

    minimum_valid_mask_area: float | None = None
    """
    Optional minimum valid mask coverage, expressed as a fraction of the full sensor area.

    Whenever masks are honored, images with zero valid pixels are filtered in every rasterization mode.
    Set this value in [0, 1] to filter additional low-coverage masked images, or None to disable
    thresholding beyond empty masks. This option cannot be combined with ``ignore_masks=True``.

    Default: None
    """

    remove_gaussians_outside_scene_bbox: bool = False
    """
    If set to ``True``, Gaussians whose configured support falls outside the scene bounding box will be removed at the
    configured refinement cadence throughout optimization, including after refinement stops. Checkpoints saved after
    refinement stops and the final export are clipped as well.

    Default: ``False``
    """

    #
    scene_bbox_support_sigma: float = 0.0
    """
    Sigma extent of each rotated Gaussian ellipsoid that must remain inside the scene bounding box
    when remove_gaussians_outside_scene_bbox is enabled. Zero preserves center-only clipping;
    positive values remove Gaussians whose support crosses the boundary.

    Default: 0.0
    """

    # Pose optimization parameters
    #

    optimize_camera_poses: bool = True
    """
    If set to ``True``, optimize camera poses during reconstruction. This can help improve the quality of the reconstruction if the initial poses are not accurate.

    Default: ``True``
    """

    pose_opt_lr: float = 1e-5
    """
    Learning rate for camera pose optimization.

    Default: ``1e-5``
    """

    pose_opt_reg: float = 1e-6
    """
    Weight for regularization of camera pose optimization. This encourages small changes to the initial camera poses.

    The pose regularization loss is defined as :math:`L_{pose}` = \\frac{1}{M} \\sum_j ||\\Delta R_j||^2 + ||\\Delta t_j||^2`,
    *i.e.* the Frobenius norm of the change in rotation and translation for each of the ``M`` camera poses in the dataset.

    Default: ``1e-6``
    """

    pose_opt_lr_decay: float = 1.0
    """
    Learning rate decay factor for camera pose optimization (will decay to this fraction of initial lr).

    Default: ``1.0`` (no decay).
    """

    pose_opt_start_epoch: int = 0
    """
    At which epoch to start optimizing camera poses.

    Default: ``0`` (start from beginning of optimization).
    """

    pose_opt_stop_epoch: int = max_epochs
    """
    At which epoch to stop optimizing camera poses.

    Default: ``max_epochs`` (optimize poses for the entire duration of optimization).
    """

    pose_opt_init_std: float = 1e-4
    """
    Standard deviation for the normal distribution used to initialize the embeddings for camera pose optimization.

    Default: ``1e-4``
    """

    #
    # Gaussian Rendering Parameters
    #

    # Near plane clipping distance
    near_plane: float = 0.01
    """
    Near plane clipping distance when rendering the Gaussians.

    Default: ``0.01``
    """

    far_plane: float = 1e10
    """
    Far plane clipping distance when rendering the Gaussians.

    Default: ``1e10``
    """

    min_radius_2d: float = 0.0
    """
    Minimum screen space radius (in pixels) below which Gaussians are ignored after projection.

    Default: ``0.0``
    """

    eps_2d: float = 0.3
    """
    Amount of padding (in pixels) to add to the screen space bounding box of each Gaussian when determining which pixels it affects.

    Default: ``0.3``
    """

    antialias: bool = False
    """
    Whether to use anti-aliasing when rendering the Gaussians.

    Default: ``False``
    """

    tile_size: int = 16
    """
    Tile size (in pixels) to use when rendering the Gaussians.
    You should generally leave this at the default value unless you have a specific reason to change it.

    Default: ``16``
    """

    render_backend: Literal["image_space", "world_space"] = "image_space"
    """
    Rendering path to use during reconstruction.

    Default: ``"image_space"``
    """

    projection_method: Literal["auto", "analytic", "unscented"] = "auto"
    """
    Projection implementation selector for FVDB camera models.

    Default: ``"auto"``
    """


class GaussianSplatReconstruction:
    """
    Engine for reconstructing a Gaussian splat radiance field from posed images in an :class:`~fvdb_reality_capture.sfm_scene.sfm_scene.SfmScene`.

    This class implements the reconstruction algorithm using a :class:`fvdb_reality_capture.GaussianSplat3d` model and a differentiable rendering pipeline.

    The reconstruction process optimizes the parameters of the Gaussian splats to minimize the difference between rendered images and the input images.
    The optimization process can be configured using a :class:`GaussianSplatReconstructionConfig` instance, and the underlying
    :class:`fvdb_reality_capture.GaussianSplat3d` model can be customized as well.

    The reconstruction can also optionally optimize camera poses if they are not accurate, using a simple pose adjustment model which stores a per-camera
    embedding which is decoded into a small change in rotation and translation for each camera.

    To create a :class:`GaussianSplatReconstruction` instance, use the :meth:`from_sfm_scene` class method, which initializes the model and optimizer
    from an :class:`~fvdb_reality_capture.sfm_scene.sfm_scene.SfmScene` and a :class:`GaussianSplatReconstructionConfig`.

    You can configure logging and checkpointing during optimization process using an instance of :class:`~fvdb_reality_capture.radiance_fields.GaussianSplatReconstructionBaseWriter`.
    By default, this class uses a :class:`~fvdb_reality_capture.radiance_fields.GaussianSplatReconstructionWriter` which logs metrics, images, and checkpoints to a directory.

    You can also visualize the optimization process using an optional :class:`fvdb.viz.Scene` instance, which can display
    the current state of the Gaussian splat radiance field interactively in a web browser or notebook.

    The reconstruction process is started by calling the :meth:`reconstruct` method, which runs the optimization loop.

    To get the reconstructed model, use the :meth:`model` attribute, which is a :class:`fvdb_reality_capture.GaussianSplat3d` instance.

    You can also get a dictionary of metadata about the reconstruction using the :meth:`reconstruction_metadata` attribute.
    This metadata is useful for downstream tasks such as extracting meshes or exporting to USDZ.

    The state of the reconstruction can be saved and loaded using the :meth:`state_dict` and :meth:`from_state_dict` methods.
    These methods allow you to pause and resume reconstructions from checkpoints.
    """

    version = "0.1.0"

    _magic = "GaussianSplattingCheckpoint"

    __PRIVATE__ = object()

    @staticmethod
    def _move_state_tensors_to_device(obj: Any, device: torch.device) -> Any:
        """
        Move all tensors in a checkpoint payload to the requested device.

        Checkpoints can be loaded on CPU and later restored onto CUDA via ``from_state_dict``.
        We detach before moving so floating-point tensors become leaf tensors again on the
        destination device; otherwise optimizer reconstruction can fail when PyTorch rejects
        non-leaf tensors in parameter groups.
        """

        def _move_leaf_tensor(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value

            moved = value.detach().to(device)
            if moved.is_floating_point() or moved.is_complex():
                moved.requires_grad_(value.requires_grad)
            return moved

        # Apply the tensor move/releaf logic to every leaf in the nested checkpoint payload.
        return _pytree.tree_map(_move_leaf_tensor, obj)

    @classmethod
    def from_sfm_scene(
        cls,
        sfm_scene: SfmScene,
        writer: GaussianSplatReconstructionBaseWriter = GaussianSplatReconstructionWriter(
            run_name=None, save_path=None
        ),
        viz_scene: Scene | None = None,
        config: GaussianSplatReconstructionConfig = GaussianSplatReconstructionConfig(),
        optimizer_config: GaussianSplatOptimizerConfig = GaussianSplatOptimizerConfig(),
        use_every_n_as_val: int = -1,
        viz_update_interval_epochs: float = 10,
        log_interval_steps: int = 10,
        device: str | torch.device = "cuda",
    ):
        """
        Create a :class:`GaussianSplatReconstruction` instance from an :class:`~fvdb_reality_capture.sfm_scene.sfm_scene.SfmScene`, used to reconstruct
        a 3D Gaussian Splat radiance field from posed images. The reconstruction process and optimizer can be
        configured using the ``config`` (see :class:`GaussianSplatReconstructionConfig`) and
        ``optimizer_config`` (see :class:`GaussianSplatOptimizerConfig`) parameters, though the defaults
        should produce acceptable results.

        You can also configure logging and checkpointing during the reconstruction process using an instance of
        :class:`~fvdb_reality_capture.radiance_fields.GaussianSplatReconstructionBaseWriter`. By default, this class uses a
        :class:`~fvdb_reality_capture.radiance_fields.GaussianSplatReconstructionWriter` which logs metrics, images, and checkpoints to a directory.

        You can interactively visualize the state of the current reconstruction using an optional :class:`fvdb.viz.Scene` instance, which
        can display the current Gaussian splat radiance field in a web browser or notebook.

        Args:
            sfm_scene (SfmScene): The Structure-from-Motion scene containing images and camera poses.
            config (GaussianSplatReconstructionConfig): Configuration for the reconstruction process.
            optimizer_config (GaussianSplatOptimizerConfig): Configuration for the optimizer. The type of the config determines the type of optimizer used,
                either :class:`GaussianSplatOptimizerConfig` for the classic insertion/splitting/deletion strategy, or
                :class:`GaussianSplatOptimizerMCMCConfig` for the MCMC strategy.
            writer (GaussianSplatReconstructionBaseWriter): Writer instance to handle logging metrics, saving images, checkpoints, PLY, files,
                and other results.
            viz_scene (Scene | None): Optional :class:`fvdb.viz.Scene` instance for visualizing optimization progress. If None,
                no visualization is performed.
            use_every_n_as_val (int): Use every n-th image as a validation image. Default of ``-1``
                means no validation images are used.
            viz_update_interval_epochs (float): Interval in epochs at which to update the visualization if ``viz_scene`` is not None.
                An epoch is one full pass through the dataset.
            log_interval_steps (int): Interval (in steps) to log metrics to the ``writer``.
            device (str | torch.device): Device to run the reconstruction on.
        Returns:
            gaussian_splat_reconstruction (GaussianSplatReconstruction): An :class:`GaussianSplatReconstruction` instance ready to reconstruct the scene.
        """

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        _validate_mask_rasterization_config(config)

        train_indices, val_indices = cls._make_index_splits(sfm_scene, use_every_n_as_val)
        filter_training_masks = not config.ignore_masks
        train_dataset = SfmDataset(
            sfm_scene,
            train_indices,
            filter_empty_masks=filter_training_masks,
            minimum_valid_mask_area=config.minimum_valid_mask_area if filter_training_masks else None,
        )
        filter_validation_masks = not config.ignore_masks and len(val_indices) > 0
        val_dataset = SfmDataset(
            sfm_scene,
            val_indices,
            filter_empty_masks=filter_validation_masks,
            minimum_valid_mask_area=config.minimum_valid_mask_area if filter_validation_masks else None,
            allow_empty_after_filtering=True,
        )

        logger.info(
            f"Created training and validation datasets with {len(train_dataset)} training images and {len(val_dataset)} validation images."
        )
        if config.optimize_camera_poses and len(val_dataset) > 0:
            logger.warning(
                "Camera pose optimization is enabled with a holdout set. Pose deltas are learned from training images "
                "only, while validation images continue to use their original camera poses, so validation metrics may "
                "not reflect the train-time objective."
            )

        # Initialize model
        model = GaussianSplatReconstruction._init_model(config, optimizer_config, device, train_dataset)
        logger.info(f"Model initialized with {model.num_gaussians:,} Gaussians")

        render_backend = make_render_backend(config.render_backend)

        # Initialize optimizer
        max_steps = config.max_epochs * len(train_dataset)
        optimizer = optimizer_config.make_optimizer(model=model, sfm_scene=train_dataset.sfm_scene)
        optimizer.reset_learning_rates_and_decay(batch_size=config.batch_size, expected_steps=max_steps)

        if config.batch_size > 1:
            num_steps_per_epoch = int(np.ceil(len(train_dataset) / config.batch_size))
        else:
            num_steps_per_epoch = len(train_dataset)

        # Initialize pose optimizer
        pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = None, None, None
        if config.optimize_camera_poses:
            pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = cls._make_pose_optimizer(
                config, device, sfm_scene.num_images, num_steps_per_epoch
            )

        return GaussianSplatReconstruction(
            model=model,
            sfm_scene=sfm_scene,
            optimizer=optimizer,
            config=config,
            train_indices=train_dataset.indices,
            val_indices=val_dataset.indices,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            writer=writer,
            start_step=0,
            viz_scene=viz_scene,
            log_interval_steps=log_interval_steps,
            viz_update_interval_epochs=viz_update_interval_epochs,
            render_backend=render_backend,
            _private=GaussianSplatReconstruction.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        override_sfm_scene: SfmScene | None = None,
        override_use_every_n_as_val: int | None = None,
        writer: GaussianSplatReconstructionBaseWriter = GaussianSplatReconstructionWriter(
            run_name=None, save_path=None
        ),
        viz_scene: Scene | None = None,
        viz_update_interval_epochs: float = 1.0,
        log_interval_steps: int = 10,
        device: str | torch.device = "cuda",
    ):
        """
        Load a :class:`GaussianSplatReconstruction` instance from a state dictionary (extracted with the :meth:`state_dict` method).
        This will restore the model, optimizer, and configuration.
        You can optionally override the :class:`~fvdb_reality_capture.sfm_scene.sfm_scene.SfmScene` and the train/validation split (via the ``override_use_every_n_as_val`` parameter).
        This is useful for resuming reconstruction on a different dataset or with a different train/validation split.

        Args:
            state_dict (dict): State dictionary containing the model, optimizer, and configuration state. Generated by
                the :meth:`state_dict` method.
            override_sfm_scene (SfmScene | None): Optional :class:`~fvdb_reality_capture.sfm_scene.sfm_scene.SfmScene` to use instead of the one in the state_dict.
            override_use_every_n_as_val (int | None): If specified, will override the train/validation split using this value.
                Default of None means to use the train/validation split from the state_dict.
            writer (GaussianSplatReconstructionBaseWriter): :class:`~fvdb_reality_capture.radiance_fields.GaussianSplatReconstructionBaseWriter` instance to handle
                logging metrics, saving images, checkpoints, PLY, files, and other results.
            viz_scene (Scene | None): Optional :class:`fvdb.viz.Scene` instance for visualizing optimization progress. If ``None``, no
                visualization is performed.
            viz_update_interval_epochs (float): Interval in epochs at which to update the visualization if ``viz_scene`` is not None.
                An epoch is one full pass through the dataset.
            log_interval_steps (int): Interval in steps to log metrics to the ``writer``.
            device (str | torch.device): Device to run the reconstruction on.
        """
        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")
        device = torch.device(device)

        # Ensure this is a valid state dict
        if state_dict.get("magic", "") != cls._magic:
            raise ValueError(f"State dict has invalid magic value.")

        # Ensure the state_dict version matches the current version of this class
        if state_dict.get("version", "") != cls.version:
            raise ValueError(
                f"Checkpoint version {state_dict.get('version', '')} does not match current version {cls.version}."
            )

        # Check that all required keys in the state dict are present and their values have the correct types
        if not isinstance(state_dict.get("step", None), int):
            raise ValueError("Checkpoint step is missing or invalid.")
        if not isinstance(state_dict.get("config", None), dict):
            raise ValueError("Checkpoint config is missing or invalid.")
        if not isinstance(state_dict.get("sfm_scene", None), dict):
            raise ValueError("Checkpoint SfM scene is missing or invalid.")
        if not isinstance(state_dict.get("model", None), dict):
            raise ValueError("Checkpoint model state is missing or invalid.")
        if not isinstance(state_dict.get("optimizer", None), dict):
            raise ValueError("Checkpoint optimizer state is missing or invalid.")
        if not isinstance(state_dict.get("train_indices", None), (list, np.ndarray, torch.Tensor)):
            raise ValueError("Checkpoint train indices are missing or invalid.")
        if not isinstance(state_dict.get("val_indices", None), (list, np.ndarray, torch.Tensor)):
            raise ValueError("Checkpoint val indices are missing or invalid.")
        if "num_training_poses" not in state_dict:
            raise ValueError("Checkpoint is missing num_training_poses key.")
        if "pose_adjust_model" not in state_dict:
            raise ValueError("Checkpoint is missing pose_adjust_model key.")
        if "pose_adjust_optimizer" not in state_dict:
            raise ValueError("Checkpoint is missing pose_adjust_optimizer key.")
        if "pose_adjust_scheduler" not in state_dict:
            raise ValueError("Checkpoint is missing pose_adjust_scheduler key.")

        global_step = state_dict["step"]
        config_state = dict(state_dict["config"])
        config_state.pop("undistort_images", None)
        legacy_initial_opacity = config_state.pop("initial_opacity", None)
        legacy_initial_covariance_scale = config_state.pop("initial_covariance_scale", None)
        config = GaussianSplatReconstructionConfig(**config_state)

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        sfm_scene: SfmScene
        if override_sfm_scene is not None:
            sfm_scene = override_sfm_scene
            logger.info("Using override SfM scene instead of the one from the checkpoint.")
        else:
            sfm_scene = SfmScene.from_state_dict(state_dict["sfm_scene"])
        if override_use_every_n_as_val is not None:
            train_indices, val_indices = cls._make_index_splits(sfm_scene, override_use_every_n_as_val)
        else:
            train_indices = np.array(state_dict["train_indices"], dtype=int)
            val_indices = np.array(state_dict["val_indices"], dtype=int)

        pose_table_train_indices = train_indices
        filter_training_masks = not config.ignore_masks
        filtered_training_dataset = SfmDataset(
            sfm_scene,
            train_indices,
            filter_empty_masks=filter_training_masks,
            minimum_valid_mask_area=config.minimum_valid_mask_area if filter_training_masks else None,
        )
        train_indices = filtered_training_dataset.indices

        model_state = cls._move_state_tensors_to_device(state_dict["model"], device)
        model = GaussianSplat3d.from_state_dict(model_state)
        optimizer_state = dict(state_dict["optimizer"])
        optimizer_config_state = dict(optimizer_state.get("config", {}))
        if "initial_opacity" not in optimizer_config_state and legacy_initial_opacity is not None:
            optimizer_config_state["initial_opacity"] = legacy_initial_opacity
        if "initial_covariance_scale" not in optimizer_config_state and legacy_initial_covariance_scale is not None:
            optimizer_config_state["initial_covariance_scale"] = legacy_initial_covariance_scale
        optimizer_state["config"] = optimizer_config_state
        optimizer_state = cls._move_state_tensors_to_device(optimizer_state, device)
        optimizer = BaseGaussianSplatOptimizer.from_state_dict(model, optimizer_state)
        num_pose_entries = state_dict["num_training_poses"]
        pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = None, None, None

        if state_dict["pose_adjust_model"] is not None:
            if not isinstance(state_dict.get("pose_adjust_model", None), dict):
                raise ValueError("Checkpoint pose adjustment model state is invalid.")
            if not isinstance(state_dict.get("pose_adjust_optimizer", None), dict):
                raise ValueError("Checkpoint pose adjustment optimizer state is invalid.")
            if not isinstance(state_dict.get("pose_adjust_scheduler", None), dict):
                raise ValueError("Checkpoint pose adjustment scheduler state is invalid.")
            pose_adjust_model_state = cls._move_state_tensors_to_device(state_dict["pose_adjust_model"], device)
            pose_adjust_optimizer_state = cls._move_state_tensors_to_device(state_dict["pose_adjust_optimizer"], device)
            pose_adjust_scheduler_state = cls._move_state_tensors_to_device(state_dict["pose_adjust_scheduler"], device)
            if config.batch_size > 1:
                num_steps_per_epoch = int(np.ceil(len(train_indices) / config.batch_size))
            else:
                num_steps_per_epoch = len(train_indices)
            pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler = cls._make_pose_optimizer(
                config, device, sfm_scene.num_images, num_steps_per_epoch
            )
            pose_embedding_weights = pose_adjust_model_state["pose_embeddings.weight"]
            if pose_embedding_weights.shape[0] == sfm_scene.num_images:
                pose_adjust_model.load_state_dict(pose_adjust_model_state)
                pose_adjust_optimizer.load_state_dict(pose_adjust_optimizer_state)
                pose_adjust_scheduler.load_state_dict(pose_adjust_scheduler_state)
            elif pose_embedding_weights.shape[0] == len(pose_table_train_indices) and num_pose_entries == len(
                pose_table_train_indices
            ):
                logger.warning(
                    "Loading legacy checkpoint with training-local pose IDs. Remapping pose deltas to scene-global IDs "
                    "and reinitializing pose optimizer state."
                )
                with torch.no_grad():
                    pose_adjust_model.pose_embeddings.weight.zero_()
                    pose_adjust_model.pose_embeddings.weight[
                        torch.as_tensor(pose_table_train_indices, dtype=torch.long, device=device)
                    ] = pose_embedding_weights.to(device)
            else:
                raise ValueError(
                    "Checkpoint pose adjustment table size does not match either the full scene or the training split."
                )

        return GaussianSplatReconstruction(
            model=model,
            sfm_scene=sfm_scene,
            optimizer=optimizer,
            config=config,
            train_indices=train_indices,
            val_indices=val_indices,
            pose_adjust_model=pose_adjust_model,
            pose_adjust_optimizer=pose_adjust_optimizer,
            pose_adjust_scheduler=pose_adjust_scheduler,
            writer=writer,
            start_step=global_step,
            viz_scene=viz_scene,
            log_interval_steps=log_interval_steps,
            viz_update_interval_epochs=viz_update_interval_epochs,
            render_backend=make_render_backend(config.render_backend),
            _private=GaussianSplatReconstruction.__PRIVATE__,
        )

    def __init__(
        self,
        model: GaussianSplat3d,
        sfm_scene: SfmScene,
        optimizer: BaseGaussianSplatOptimizer,
        config: GaussianSplatReconstructionConfig,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        pose_adjust_model: CameraPoseAdjustment | None,
        pose_adjust_optimizer: torch.optim.Adam | None,
        pose_adjust_scheduler: torch.optim.lr_scheduler.ExponentialLR | None,
        writer: GaussianSplatReconstructionBaseWriter,
        start_step: int,
        viz_scene: Scene | None,
        log_interval_steps: int,
        viz_update_interval_epochs: float,
        render_backend: RenderBackend,
        _private: object | None = None,
    ) -> None:
        """
        Initialize the Runner with the provided configuration, model, optimizer, datasets, and paths.

        .. note::

            This constructor should only be called by the :meth:`from_sfm_scene` or :meth:`from_state_dict` methods.

        Args:
            model (GaussianSplat3d): The Gaussian Splatting model to optimize.
            sfm_scene (SfmScene): The Structure-from-Motion scene.
            optimizer (GaussianSplatOptimizer | None): The optimizer for the model.
            config (Config): Configuration object containing model parameters.
            train_indices (np.ndarray): The indices for the training set.
            val_indices (np.ndarray): The indices for the validation set.
            pose_adjust_model (CameraPoseAdjustment | None): The camera pose adjustment model, if used
            pose_adjust_optimizer (torch.optim.Adam | None): The optimizer for camera pose adjustment, if used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The learning rate scheduler
                for camera pose adjustment, if used.
            writer (GaussianSplatReconstructionBaseWriter): Writer instance to handle saving images, ply files,
                and other results.
            start_step (int): The step to start optimization from (useful for resuming optimization from a checkpoint).
            viz_scene (Scene | None): The :class:`fvdb.viz.Scene` instance to use for this run.
            log_interval_steps (int): Interval (in steps) at which to log metrics during optimization.
            viz_update_interval_epochs (float): Interval (in epochs) at which to update the :class:`fvdb.viz.Scene` with new results if a ``viz_scene`` is provided.
            _private (object | None): Private object to ensure this class is only initialized through :meth:`from_sfm_scene` or :meth:`from_state_dict`.
        """
        if _private is not GaussianSplatReconstruction.__PRIVATE__:
            raise ValueError("Runner should only be initialized through `from_sfm_scene` or `from_state_dict`.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._cfg = config
        self._model = model
        self._optimizer = optimizer
        self._pose_adjust_model = pose_adjust_model
        self._pose_adjust_optimizer = pose_adjust_optimizer
        self._pose_adjust_scheduler = pose_adjust_scheduler
        self._start_step = start_step
        self._viz_update_interval_epochs = viz_update_interval_epochs
        self._render_backend = render_backend

        _validate_mask_rasterization_config(config, model.num_channels)
        if config.prune_after_refinement_stop and not isinstance(optimizer, GaussianSplatOptimizer):
            raise ValueError("prune_after_refinement_stop is supported only by the classic GaussianSplatOptimizer.")
        self._sfm_scene = sfm_scene

        # Resolve dense-depth supervision: which attribute to load. The comparison form
        # (direct L1 vs scale-shift invariant) is keyed off the attribute's DepthScale at
        # loss-evaluation time, not configured separately.
        dense_depth_load: list[str] = []
        # Resolve the loss form (direct L1 vs scale-shift invariant) once, here, so the
        # training loop is a plain boolean check with no per-step import.
        self._dense_depth_is_relative: bool = False
        if self.config.dense_depth_reg > 0.0:
            if self.config.dense_depth_attribute is None:
                raise ValueError(
                    "dense_depth_reg > 0 requires dense_depth_attribute to name a DepthMapAttribute "
                    "registered on the scene."
                )
            from fvdb_reality_capture.sfm_scene import DepthMapAttribute, DepthScale

            try:
                attr = sfm_scene.get_attribute(self.config.dense_depth_attribute)
            except KeyError:
                raise ValueError(
                    f"dense_depth_attribute '{self.config.dense_depth_attribute}' is not registered on the "
                    f"scene. Available attributes: {sorted(sfm_scene.attributes.keys())}."
                ) from None
            if not isinstance(attr, DepthMapAttribute):
                raise TypeError(
                    f"Scene attribute '{self.config.dense_depth_attribute}' is "
                    f"{type(attr).__name__}, expected DepthMapAttribute."
                )
            dense_depth_load = [self.config.dense_depth_attribute]
            self._dense_depth_is_relative = attr.scale == DepthScale.RELATIVE

        filter_training_masks = not self.config.ignore_masks
        self._training_dataset = SfmDataset(
            sfm_scene=sfm_scene,
            dataset_indices=train_indices,
            return_visible_points=(self.config.sparse_depth_reg > 0.0),
            load_attributes=dense_depth_load,
            mask_rasterization_mode=self.config.mask_rasterization_mode,
            raster_tile_size=self.config.tile_size,
            raster_context_pixels=MASK_RASTERIZATION_CONTEXT_PIXELS,
            filter_empty_masks=filter_training_masks,
            minimum_valid_mask_area=self.config.minimum_valid_mask_area if filter_training_masks else None,
        )
        filter_validation_masks = not self.config.ignore_masks and len(val_indices) > 0
        self._validation_dataset = SfmDataset(
            sfm_scene=sfm_scene,
            dataset_indices=val_indices,
            filter_empty_masks=filter_validation_masks,
            minimum_valid_mask_area=self.config.minimum_valid_mask_area if filter_validation_masks else None,
            allow_empty_after_filtering=True,
        )

        self.device: torch.device = model.device
        self._render_backend.validate_scene_cameras(self._model, self._training_dataset, self._cfg, self.device)

        self._global_step: int = 0

        self._log_interval_steps: int = log_interval_steps

        self._writer = writer

        # Add Gaussians to the fvdb.viz.Scene if provided.
        self._viz_scene = viz_scene
        self._viz_scene_name = "Gaussian Splat Reconstruction"
        if self._viz_scene is not None:
            with torch.no_grad():
                self._viz_scene.add_gaussian_splat_3d(
                    self._viz_scene_name,
                    gaussian_splat_to_view_data(self._model),
                    tile_size=self._cfg.tile_size,
                    min_radius_2d=self._cfg.min_radius_2d,
                    eps_2d=self._cfg.eps_2d,
                    antialias=self._cfg.antialias,
                    sh_degree_to_use=0,
                )
                camera_eye = self._sfm_scene.image_camera_positions[0]
                camera_lookat = np.median(self._sfm_scene.points, axis=0)
                camera_up = (0, 0, 1)
                self._viz_scene.set_camera_lookat(eye=camera_eye, center=camera_lookat, up=camera_up)

        # Losses & Metrics.
        if self.config.lpips_net == "alex":
            self._lpips = LPIPSLoss(backbone="alex").to(model.device)
        elif self.config.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self._lpips = LPIPSLoss(backbone="vgg").to(model.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self.config.lpips_net}")

    @torch.no_grad()
    def state_dict(self) -> dict[str, Any]:
        """
        Get the state dictionary of the current optimization state, including model, optimizer, and configuration parameters.

        The state dictionary can be used to save and resume optimization from checkpoints. Its keys include:

        * ``"magic"``: A magic string to identify the checkpoint type.
        * ``"version"``: The version of the checkpoint format.
        * ``"step"``: The current global optimization step.
        * ``"config"``: The configuration parameters used for optimization.
        * ``"sfm_scene"``: The state dictionary of the SfM scene.
        * ``"model"``: The state dictionary of the Gaussian Splatting model.
        * ``"optimizer"``: The state dictionary of the optimizer.
        * ``"train_indices"``: The indices of the training images in the dataset.
        * ``"val_indices"``: The indices of the validation images in the dataset.
        * ``"num_training_poses"``: Legacy checkpoint key storing the number of pose entries if pose adjustment is used, otherwise None.
        * ``"pose_adjust_model"``: The state dictionary of the camera pose adjustment model if used, otherwise None.
        * ``"pose_adjust_optimizer"``: The state dictionary of the pose adjustment optimizer if used, otherwise None.
        * ``"pose_adjust_scheduler"``: The state dictionary of the pose adjustment scheduler if used, otherwise None.

        Returns:
            state_dict (dict[str, Any]): A dictionary containing the state of the optimization process.

        """
        return {
            "magic": "GaussianSplattingCheckpoint",
            "version": self.version,
            "step": self._global_step,
            "config": vars(self.config),
            "sfm_scene": self._sfm_scene.state_dict(),
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "train_indices": self._training_dataset.indices,
            "val_indices": self._validation_dataset.indices,
            "num_training_poses": self._pose_adjust_model.num_poses if self._pose_adjust_model else None,
            "pose_adjust_model": self._pose_adjust_model.state_dict() if self._pose_adjust_model else None,
            "pose_adjust_optimizer": self._pose_adjust_optimizer.state_dict() if self._pose_adjust_optimizer else None,
            "pose_adjust_scheduler": self._pose_adjust_scheduler.state_dict() if self._pose_adjust_scheduler else None,
        }

    def save_usd(self, path: str | pathlib.Path, *, usdz: bool = False) -> None:
        """
        Save the current Gaussian Splatting model to a USD file.

        Args:
            path (str | Path): The file path where the USD file will be saved.
            usdz (bool): If True, package the export as a ``.usdz`` archive instead of a single ``.usdc`` file.
        """
        export_splats_to_usd(self._model, str(path), usdz=usdz)

    def save_ply(self, path: str | pathlib.Path) -> None:
        """
        Save the current Gaussian Splatting model to a PLY file.

        Args:
            path (str | Path): The file path where the PLY file will be saved.
        """
        self._model.save_ply(str(path), self.reconstruction_metadata)

    @property
    def reconstruction_metadata(self) -> dict[str, torch.Tensor | float | int | str]:
        """
        Get metadata about the reconstruction, including camera parameters and Gaussian rendering parameters.

        This metadata is useful for downstream tasks such as extracting meshes or point clouds. It includes:

        * ``normalization_transform``: The transformation matrix used to normalize the scene.
        * ``camera_to_world_matrices``: The optimized camera-to-world matrices for the images used during reconstruction.
        * ``projection_matrices``: The projection matrices for the images used during reconstruction.
        * ``image_sizes``: The sizes of the images used during reconstruction.
        * ``median_depths``: The median depth values (distance from camera to scene) for each image used during reconstruction.
        * ``eps2d``: The 2D epsilon value used when rendering the Gaussian splat radiance field.
        * ``near_plane``: The near plane distance used when rendering the Gaussian splat radiance field.
        * ``far_plane``: The far plane distance used when rendering the Gaussian splat radiance field.
        * ``min_radius_2d``: The minimum 2D radius below which splats are not rendered.
        * ``antialias``: Whether anti-aliasing is enabled (1) or not (0).
        * ``tile_size``: The tile size used to render the Gaussian splat radiance field.

        Returns:
            metadata (dict[str, torch.Tensor | float | int | str]): A dictionary containing metadata about the reconstruction.
        """
        training_camera_to_world_matrices = torch.from_numpy(self._training_dataset.camera_to_world_matrices).to(
            dtype=torch.float32, device=self.device
        )
        training_median_depths = torch.from_numpy(self._training_dataset.sfm_scene.median_depth_per_image).to(
            dtype=torch.float32, device=self.device
        )[self._training_dataset.indices]
        if self.pose_adjust_model is not None:
            training_camera_to_world_matrices = self.pose_adjust_model(
                training_camera_to_world_matrices,
                torch.as_tensor(self._training_dataset.indices, dtype=torch.long, device=self.device),
            )

        # Save projection parameters as a per-camera tuple (fx, fy, cx, cy, h, w)
        training_projection_matrices = torch.from_numpy(self._training_dataset.projection_matrices.astype(np.float32))
        training_image_sizes = torch.from_numpy(self._training_dataset.image_sizes.astype(np.int32))
        training_camera_models = torch.from_numpy(self._training_dataset.camera_models.astype(np.int32))
        training_distortion_coeffs = torch.from_numpy(self._training_dataset.distortion_coeffs.astype(np.float32))
        normalization_transform = torch.from_numpy(self.training_dataset.sfm_scene.transformation_matrix).to(
            torch.float32
        )

        return {
            "normalization_transform": normalization_transform,
            "camera_to_world_matrices": training_camera_to_world_matrices,
            "projection_matrices": training_projection_matrices,
            "image_sizes": training_image_sizes,
            "camera_models": training_camera_models,
            "distortion_coeffs": training_distortion_coeffs,
            "median_depths": training_median_depths,
            "eps2d": self.config.eps_2d,
            "near_plane": self.config.near_plane,
            "far_plane": self.config.far_plane,
            "min_radius_2d": self.config.min_radius_2d,
            "antialias": int(self.config.antialias),
            "tile_size": self.config.tile_size,
            "projection_method": self.config.projection_method,
            "render_backend": self.config.render_backend,
        }

    @property
    def config(self) -> GaussianSplatReconstructionConfig:
        """
        Get the configuration object for the current reconstruction. See :class:`GaussianSplatReconstructionConfig` for details.

        Returns:
            config (GaussianSplatReconstructionConfig): The configuration object containing all parameters for the reconstruction.
        """
        return self._cfg

    @property
    def model(self) -> GaussianSplat3d:
        """
        Get the Gaussian Splatting model being optimized.

        Returns:
            model (GaussianSplat3d): The :class:`fvdb_reality_capture.GaussianSplat3d` instance being optimized.
        """
        return self._model

    @property
    def optimizer(self) -> BaseGaussianSplatOptimizer:
        """
        Get the optimizer used for optimizing the Gaussian Splat radiance field's parameters.

        Returns:
            optimizer (BaseGaussianSplatOptimizer): The optimizer instance. See :class:`GaussianSplatOptimizer` for details.
        """
        return self._optimizer

    @property
    def pose_adjust_model(self) -> CameraPoseAdjustment | None:
        """
        Get the camera pose adjustment model used for optimizing camera poses during reconstruction.

        Returns:
            pose_adjust_model (CameraPoseAdjustment | None): The pose adjustment model instance, or None if not used.
        """
        return self._pose_adjust_model

    @property
    def pose_adjust_optimizer(self) -> torch.optim.Adam | None:
        """
        Get the optimizer used for adjusting camera poses during reconstruction.

        Returns:
            pose_adjust_optimizer (torch.optim.Optimizer | None): The pose adjustment optimizer instance, or ``None`` if not used.
        """
        return self._pose_adjust_optimizer

    @property
    def pose_adjust_scheduler(self) -> torch.optim.lr_scheduler.ExponentialLR | None:
        """
        Get the learning rate scheduler used for adjusting camera poses during reconstruction.

        Returns:
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None): The pose adjustment scheduler instance, or ``None`` if not used.
        """
        return self._pose_adjust_scheduler

    @property
    def training_dataset(self) -> SfmDataset:
        """
        Get the training dataset used for training the Gaussian Splatting model.

        Returns:
            training_dataset (SfmDataset): The training dataset instance.
        """
        return self._training_dataset

    @property
    def validation_dataset(self) -> SfmDataset:
        """
        Get the validation dataset used for evaluating the Gaussian Splatting model.

        Returns:
            validation_dataset (SfmDataset): The validation dataset instance.
        """
        return self._validation_dataset

    @staticmethod
    def _init_model(
        config: GaussianSplatReconstructionConfig,
        optimizer_config: GaussianSplatOptimizerConfig,
        device: torch.device | str,
        training_dataset: SfmDataset,
    ):
        """
        Initialize the Gaussian Splatting model with random parameters based on the input dataset.

        Args:
            config (GaussianSplatReconstructionConfig): Configuration object containing model parameters.
            device (torch.device | str): The device to run the model on (e.g., "cuda" or "cpu").
            training_dataset (SfmDataset): The dataset used for optimization, which provides the initial points and RGB values
                            for the Gaussians.
        """

        def _knn_rms_distance(x_np: np.ndarray, k: int = 4) -> torch.Tensor:
            num_points = len(x_np)
            if num_points == 0:
                raise ValueError("Cannot initialize a Gaussian model from an empty point cloud")
            if num_points == 1:
                scene_bbox = np.asarray(training_dataset.scene_bbox, dtype=np.float64)
                scene_diagonal = float(np.linalg.norm(scene_bbox[3:] - scene_bbox[:3]))
                fallback = scene_diagonal * 1.0e-3 if np.isfinite(scene_diagonal) else 1.0e-3
                return torch.full((1,), max(fallback, np.finfo(np.float32).eps), device=device)

            query_k = min(k, num_points)
            kd_tree = cKDTree(x_np)  # type: ignore
            rms_distances = np.empty((num_points,), dtype=np.float32)
            for query_start in range(0, num_points, _KNN_QUERY_BATCH_SIZE):
                query_stop = min(query_start + _KNN_QUERY_BATCH_SIZE, num_points)
                distances, _ = kd_tree.query(x_np[query_start:query_stop], k=query_k, workers=-1)
                neighbor_distances = distances[:, 1:]
                rms_distances[query_start:query_stop] = np.sqrt(np.mean(np.square(neighbor_distances), axis=1)).astype(
                    np.float32
                )
            np.maximum(rms_distances, np.finfo(np.float32).eps, out=rms_distances)
            return torch.from_numpy(rms_distances).to(device=device)

        def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        num_gaussians = training_dataset.points.shape[0]
        if optimizer_config.max_gaussians > 0 and num_gaussians > optimizer_config.max_gaussians:
            raise ValueError(
                f"Initialization contains {num_gaussians:,} points, exceeding max_gaussians="
                f"{optimizer_config.max_gaussians:,}. max_gaussians applies independently to each chunk; "
                "increase nchunks or max_gaussians rather than silently discarding dense initialization points."
            )

        dist_avg = _knn_rms_distance(training_dataset.points, 4)

        log_scales = (
            torch.log(dist_avg * optimizer_config.initial_covariance_scale).unsqueeze(-1).repeat(1, 3)
        )  # [N, 3]

        means = torch.from_numpy(training_dataset.points).to(device=device, dtype=torch.float32)  # [N, 3]
        quats = torch.rand((num_gaussians, 4), device=device)  # [N, 4]
        logit_opacities = torch.logit(
            torch.full((num_gaussians,), optimizer_config.initial_opacity, device=device)
        )  # [N,]

        rgbs = torch.from_numpy(training_dataset.points_rgb).to(device=device, dtype=torch.float32)  # [N, 3]
        rgbs.div_(255.0)
        sh_0 = _rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]

        sh_n = torch.zeros((num_gaussians, (config.sh_degree + 1) ** 2 - 1, 3), device=device)  # [N, K-1, 3]

        model = GaussianSplat3d.from_tensors(means, quats, log_scales, logit_opacities, sh_0, sh_n, True)
        model.requires_grad = True

        return model

    @staticmethod
    def _make_index_splits(sfm_scene: SfmScene, use_every_n_as_val: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Create training and validation splits from the images in the :class:`~fvdb_reality_capture.sfm_scene.sfm_scene.SfmScene`.

        Args:
            sfm_scene (SfmScene): The scene loaded from an structure-from-motion (SfM) pipeline.
            use_every_n_as_val (int): How often to use a training image as a validation image

        Returns:
            train_indices (np.ndarray): Indices of images to use for training.
            val_indices (np.ndarray): Indices of images to use for validation.
        """
        indices = np.arange(sfm_scene.num_images)
        if use_every_n_as_val > 0:
            mask = np.ones(len(indices), dtype=bool)
            mask[::use_every_n_as_val] = False
            train_indices = indices[mask]
            val_indices = indices[~mask]
        else:
            train_indices = indices
            val_indices = np.array([], dtype=np.int64)
        return train_indices, val_indices

    @classmethod
    def _make_pose_optimizer(
        cls,
        optimization_config: GaussianSplatReconstructionConfig,
        device: torch.device | str,
        num_images: int,
        num_steps_per_epoch: int,
    ) -> tuple[CameraPoseAdjustment, torch.optim.Adam, torch.optim.lr_scheduler.ExponentialLR]:
        """
        Create a camera pose adjustment model, optimizer, and scheduler if camera pose optimization is enabled in the config.

        Args:
            optimization_config (Config): Configuration object containing optimization parameters.
            device (torch.device | str): The device to run the model on (e.g., ``"cuda"`` or ``"cpu"``).
            num_images (int): The number of scene images to allocate pose entries for.
            num_steps_per_epoch (int): The number of optimization steps in each training epoch.

        Returns:
            pose_adjust_model (CameraPoseAdjustment | None):
                The camera pose adjustment model, or ``None`` if not used.
            pose_adjust_optimizer (torch.optim.Adam | None):
                The optimizer for the pose adjustment model, or ``None`` if not used.
            pose_adjust_scheduler (torch.optim.lr_scheduler.ExponentialLR | None):
                The learning rate scheduler for the pose adjustment optimizer, or ``None`` if not used.
        """
        if not optimization_config.optimize_camera_poses:
            raise ValueError("Camera pose optimization is not enabled in the config.")

        # Module to adjust camera poses during training
        pose_adjust_model = CameraPoseAdjustment(num_images, init_std=optimization_config.pose_opt_init_std).to(device)

        pose_adjust_optimizer = torch.optim.Adam(
            pose_adjust_model.parameters(),
            lr=optimization_config.pose_opt_lr,
            weight_decay=optimization_config.pose_opt_reg,
        )

        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(pose_adjust_model.parameters(), max_norm=1.0)

        # Add learning rate scheduler for pose optimization
        pose_opt_start_step = int(optimization_config.pose_opt_start_epoch * num_steps_per_epoch)
        pose_opt_stop_step = int(optimization_config.pose_opt_stop_epoch * num_steps_per_epoch)
        num_pose_opt_steps = max(1, pose_opt_stop_step - pose_opt_start_step)
        pose_adjust_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            pose_adjust_optimizer, gamma=optimization_config.pose_opt_lr_decay ** (1.0 / num_pose_opt_steps)
        )
        return pose_adjust_model, pose_adjust_optimizer, pose_adjust_scheduler

    def _clip_gaussians_to_scene_bbox(self) -> None:
        """
        Remove Gaussians whose configured support lies outside the dataset scene bounding box.
        """
        bbox = self.training_dataset.scene_bbox
        bbox_min, bbox_max = bbox[:3], bbox[3:]
        if (
            np.any(np.isinf(bbox_min))
            or np.any(np.isinf(bbox_max))
            or np.any(np.isnan(bbox_min))
            or np.any(np.isnan(bbox_max))
        ):
            self._logger.warning("Scene bounding box is infinite or NaN. Skipping Gaussian clipping.")
            return

        num_gaussians_before_clipping = self.model.num_gaussians
        with torch.no_grad():
            points = self.model.means
            support_extent = torch.zeros_like(points)
            support_sigma = self.config.scene_bbox_support_sigma
            if support_sigma > 0.0:
                unit_quats = nnf.normalize(self.model.quats, dim=-1)
                rotation_matrices = GaussianSplatOptimizer._unit_quats_to_rotation_matrices(unit_quats)
                scaled_axes = rotation_matrices * self.model.scales.unsqueeze(-2)
                squared_extent = torch.sum(torch.square(scaled_axes), dim=-1, keepdim=True)
                support_extent = support_sigma * torch.sqrt(squared_extent).squeeze(-1)

            bbox_min_tensor = torch.as_tensor(bbox_min, device=points.device, dtype=points.dtype)
            bbox_max_tensor = torch.as_tensor(bbox_max, device=points.device, dtype=points.dtype)
            support_min = points - support_extent
            support_max = points + support_extent
            outside_mask = torch.logical_or(
                support_min < bbox_min_tensor.unsqueeze(0), support_max > bbox_max_tensor.unsqueeze(0)
            ).any(dim=-1)

        num_outside = int(outside_mask.sum().item())
        if num_outside == 0:
            self._logger.debug(f"Clipped 0 Gaussians outside the crop bounding box min={bbox_min}, max={bbox_max}.")
            return

        keep_indices = torch.where(~outside_mask)[0]
        # Use integer indices rather than a boolean advanced-indexing expression.
        # This is supported by Torch-DGX and matches the indexing path used during refinement.
        self.optimizer.filter_gaussians(keep_indices)
        num_gaussians_after_clipping = self.model.num_gaussians
        num_clipped_gaussians = num_gaussians_before_clipping - num_gaussians_after_clipping
        self._logger.debug(
            f"Clipped {num_clipped_gaussians:,} Gaussians outside the crop bounding box min={bbox_min}, max={bbox_max}."
        )

    def optimize(self, show_progress: bool = True, log_tag: str = "reconstruct") -> None:
        """
        Run the reconstruction optimization loop to optimize reconstruct a Gaussian Splatting radiance field from a set of posed images.

        The optimization loop iterates over the images and poses in the dataset, computes losses, updates the Gaussian's parameters,
        and logs metrics at each step. It also handles scheduling refinement steps at specified intervals.

        Args:
            show_progress (bool): Whether to display a progress bar during reconstruction.
            log_tag (str): Tag to use for logging metrics (e.g., ``"train"``). Data logged will use this tag as a prefix.
                For metrics, this will be ``"{log_tag}/metric_name"``.
                For checkpoints, this will be ``"{log_tag}_ckpt.pt"``.
                For PLY files, this will be ``"{log_tag}_ckpt.ply"``.

        .. note:: When calling evaluation from the reconstruction loop, the log_tag for evaluation will be ``log_tag+"_eval"``.
        """
        if self.optimizer is None:
            raise ValueError("This runner was not created with an optimizer. Cannot run reconstruction.")

        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )

        if self.config.batch_size > 1:
            num_steps_per_epoch = np.ceil(len(self.training_dataset) / self.config.batch_size).astype(int)
        else:
            num_steps_per_epoch = len(self.training_dataset)

        # Calculate total steps, allowing max_steps to override the computed value
        computed_total_steps: int = int(self.config.max_epochs * num_steps_per_epoch)
        total_steps: int = self.config.max_steps if self.config.max_steps is not None else computed_total_steps

        refine_start_step: int = int(self.config.refine_start_epoch * num_steps_per_epoch)
        refine_stop_step: int = int(self.config.refine_stop_epoch * num_steps_per_epoch)
        refine_every_step: int = int(self.config.refine_every_epoch * num_steps_per_epoch)
        increase_sh_degree_every_step: int = int(self.config.increase_sh_degree_every_epoch * num_steps_per_epoch)
        pose_opt_start_step: int = int(self.config.pose_opt_start_epoch * num_steps_per_epoch)
        pose_opt_stop_step: int = int(self.config.pose_opt_stop_epoch * num_steps_per_epoch)

        last_post_refinement_prune_step: int | None = None

        def prune_after_refinement_stop() -> None:
            """Prune at most once at a given global step."""
            nonlocal last_post_refinement_prune_step
            if last_post_refinement_prune_step == self._global_step:
                return
            assert isinstance(self.optimizer, GaussianSplatOptimizer)
            self.optimizer.prune(use_scale_threshold=True)
            last_post_refinement_prune_step = self._global_step

        def prepare_post_refinement_step() -> None:
            """Disable work that is needed only while insertion and splitting are active."""
            if self._global_step < refine_stop_step:
                return
            self.model.accumulate_mean_2d_gradients = False
            self.model.accumulate_max_2d_radii = False
            if self.config.freeze_scales_after_refinement_stop:
                # Disabling autograd before rendering avoids computing scale gradients. Filtering can replace
                # parameter tensors and re-enable gradients, so this helper is also called before optimizer.step().
                self.model.log_scales.grad = None
                self.model.log_scales.requires_grad_(False)

        update_viz_every_step = int(self._viz_update_interval_epochs * num_steps_per_epoch)

        # Progress bar to track reconstruction progress
        if self.config.max_steps is not None:
            self._logger.info(
                f"Using max_steps={self.config.max_steps} (overriding computed {computed_total_steps} steps)"
            )
        if show_progress:
            pbar = tqdm.tqdm(range(0, total_steps), unit="steps", desc="Gaussian Splat Reconstruction")
        else:
            pbar = None

        # Flag to break out of outer epoch loop when max_steps is reached
        reached_max_steps = False

        # Zero out gradients before reconstruction in case we resume resume from a checkpoint
        self.optimizer.zero_grad()
        if self.pose_adjust_optimizer is not None:
            self.pose_adjust_optimizer.zero_grad()

        # Iterate over the sampler as opposed to the trainloader. This preserves the same sampling characteristics for
        # a shuffled dataloader while also avoiding the (often significant) overhead of loading each batch from disk.
        start_epoch = self._start_step // len(trainloader)
        for epoch in range(start_epoch):
            for _ in trainloader.sampler:
                # Skip steps before the start step
                if self._global_step < self._start_step:
                    if pbar:
                        pbar.set_description(
                            f"Skipping step {self._global_step:,} (before start step {self._start_step:,})"
                        )
                        pbar.update(1)
                        self._global_step = pbar.n
                    else:
                        self._global_step += 1
                    continue

        # Re-initialize the progress bar so that the speed of the iterations over the sampler (which are very fast) are
        # not averaged with the speed of actual training iterations.
        if pbar:
            pbar = tqdm.tqdm(
                range(0, total_steps), initial=self._start_step, unit="steps", desc="Gaussian Splat Reconstruction"
            )

        for epoch in range(start_epoch, self.config.max_epochs):
            for minibatch in trainloader:
                prepare_post_refinement_step()

                # Camera pose optimization
                image_ids: torch.Tensor | None = None
                if self.pose_adjust_model is not None:
                    cam_to_world_mats = minibatch["camera_to_world"].to(self.device)  # [B, 4, 4]
                    image_ids = minibatch["image_id"].to(self.device)  # [B]
                    if self._global_step == pose_opt_start_step:
                        self._logger.info(
                            f"Starting to optimize camera poses at step {self._global_step:,} (epoch {epoch})"
                        )
                    if pose_opt_start_step <= self._global_step < pose_opt_stop_step:
                        cam_to_world_mats = self.pose_adjust_model(cam_to_world_mats, image_ids)
                    elif self._global_step >= pose_opt_stop_step:
                        # After pose_opt_stop_iter, don't track gradients through pose adjustment
                        with torch.no_grad():
                            cam_to_world_mats = self.pose_adjust_model(cam_to_world_mats, image_ids)
                    world_to_cam_mats = torch.linalg.inv_ex(cam_to_world_mats)[0].contiguous()
                else:
                    world_to_cam_mats = minibatch["world_to_camera"].to(self.device)  # [B, 4, 4]

                projection_mats = minibatch["projection"].to(self.device)  # [B, 3, 3]
                camera_models = minibatch["camera_model"]
                distortion_coeffs = minibatch["distortion_coeffs"]
                image = minibatch["image"]  # [B, H, W, 3]
                mask = minibatch["mask"] if "mask" in minibatch and not self.config.ignore_masks else None
                mask_rasterization_mode = self.config.mask_rasterization_mode
                raster_tile_mask = minibatch.get("raster_tile_mask")
                if raster_tile_mask is not None:
                    raster_tile_mask = raster_tile_mask.to(self.device, dtype=torch.bool)

                if mask_rasterization_mode == "bbox":
                    full_image_size = minibatch["full_image_size"][0]
                    image_height = int(full_image_size[0].item())
                    image_width = int(full_image_size[1].item())
                    raster_crop = minibatch["raster_crop"][0]
                    crop = tuple(int(value.item()) for value in raster_crop)
                    training_crops = ((image, mask, crop, True),)
                else:
                    image_height, image_width = image.shape[1:3]
                    training_crops = crop_image_batch(image, mask, self.config.crops_per_image)

                image_loss_scale = 1.0
                if "image_loss_scale" in minibatch:
                    image_loss_scale = float(minibatch["image_loss_scale"].reshape(-1)[0].item())
                sparse_depth = minibatch["sparse_depth"].to(self.device) if "sparse_depth" in minibatch else None
                sparse_depth_uv = (
                    minibatch["sparse_depth_uv"].to(self.device, dtype=torch.int32)
                    if "sparse_depth_uv" in minibatch
                    else None
                )
                median_depths = (
                    minibatch["median_depth"].to(self.device) if "median_depth" in minibatch else None
                )  # [B,]

                dense_depth_key = self.config.dense_depth_attribute
                if dense_depth_key is not None and dense_depth_key in minibatch:
                    dense_depth_tgt = minibatch[dense_depth_key].to(self.device)  # [B, H, W]
                    dense_depth_valid = minibatch[f"{dense_depth_key}_valid"].to(self.device)  # [B, H, W] bool
                else:
                    dense_depth_tgt = None
                    dense_depth_valid = None

                # Progressively use higher spherical harmonic degree as we optimize
                sh_degree_to_use = min(self._global_step // increase_sh_degree_every_step, self.config.sh_degree)
                # If you have very large images, you can iterate over disjoint crops and accumulate gradients
                # If self.optimization_config.crops_per_image is 1, then this just returns the image
                for pixels, mask_pixels, crop, is_last in training_crops:
                    # Actual pixels to compute the loss on, normalized to [0, 1]
                    pixels: torch.Tensor = pixels.to(device=self.device) / 255.0  # [1, H, W, 3]

                    # Render an image from the gaussian splats
                    # possibly using a crop of the full image
                    render_outputs = self._render_backend.forward_train(
                        model=self.model,
                        config=self.config,
                        world_to_camera_matrices=world_to_cam_mats,
                        projection_matrices=projection_mats,
                        camera_models=camera_models,
                        distortion_coeffs=distortion_coeffs,
                        image_width=image_width,
                        image_height=image_height,
                        sh_degree_to_use=sh_degree_to_use,
                        crop=crop,
                        raster_tile_mask=raster_tile_mask,
                    )
                    image = render_outputs.image

                    # If you want to add random background, we'll mix it in here
                    if self.config.random_bkgd:
                        bkgd = torch.rand(1, 3, device=self.device)
                        image = _composite_background(image, render_outputs.alpha, bkgd)

                    # Image losses
                    if mask_pixels is not None:
                        l1loss, valid_l1loss = _masked_l1_losses(image, pixels, mask_pixels)
                        ssimloss, valid_ssimloss = _masked_ssim_losses(image, pixels, mask_pixels)
                    else:
                        l1loss = nnf.l1_loss(image, pixels)
                        ssimloss = 1.0 - ssim(
                            image.permute(0, 3, 1, 2).contiguous(),
                            pixels.permute(0, 3, 1, 2).contiguous(),
                        )
                        valid_l1loss = l1loss
                        valid_ssimloss = ssimloss

                    l1loss = l1loss * image_loss_scale
                    ssimloss = ssimloss * image_loss_scale
                    valid_image_loss = torch.lerp(valid_l1loss, valid_ssimloss, self.config.ssim_lambda)
                    loss = torch.lerp(l1loss, ssimloss, self.config.ssim_lambda)  # type: ignore

                    # Apply any additional regularization to the model for the given
                    # optimizer.
                    loss = loss + self.optimizer.regularization_loss()
                    # Sparse depth loss
                    if sparse_depth is not None and sparse_depth_uv is not None and median_depths is not None:
                        if self.config.batch_size > 1:
                            raise NotImplementedError("Sparse depth loss is not implemented for batch_size > 1.")
                        if render_outputs.depth is None:
                            raise RuntimeError("Model did not render depth channel, but sparse depth loss is enabled.")
                        if sparse_depth_uv.numel() == 0:
                            depth_loss = 0.0
                        else:
                            depth = render_outputs.depth[..., 0]  # [1, H, W]
                            depth_uv = depth[:, sparse_depth_uv[0, :, 1], sparse_depth_uv[0, :, 0]]  # [B, N]
                            alpha_uv = render_outputs.alpha[
                                :, sparse_depth_uv[0, :, 1], sparse_depth_uv[0, :, 0], 0
                            ]  # [B, N]
                            pred_depth = depth_uv / torch.clamp(alpha_uv, min=1e-6)  # [B, N]
                            pred_depth = pred_depth / median_depths.unsqueeze(1)  # Normalize by median depth
                            sparse_depth = sparse_depth / median_depths.unsqueeze(1)  # Normalize by median depth

                            depth_loss = nnf.l1_loss(pred_depth, sparse_depth) * self.config.sparse_depth_reg
                            loss = loss + depth_loss
                    else:
                        depth_loss = 0.0

                    # Dense depth loss (from a DepthMapAttribute on the scene).
                    if dense_depth_tgt is not None and self.config.dense_depth_reg > 0.0:
                        if self.config.batch_size > 1:
                            raise NotImplementedError("Dense depth loss is not implemented for batch_size > 1.")
                        if render_outputs.depth is None:
                            raise RuntimeError("Model did not render depth channel, but dense depth loss is enabled.")
                        cx, cy, cw, ch = crop
                        gt_depth_crop = dense_depth_tgt[:, cy : cy + ch, cx : cx + cw]
                        gt_valid_crop = dense_depth_valid[:, cy : cy + ch, cx : cx + cw]
                        pred_dense_depth = render_outputs.depth[..., 0] / torch.clamp(
                            render_outputs.alpha[..., 0], min=1e-6
                        )  # [B, h, w]
                        if self._dense_depth_is_relative:
                            dense_depth_loss = (
                                _scale_shift_invariant_l1(pred_dense_depth, gt_depth_crop, gt_valid_crop)
                                * self.config.dense_depth_reg
                            )
                        else:
                            valid_mask = gt_valid_crop.float()
                            valid_count = valid_mask.sum().clamp(min=1.0)
                            dense_depth_loss = (
                                (torch.abs(pred_dense_depth - gt_depth_crop) * valid_mask).sum()
                                / valid_count
                                * self.config.dense_depth_reg
                            )
                        loss = loss + dense_depth_loss
                    else:
                        dense_depth_loss = 0.0

                    # If you're optimizing poses, regularize the pose parameters so the poses
                    # don't drift too far from the initial values
                    if (
                        self.pose_adjust_model is not None
                        and pose_opt_start_step <= self._global_step < pose_opt_stop_step
                    ):
                        assert image_ids is not None
                        pose_params = self.pose_adjust_model.pose_embeddings(image_ids)
                        pose_reg = torch.mean(torch.abs(pose_params))
                        loss = loss + self.config.pose_opt_reg * pose_reg
                    else:
                        pose_reg = None

                    # If we're splitting into crops, accumulate gradients, so pass retain_graph=True
                    # for every crop but the last one
                    loss.backward(retain_graph=not is_last)

                # Refine before the stop step and enforce bbox clipping on the same cadence throughout training.
                is_refinement_cadence_step = self._global_step % refine_every_step == 0
                is_after_refinement_start = self._global_step >= refine_start_step
                if is_refinement_cadence_step and (is_after_refinement_start or self._global_step >= refine_stop_step):
                    if is_after_refinement_start and self._global_step < refine_stop_step:
                        self.optimizer.refine()
                    elif self.config.prune_after_refinement_stop:
                        prune_after_refinement_stop()

                    # Keep enforcing the crop at the same cadence after densification stops so drifting
                    # Gaussians do not consume resources for the remainder of optimization.
                    if self.config.remove_gaussians_outside_scene_bbox:
                        self._clip_gaussians_to_scene_bbox()

                # Filtering replaces parameter tensors and re-enables gradients, so restore the post-stop state.
                prepare_post_refinement_step()

                # Step the Gaussian optimizer
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # If you enabled pose optimization, step the pose optimizer if we performed a
                # pose update this iteration
                if self.config.optimize_camera_poses and pose_opt_start_step <= self._global_step < pose_opt_stop_step:
                    assert self.pose_adjust_optimizer is not None, (
                        "Pose optimizer should be initialized if pose optimization is enabled."
                    )
                    assert self.pose_adjust_scheduler is not None, (
                        "Pose scheduler should be initialized if pose optimization is enabled."
                    )
                    self.pose_adjust_optimizer.step()
                    self.pose_adjust_scheduler.step()
                    self.pose_adjust_optimizer.zero_grad(set_to_none=True)

                # Update the log in the progress bar
                if pbar is not None:
                    pbar.set_description(
                        f"loss={loss.item():.3f}| "
                        f"sh degree={sh_degree_to_use}| "
                        f"num gaussians={self.model.num_gaussians:,}"
                    )

                # Log metrics
                if self._global_step % self._log_interval_steps == 0:
                    if self.device.type == "dgx":
                        cuda_devices = range(torch.cuda.device_count())
                        mem_allocated = sum(torch.cuda.memory_allocated(device) for device in cuda_devices) / (1024**3)
                        mem_reserved = sum(torch.cuda.memory_reserved(device) for device in cuda_devices) / (1024**3)
                    else:
                        mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                        mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                    self._writer.log_metric(self._global_step, f"{log_tag}/loss", loss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/l1loss", l1loss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/ssimloss", ssimloss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/l1loss_valid", valid_l1loss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/ssimloss_valid", valid_ssimloss.item())
                    self._writer.log_metric(self._global_step, f"{log_tag}/image_loss_valid", valid_image_loss.item())
                    self._writer.log_metric(
                        self._global_step,
                        f"{log_tag}/depth_loss",
                        depth_loss if isinstance(depth_loss, float) else depth_loss.item(),
                    )
                    self._writer.log_metric(
                        self._global_step,
                        f"{log_tag}/dense_depth_loss",
                        dense_depth_loss if isinstance(dense_depth_loss, float) else dense_depth_loss.item(),
                    )
                    self._writer.log_metric(self._global_step, f"{log_tag}/num_gaussians", self.model.num_gaussians)
                    self._writer.log_metric(self._global_step, f"{log_tag}/sh_degree", sh_degree_to_use)
                    self._writer.log_metric(self._global_step, f"{log_tag}/mem_allocated", mem_allocated)
                    self._writer.log_metric(self._global_step, f"{log_tag}/mem_reserved", mem_reserved)
                    if pose_reg is not None:
                        self._writer.log_metric(self._global_step, f"{log_tag}/pose_reg_loss", pose_reg.item())

                # Update the visualization if provided
                if self._viz_scene is not None and self._global_step % update_viz_every_step == 0:
                    with torch.no_grad():
                        self._logger.info(f"Updating visualization at step {self._global_step:,}")
                        self._viz_scene.add_gaussian_splat_3d(
                            self._viz_scene_name,
                            gaussian_splat_to_view_data(self.model),
                            tile_size=self._cfg.tile_size,
                            min_radius_2d=self._cfg.min_radius_2d,
                            eps_2d=self._cfg.eps_2d,
                            antialias=self._cfg.antialias,
                            sh_degree_to_use=sh_degree_to_use,
                        )

                # Update the progress bar and global step
                if pbar is not None:
                    pbar.update(1)
                    self._global_step = pbar.n
                else:
                    self._global_step += 1

                # Check if we've reached max_steps and break out of the optimization loop
                if self.config.max_steps is not None and self._global_step >= self.config.max_steps:
                    reached_max_steps = True
                    break

            # Check if we've reached max_steps and break out of outer epoch loop
            if reached_max_steps:
                break

            # Save the model if we've reached a percentage of the total epochs specified in save_at_percent
            if epoch in [(pct * self.config.max_epochs // 100) - 1 for pct in self.config.save_at_percent]:
                if self._global_step <= self._start_step:
                    self._logger.info(
                        f"Skipping checkpoint save at epoch {epoch + 1} (before start step {self._start_step})."
                    )
                    continue
                if self.config.prune_after_refinement_stop and self._global_step >= refine_stop_step:
                    prune_after_refinement_stop()
                if self.config.remove_gaussians_outside_scene_bbox and self._global_step >= refine_stop_step:
                    self._clip_gaussians_to_scene_bbox()
                prepare_post_refinement_step()
                self._logger.info(f"Saving checkpoint at global step {self._global_step}.")
                self._writer.save_checkpoint(self._global_step, f"{log_tag}_ckpt.pt", self.state_dict())
                self._writer.save_ply(
                    self._global_step, f"{log_tag}_ckpt.ply", self.model, self.reconstruction_metadata
                )

            # Run evaluation if we've reached a percentage of the total epochs specified in eval_at_percent
            if epoch in [(pct * self.config.max_epochs // 100) - 1 for pct in self.config.eval_at_percent]:
                if len(self.validation_dataset) == 0:
                    continue
                if self._global_step <= self._start_step:
                    self._logger.info(
                        f"Skipping evaluation at epoch {epoch + 1} (before start step {self._start_step})."
                    )
                    continue
                self.eval(log_tag=log_tag + "_eval")

        # Means continue to optimize after refinement stops and can drift outside the scene
        # bounding box. Enforce the crop once more before the caller performs its final export.
        if self.config.prune_after_refinement_stop and self._global_step >= refine_stop_step:
            prune_after_refinement_stop()
        if self.config.remove_gaussians_outside_scene_bbox:
            self._clip_gaussians_to_scene_bbox()
        prepare_post_refinement_step()

        self._logger.info("Training completed.")

    @torch.no_grad()
    def eval(self, show_progress: bool = True, log_tag: str = "eval") -> None:
        """
        Evaluate the quality of the Gaussian Splat radiance field on the validation dataset.

        This method evaluates the model by rendering images from the Gaussian Splat radiance field and computing
        various image quality metrics including PSNR, SSIM, and LPIPS. It also saves the rendered images and ground truth images
        to the log writer for visualization.

        Args:
            show_progress (bool): Whether to display a progress bar during evaluation.
            log_tag (str): Tag to use for logging metrics and images. Data logged will use this tag as a prefix.
                For metrics, this will be ``"{log_tag}/metric_name"``.
                For images, this will be ``"{log_tag}/predicted_imageXXXX.jpg"`` and ``"{log_tag}/ground_truth_imageXXXX.jpg"``.
        """
        self._logger.info("Running evaluation...")
        device = self.device

        valloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=1, shuffle=False, num_workers=1)
        if show_progress:
            pbar = tqdm.tqdm(enumerate(valloader), total=len(self.validation_dataset), unit="imgs", desc="Evaluating")
        else:
            pbar = enumerate(valloader)
        evaluation_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "psnr_valid": [], "ssim_valid": [], "l1_valid": []}
        for i, data in pbar:
            world_to_cam_matrices = data["world_to_camera"].to(device)
            projection_matrices = data["projection"].to(device)
            camera_models = data["camera_model"].to(device)
            distortion_coeffs = data["distortion_coeffs"].to(device)
            ground_truth_image = data["image"].to(device) / 255.0
            mask_pixels = data["mask"] if "mask" in data and not self.config.ignore_masks else None

            height, width = ground_truth_image.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            render_outputs = self._render_backend.forward_eval(
                model=self.model,
                config=self.config,
                world_to_camera_matrices=world_to_cam_matrices,
                projection_matrices=projection_matrices,
                camera_models=camera_models,
                distortion_coeffs=distortion_coeffs,
                image_width=width,
                image_height=height,
                sh_degree_to_use=self.config.sh_degree,
            )
            predicted_image = render_outputs.image
            predicted_image = torch.clamp(predicted_image, 0.0, 1.0)
            # depths = image[..., -1:] / alpha.clamp(min=1e-10)
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            # depths = depths / depths.max()

            torch.cuda.synchronize()

            evaluation_time += time.time() - tic

            valid_psnr = None
            valid_ssim = None
            if mask_pixels is not None:
                _, valid_l1 = _masked_l1_losses(predicted_image, ground_truth_image, mask_pixels)
                valid_psnr = _masked_psnr(predicted_image, ground_truth_image, mask_pixels)
                valid_ssim = masked_ssim(
                    predicted_image.permute(0, 3, 1, 2).contiguous(),
                    ground_truth_image.permute(0, 3, 1, 2).contiguous(),
                    mask_pixels.to(device=device, dtype=torch.bool).unsqueeze(1),
                    reduction="valid",
                    train=False,
                )
                # Preserve the existing full-image-normalized metric definitions.
                ground_truth_image = _masked_ground_truth(ground_truth_image, predicted_image, mask_pixels)
            else:
                valid_l1 = nnf.l1_loss(predicted_image, ground_truth_image)

            # Save images
            self._writer.save_image(self._global_step, f"{log_tag}/predicted_image{i:04d}.jpg", predicted_image)
            self._writer.save_image(self._global_step, f"{log_tag}/ground_truth_image{i:04d}.jpg", ground_truth_image)

            ground_truth_image = ground_truth_image.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            predicted_image = predicted_image.permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
            image_psnr = psnr(predicted_image, ground_truth_image)
            image_ssim = ssim(predicted_image, ground_truth_image)
            metrics["psnr"].append(image_psnr)
            metrics["ssim"].append(image_ssim)
            metrics["lpips"].append(self._lpips(predicted_image, ground_truth_image))
            metrics["psnr_valid"].append(image_psnr if valid_psnr is None else valid_psnr)
            metrics["ssim_valid"].append(image_ssim if valid_ssim is None else valid_ssim)
            metrics["l1_valid"].append(valid_l1)

        evaluation_time /= len(valloader)

        psnr_mean = torch.stack(metrics["psnr"]).mean()
        ssim_mean = torch.stack(metrics["ssim"]).mean()
        lpips_mean = torch.stack(metrics["lpips"]).mean()
        psnr_valid_mean = torch.stack(metrics["psnr_valid"]).mean()
        ssim_valid_mean = torch.stack(metrics["ssim_valid"]).mean()
        l1_valid_mean = torch.stack(metrics["l1_valid"]).mean()
        self._logger.info(f"Evaluation for stage {log_tag} completed. Average time per image: {evaluation_time:.3f}s")
        self._logger.info(f"PSNR: {psnr_mean.item():.3f}, SSIM: {ssim_mean.item():.4f}, LPIPS: {lpips_mean.item():.3f}")
        self._logger.info(
            f"Valid PSNR: {psnr_valid_mean.item():.3f}, SSIM: {ssim_valid_mean.item():.4f}, L1: {l1_valid_mean.item():.5f}"
        )

        self._writer.log_metric(self._global_step, f"{log_tag}/psnr", psnr_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/ssim", ssim_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/lpips", lpips_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/psnr_valid", psnr_valid_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/ssim_valid", ssim_valid_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/l1_valid", l1_valid_mean.item())
        self._writer.log_metric(self._global_step, f"{log_tag}/evaluation_time", evaluation_time)
        self._writer.log_metric(self._global_step, f"{log_tag}/num_gaussians", self.model.num_gaussians)
