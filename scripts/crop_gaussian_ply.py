#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Crop an fVDB Gaussian-splat PLY using per-axis coordinate percentiles.

The source PLY is never modified. Gaussian centers determine membership in the
crop, and all attributes belonging to retained Gaussians are preserved.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from fvdb_reality_capture import GaussianSplat3d

LOGGER = logging.getLogger(__name__)
PercentileRange = tuple[float, float]
DEFAULT_MAX_QUANTILE_SAMPLES = 1_000_000


@dataclass(frozen=True)
class CropResult:
    """Spatial crop mask and the coordinate bounds used to construct it."""

    mask: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    finite_count: int
    sampled_count: int
    sample_stride: int

    @property
    def input_count(self) -> int:
        return int(self.mask.numel())

    @property
    def retained_count(self) -> int:
        return int(self.mask.sum().item())

    @property
    def removed_count(self) -> int:
        return self.input_count - self.retained_count

    @property
    def removed_percent(self) -> float:
        if self.input_count == 0:
            return 0.0
        return 100.0 * self.removed_count / self.input_count


def validate_percentile_ranges(
    x_percentiles: Sequence[float],
    y_percentiles: Sequence[float],
    z_percentiles: Sequence[float],
) -> tuple[PercentileRange, PercentileRange, PercentileRange]:
    """Validate and normalize three ``(low, high)`` percentile ranges."""

    normalized: list[PercentileRange] = []
    for axis, values in zip("xyz", (x_percentiles, y_percentiles, z_percentiles), strict=True):
        if len(values) != 2:
            raise ValueError(f"{axis.upper()} percentiles must contain exactly two values")
        low, high = float(values[0]), float(values[1])
        if not 0.0 <= low <= high <= 100.0:
            raise ValueError(
                f"{axis.upper()} percentiles must satisfy 0 <= LOW <= HIGH <= 100; received {low:g} {high:g}"
            )
        normalized.append((low, high))
    return normalized[0], normalized[1], normalized[2]


def compute_percentile_crop(
    means: torch.Tensor,
    x_percentiles: Sequence[float] = (0.0, 100.0),
    y_percentiles: Sequence[float] = (0.0, 100.0),
    z_percentiles: Sequence[float] = (0.0, 100.0),
    sample_stride: int = 1,
    max_quantile_samples: int = DEFAULT_MAX_QUANTILE_SAMPLES,
) -> CropResult:
    """Build an inclusive XYZ crop mask from Gaussian-center percentiles."""

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"Expected Gaussian means with shape (N, 3); received {tuple(means.shape)}")
    if means.shape[0] == 0:
        raise ValueError("Cannot crop an empty Gaussian model")
    if not means.is_floating_point():
        raise ValueError("Gaussian means must use a floating-point dtype")
    if sample_stride < 1:
        raise ValueError(f"sample_stride must be at least 1; received {sample_stride}")
    if max_quantile_samples < 1:
        raise ValueError(f"max_quantile_samples must be at least 1; received {max_quantile_samples}")

    percentile_ranges = validate_percentile_ranges(x_percentiles, y_percentiles, z_percentiles)
    finite_mask = torch.isfinite(means).all(dim=1)
    finite_count = int(finite_mask.sum().item())
    if finite_count == 0:
        raise ValueError("Cannot crop a model with no finite Gaussian centers")
    finite_means = means if finite_count == means.shape[0] else means[finite_mask]

    minimum_stride = math.ceil(finite_count / max_quantile_samples)
    effective_stride = max(sample_stride, minimum_stride)
    sampled_means = finite_means[::effective_stride]
    quantiles = means.new_tensor(percentile_ranges) / 100.0
    axis_bounds: list[torch.Tensor] = []
    for axis in range(3):
        low_percentile, high_percentile = percentile_ranges[axis]
        if low_percentile == 0.0 and high_percentile == 100.0:
            axis_bounds.append(torch.stack((finite_means[:, axis].min(), finite_means[:, axis].max())))
            continue
        try:
            sampled_bounds = torch.quantile(sampled_means[:, axis], quantiles[axis])
        except RuntimeError as error:
            if "input tensor is too large" not in str(error):
                raise
            raise ValueError(
                f"PyTorch could not calculate quantiles from {sampled_means.shape[0]:,} samples; "
                "lower --max-quantile-samples and retry"
            ) from error
        lower_bound = finite_means[:, axis].min() if low_percentile == 0.0 else sampled_bounds[0]
        upper_bound = finite_means[:, axis].max() if high_percentile == 100.0 else sampled_bounds[1]
        axis_bounds.append(torch.stack((lower_bound, upper_bound)))
    bounds = torch.stack(axis_bounds, dim=0)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    mask = finite_mask
    for axis in range(3):
        mask &= means[:, axis] >= lower_bounds[axis]
        mask &= means[:, axis] <= upper_bounds[axis]
    return CropResult(
        mask=mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        finite_count=finite_count,
        sampled_count=int(sampled_means.shape[0]),
        sample_stride=effective_stride,
    )


def _default_output_path(input_path: pathlib.Path) -> pathlib.Path:
    return input_path.with_name(f"{input_path.stem}_cropped.ply")


def _validate_paths(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    force: bool,
    *,
    will_write: bool = True,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() != ".ply":
        raise ValueError(f"Input path must end in .ply: {input_path}")
    if output_path.suffix.lower() != ".ply":
        raise ValueError(f"Output path must end in .ply: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different; the source PLY is never modified")
    if will_write and output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output without --force: {output_path}")
    if will_write and not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")
    if will_write and not output_path.parent.is_dir():
        raise ValueError(f"Output parent is not a directory: {output_path.parent}")


def _save_ply_atomically(
    model: GaussianSplat3d,
    output_path: pathlib.Path,
    metadata: Mapping[str, str | int | float | torch.Tensor],
) -> None:
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}.", dir=output_path.parent) as directory:
        temporary_path = pathlib.Path(directory) / output_path.name
        try:
            model.save_ply(temporary_path, metadata)
            os.replace(temporary_path, output_path)
        except NotImplementedError as error:
            raise ValueError(
                f"fVDB could not save a PLY from device {model.device}; try --device cuda: {error}"
            ) from error


def _format_bounds(values: torch.Tensor) -> str:
    return " ".join(f"{value:.9g}" for value in values.detach().cpu().tolist())


def _log_crop_summary(result: CropResult) -> None:
    LOGGER.info("Coordinate lower bounds (x y z): %s", _format_bounds(result.lower_bounds))
    LOGGER.info("Coordinate upper bounds (x y z): %s", _format_bounds(result.upper_bounds))
    LOGGER.info(
        "Gaussians: input=%d finite=%d sampled=%d sample_stride=%d retained=%d removed=%d (%.2f%%)",
        result.input_count,
        result.finite_count,
        result.sampled_count,
        result.sample_stride,
        result.retained_count,
        result.removed_count,
        result.removed_percent,
    )


def crop_ply(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    x_percentiles: Sequence[float] = (0.0, 100.0),
    y_percentiles: Sequence[float] = (0.0, 100.0),
    z_percentiles: Sequence[float] = (0.0, 100.0),
    sample_stride: int = 1,
    max_quantile_samples: int = DEFAULT_MAX_QUANTILE_SAMPLES,
    device: str = "cuda",
    dry_run: bool = False,
    force: bool = False,
    max_remove_percent: float = 50.0,
    allow_large_crop: bool = False,
) -> CropResult:
    """Load, percentile-crop, and optionally save a Gaussian-splat PLY."""

    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    _validate_paths(input_path, output_path, force=force, will_write=not dry_run)
    if not 0.0 <= max_remove_percent <= 100.0:
        raise ValueError(f"max_remove_percent must be between 0 and 100; received {max_remove_percent:g}")

    LOGGER.info("Loading Gaussian splats from %s on %s", input_path, device)
    model, metadata = GaussianSplat3d.from_ply(input_path, device=device)
    result = compute_percentile_crop(
        model.means,
        x_percentiles=x_percentiles,
        y_percentiles=y_percentiles,
        z_percentiles=z_percentiles,
        sample_stride=sample_stride,
        max_quantile_samples=max_quantile_samples,
    )
    if result.sample_stride > sample_stride:
        LOGGER.info(
            "Automatically increased quantile sample stride from %d to %d to stay within %d samples",
            sample_stride,
            result.sample_stride,
            max_quantile_samples,
        )
    _log_crop_summary(result)

    if dry_run:
        if result.retained_count == 0:
            LOGGER.warning("Crop would remove every Gaussian")
        elif result.removed_percent > max_remove_percent and not allow_large_crop:
            LOGGER.warning(
                "Crop would exceed --max-remove-percent %g; a real write would require --allow-large-crop",
                max_remove_percent,
            )
        LOGGER.info("Dry run complete; no output was written")
        return result

    if result.retained_count == 0:
        raise ValueError("Crop would remove every Gaussian; no output was written")
    if result.removed_percent > max_remove_percent and not allow_large_crop:
        raise ValueError(
            f"Crop would remove {result.removed_percent:.2f}% of Gaussians, exceeding "
            f"--max-remove-percent {max_remove_percent:g}; inspect with --dry-run or pass --allow-large-crop"
        )

    cropped_model = model[result.mask]
    _save_ply_atomically(cropped_model, output_path, metadata)
    LOGGER.info("Wrote cropped PLY to %s", output_path)
    return result


def _show_output(output_path: pathlib.Path, device: str, viewer_port: int) -> None:
    frgs_path = shutil.which("frgs")
    if frgs_path is None:
        raise FileNotFoundError("Cannot launch viewer because `frgs` is not available on PATH")
    command = [
        frgs_path,
        "show",
        str(output_path),
        "--device",
        device,
        "--viewer-port",
        str(viewer_port),
    ]
    LOGGER.info("Launching viewer: %s", " ".join(command))
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop an fVDB Gaussian-splat PLY by retaining low/high coordinate percentiles independently along X, Y, "
            "and Z. Percentiles apply to Gaussian centers."
        )
    )
    parser.add_argument("input", type=pathlib.Path, help="Source Gaussian-splat PLY")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output PLY (default: <input_stem>_cropped.ply alongside the source)",
    )
    parser.add_argument(
        "--x-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, 100.0),
        help="Retained X-coordinate percentiles in 0..100 (default: 0 100)",
    )
    parser.add_argument(
        "--y-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, 100.0),
        help="Retained Y-coordinate percentiles in 0..100 (default: 0 100)",
    )
    parser.add_argument(
        "--z-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.0, 100.0),
        help="Retained Z-coordinate percentiles in 0..100 (default: 0 100)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Minimum stride used to estimate percentile bounds; automatically increased for large scenes (default: 1)",
    )
    parser.add_argument(
        "--max-quantile-samples",
        type=int,
        default=DEFAULT_MAX_QUANTILE_SAMPLES,
        help="Maximum centers used for quantile estimation; all centers are still cropped (default: 1000000)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device used to load and crop the PLY (default: cuda)")
    parser.add_argument("--dry-run", action="store_true", help="Report bounds and removal counts without writing a PLY")
    parser.add_argument("--force", action="store_true", help="Allow replacement of an existing output PLY")
    parser.add_argument(
        "--max-remove-percent",
        type=float,
        default=50.0,
        help="Refuse crops removing more than this percentage unless --allow-large-crop is set (default: 50)",
    )
    parser.add_argument(
        "--allow-large-crop",
        action="store_true",
        help="Allow removal beyond --max-remove-percent",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Launch `frgs show` for the saved output; the viewer blocks until stopped",
    )
    parser.add_argument("--viewer-port", type=int, default=8080, help="Port used with --show (default: 8080)")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run and args.show:
        parser.error("--show cannot be used with --dry-run because no output is written")
    if not 1 <= args.viewer_port <= 65535:
        parser.error("--viewer-port must be between 1 and 65535")

    input_path = args.input.expanduser()
    output_path = args.output.expanduser() if args.output is not None else _default_output_path(input_path)
    try:
        crop_ply(
            input_path,
            output_path,
            x_percentiles=args.x_percentiles,
            y_percentiles=args.y_percentiles,
            z_percentiles=args.z_percentiles,
            sample_stride=args.sample_stride,
            max_quantile_samples=args.max_quantile_samples,
            device=args.device,
            dry_run=args.dry_run,
            force=args.force,
            max_remove_percent=args.max_remove_percent,
            allow_large_crop=args.allow_large_crop,
        )
        if args.show:
            _show_output(output_path, args.device, args.viewer_port)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
