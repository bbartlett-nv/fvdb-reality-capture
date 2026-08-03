#!/usr/bin/env bash

# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly CLUSTER_USER_ROOT="/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_computervision/users/bbartlett"
readonly FRGS_DATASET_PATH="${CLUSTER_USER_ROOT}/data/sf_footprints_colmap_usgs_dsm_030m_4cluster"
readonly FRGS_RUN_NAME="sf_footprints_usgs_dsm030_metric_facade_every90_ds2_max24m_sh3p40_20260802_160312"
readonly FRGS_OUTPUT_PATH="${CLUSTER_USER_ROOT}/${FRGS_RUN_NAME}.ply"

cd "${CLUSTER_USER_ROOT}/fvdb-reality-capture"

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export CUDA_LOG_FILE=stdout

exec frgs reconstruct \
    "${FRGS_DATASET_PATH}" \
    -o "${FRGS_OUTPUT_PATH}" \
    --run-name "${FRGS_RUN_NAME}" \
    --device dgx \
    --verbose \
    --tx.normalization-type none \
    --tx.image-downsample-factor 2 \
    --tx.min-points-per-image -1 \
    --tx.crop-bbox -714.046 -587.823 -100 487.600 609.579 150 \
    --cfg.remove-gaussians-outside-scene-bbox \
    --cfg.mask-rasterization-mode bbox \
    --cfg.random-bkgd \
    --cfg.max-epochs 200 \
    --cfg.refine-start-epoch 10 \
    --cfg.refine-stop-epoch 80 \
    --cfg.refine-every-epoch 5.0 \
    --cfg.freeze-scales-after-refinement-stop \
    --cfg.no-prune-after-refinement-stop \
    --cfg.sh-degree 3 \
    --cfg.increase-sh-degree-every-epoch 40 \
    --cfg.ssim-lambda 0.2 \
    --opt.spatial-scale-mode ABSOLUTE_UNITS \
    --opt.spatial-scale-multiplier 1.0 \
    --opt.means-lr 0.005 \
    --opt.log-scales-lr 0.003 \
    --opt.quats-lr 0.002 \
    --opt.shN-lr 0.000125 \
    --opt.insertion-grad-2d-threshold-mode PERCENTILE_EVERY_ITERATION \
    --opt.insertion-grad-2d-threshold 0.9 \
    --opt.insertion-scale-3d-threshold 0.15 \
    --opt.insertion-split-factor 2 \
    --opt.deletion-opacity-threshold 0.002 \
    --opt.deletion-scale-3d-threshold 2.0 \
    --opt.reset-opacities-every-n-refinements 0 \
    --opt.use-scales-for-deletion-after-n-refinements 1000000 \
    --opt.opacity-updates-use-revised-formulation \
    --opt.max-gaussians 24000000 \
    --cfg.save-at-percent 10 25 40 50 75 100 \
    --cfg.no-optimize-camera-poses \
    --cfg.scene-bbox-support-sigma 3.0
