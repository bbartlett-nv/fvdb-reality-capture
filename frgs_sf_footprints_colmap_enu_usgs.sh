#!/usr/bin/env bash

# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly CLUSTER_USER_ROOT="/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_computervision/users/bbartlett"
readonly FRGS_DATASET_PATH="${CLUSTER_USER_ROOT}/data/sf_footprints_colmap_usgs_hybrid_015m_4cluster"
readonly FRGS_OUTPUT_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly FRGS_OUTPUT_PATH="${CLUSTER_USER_ROOT}/sf_footprints_usgs_hybrid015_metric_facade_every85_ds1_max125m_sh3p40_${FRGS_OUTPUT_TIMESTAMP}.ply"
readonly FRGS_LOG_PATH="${CLUSTER_USER_ROOT}/frgs_logs"

cd "${CLUSTER_USER_ROOT}/fvdb-reality-capture"

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export CUDA_LOG_FILE=stdout

NVIDIA_DRIVER_VERSION=""
if [[ -r /sys/module/nvidia/version ]]; then
    read -r NVIDIA_DRIVER_VERSION < /sys/module/nvidia/version
elif [[ -r /proc/driver/nvidia/version ]]; then
    readonly NVIDIA_VERSION_TEXT="$(</proc/driver/nvidia/version)"
    if [[ "${NVIDIA_VERSION_TEXT}" =~ Kernel[[:space:]]Module[[:space:]]+([0-9]+(\.[0-9]+)+) ]]; then
        NVIDIA_DRIVER_VERSION="${BASH_REMATCH[1]}"
    fi
fi

if [[ -z "${NVIDIA_DRIVER_VERSION}" ]]; then
    echo "Unable to read the host NVIDIA driver version from sysfs or procfs." >&2
    exit 1
fi
readonly NVIDIA_DRIVER_VERSION
readonly NVIDIA_DRIVER_MAJOR="${NVIDIA_DRIVER_VERSION%%.*}"
if [[ ! "${NVIDIA_DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
    echo "Unable to parse NVIDIA driver version: ${NVIDIA_DRIVER_VERSION}" >&2
    exit 1
fi
echo "NVIDIA driver: ${NVIDIA_DRIVER_VERSION}"

# The pinned Torch-DGX image uses CUDA 13. On pre-R580 data-center drivers,
# select its bundled forward-compatibility libraries. Do not select them on
# R580+ drivers, where the host driver is already CUDA-13-compatible.
if ((NVIDIA_DRIVER_MAJOR < 535)); then
    echo "CUDA 13 forward compatibility requires a supported R535+ data-center driver." >&2
    exit 1
elif ((NVIDIA_DRIVER_MAJOR < 580)); then
    readonly CUDA_COMPAT_PATH="/usr/local/cuda/compat"
    if [[ ! -r "${CUDA_COMPAT_PATH}/libcuda.so.1" ]]; then
        echo "CUDA 13 requires an R580+ driver or the cuda-compat-13-0 package." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${CUDA_COMPAT_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Using CUDA 13 forward-compatibility libraries for driver ${NVIDIA_DRIVER_VERSION}."
fi

# Fail before the expensive visibility/mask preprocessing if the node driver cannot
# initialize the CUDA runtime used by PyTorch and Torch-DGX.
/opt/venv/bin/python - <<'PY'
import torch
import fvdb  # noqa: F401 - imports the fVDB/Torch-DGX device extensions

print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
print(f"PrivateUse1 backend: {torch._C._get_privateuse1_backend_name()}")
for device in ("cuda:0", "dgx"):
    try:
        probe = torch.empty(1).to(device)
    except Exception as error:
        if device == "cuda:0":
            hint = f"Check the node driver/container GPU passthrough against CUDA {torch.version.cuda}."
        else:
            hint = "CUDA works but Torch-DGX does not; check its PyTorch/CUDA build compatibility."
        raise SystemExit(
            f"{device} allocation preflight failed before dataset preprocessing. {hint} {error}"
        ) from None
    del probe
    print(f"{device} allocation preflight passed.")

visible_cuda_devices = torch.cuda.device_count()
print(f"Visible CUDA devices: {visible_cuda_devices}")
if visible_cuda_devices != 8:
    raise SystemExit(
        f"This launcher requires 8 visible CUDA devices, but PyTorch sees {visible_cuda_devices}."
    )
PY

# With no arguments, resume the checkpoint below. To resume a later checkpoint, run:
#   frgs_sf_footprints_colmap_enu_usgs.sh resume /path/to/checkpoint-directory-or-file
# To start a new reconstruction with the arguments below, run:
#   frgs_sf_footprints_colmap_enu_usgs.sh reconstruct
readonly FRGS_LAUNCH_MODE="${1:-resume}"
if [[ "${FRGS_LAUNCH_MODE}" == "resume" ]]; then
    FRGS_CHECKPOINT_PATH="${2:-${CLUSTER_USER_ROOT}/frgs_logs/run_2026-08-03-10-32-05/checkpoints/00002980}"
    if [[ "${FRGS_CHECKPOINT_PATH}" != *.pt ]]; then
        FRGS_CHECKPOINT_PATH="${FRGS_CHECKPOINT_PATH%/}/reconstruct_ckpt.pt"
    fi
    readonly FRGS_CHECKPOINT_PATH

    if [[ ! -r "${FRGS_CHECKPOINT_PATH}" ]]; then
        echo "Checkpoint is not readable: ${FRGS_CHECKPOINT_PATH}" >&2
        exit 1
    fi

    echo "Resuming reconstruction from ${FRGS_CHECKPOINT_PATH}"
    exec frgs resume \
        "${FRGS_CHECKPOINT_PATH}" \
        -o "${FRGS_OUTPUT_PATH}" \
        --io.log-path "${FRGS_LOG_PATH}" \
        --device dgx \
        --verbose
elif [[ "${FRGS_LAUNCH_MODE}" != "reconstruct" ]]; then
    echo "Usage: $0 [resume [CHECKPOINT_DIRECTORY_OR_FILE] | reconstruct]" >&2
    exit 2
fi

exec frgs reconstruct \
    "${FRGS_DATASET_PATH}" \
    -o "${FRGS_OUTPUT_PATH}" \
    --io.log-path "${FRGS_LOG_PATH}" \
    --device dgx \
    --verbose \
    --tx.normalization-type none \
    --tx.image-downsample-factor 1 \
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
    --opt.insertion-grad-2d-threshold 0.85 \
    --opt.insertion-scale-3d-threshold 0.075 \
    --opt.insertion-split-factor 2 \
    --opt.deletion-opacity-threshold 0.002 \
    --opt.deletion-scale-3d-threshold 2.0 \
    --opt.reset-opacities-every-n-refinements 0 \
    --opt.use-scales-for-deletion-after-n-refinements 1000000 \
    --opt.opacity-updates-use-revised-formulation \
    --opt.max-gaussians 125000000 \
    --cfg.save-at-percent 10 50 100 \
    --cfg.no-optimize-camera-poses \
    --cfg.scene-bbox-support-sigma 3.0
