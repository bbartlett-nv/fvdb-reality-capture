# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Devcontainer image for converting ZED2 .svo recordings to COLMAP format
# via cuSFM.  Open scripts/convert_zed2/ in Cursor / VS Code and choose
# "Reopen in Container" to use this environment.
#
# Base image: same as pyCuSFM's official Docker (CUDA 12, TensorRT, Ubuntu 24.04)
FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System dependencies (pyCuSFM + ZED SDK prerequisites)
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        git-lfs \
        libabsl-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libopencv-dev \
        libprotobuf-dev \
        protobuf-compiler \
        wget \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# ZED SDK 5.1 (silent install, skip CUDA since it is in the base image)
# ----------------------------------------------------------------------------
RUN wget -q -O /tmp/zed_sdk.run \
        https://download.stereolabs.com/zedsdk/5.1/cu12/ubuntu24 \
    && chmod +x /tmp/zed_sdk.run \
    && /tmp/zed_sdk.run -- silent skip_cuda skip_od_module skip_hub \
    && rm /tmp/zed_sdk.run \
    && rm -rf /var/lib/apt/lists/*

# Install ZED Python API into the system Python
RUN python3 /usr/local/zed/get_python_api.py

# ----------------------------------------------------------------------------
# pyCuSFM
# ----------------------------------------------------------------------------
RUN git lfs install \
    && git clone https://github.com/nvidia-isaac/pyCuSFM /opt/pycusfm

WORKDIR /opt/pycusfm
RUN ./setup.bash cuda12
RUN pip3 install --no-cache-dir -e .

# Ensure cusfm_cli is on PATH
ENV PATH="/root/.local/bin:${PATH}"

# ----------------------------------------------------------------------------
# Python dependencies for the conversion script
# ----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir scipy tqdm

# ----------------------------------------------------------------------------
# Working directory (devcontainer mounts the project here)
# ----------------------------------------------------------------------------
WORKDIR /workspace
