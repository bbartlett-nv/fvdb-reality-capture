# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Devcontainer image for converting SplatKing LiDAR captures to COLMAP format
# via cuSFM.  Open scripts/convert_splatking/ in Cursor / VS Code and choose
# "Reopen in Container" to use this environment.
#
# Base image: same as pyCuSFM's official Docker (CUDA 12, TensorRT, Ubuntu 24.04)
FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System dependencies (pyCuSFM prerequisites)
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
    && rm -rf /var/lib/apt/lists/*

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
RUN pip3 install --no-cache-dir scipy tqdm Pillow

# ----------------------------------------------------------------------------
# Working directory (devcontainer mounts the project here)
# ----------------------------------------------------------------------------
WORKDIR /workspace
