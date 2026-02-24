# ZED2 SVO to COLMAP Conversion (via cuSFM)

Convert a ZED2 `.svo` recording into a COLMAP dataset compatible with
`frgs reconstruct`, using NVIDIA cuSFM for CUDA-accelerated pose refinement
with ZED stereo-pair and tracking-pose priors.

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- An NVIDIA GPU
- A ZED2 `.svo` recording

## Quick start (devcontainer)

1. Open this folder (`scripts/convert_zed2/`) in Cursor or VS Code.
2. Configure the data mount -- copy the example env file and set your path:

   ```bash
   cp .devcontainer/.env.example .devcontainer/.env
   # Edit .devcontainer/.env and set DATA_DIR to the directory with your .svo files
   ```

3. When prompted, choose **Reopen in Container**.  The first build takes
   a while as it installs ZED SDK, pyCuSFM, and dependencies.
4. Inside the container terminal, run:

   ```bash
   python create_colmap_from_zed2.py \
       --svo-path /data/recording.svo \
       --output-dir /data/colmap_output
   ```

5. Back on the host, train a 3DGS model:

   ```bash
   frgs reconstruct /path/to/svo/files/colmap_output -o scene.ply
   ```

## Quick start (standalone Docker)

If you prefer not to use devcontainers, build and run directly:

```bash
# Build from the convert_zed2/ directory
docker compose -f .devcontainer/docker-compose.yml build

# Run the conversion
docker compose -f .devcontainer/docker-compose.yml run --rm devcontainer \
    python create_colmap_from_zed2.py \
        --svo-path /data/recording.svo \
        --output-dir /data/colmap_output
```

## Script options

```
python create_colmap_from_zed2.py --help
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--svo-path` | (required) | Path to the input `.svo` file |
| `--output-dir` | (required) | Output COLMAP dataset directory |
| `--frame-stride` | `5` | Extract every Nth frame |
| `--min-inter-frame-distance` | `0.06` | cuSFM keyframe distance threshold (meters) |
| `--min-inter-frame-rotation` | `1.5` | cuSFM keyframe rotation threshold (degrees) |
| `--skip-cusfm` | off | Extract frames + metadata only (skip SfM) |
| `--verbose` | off | Debug logging |

## Pipeline overview

```
ZED2 .svo
  |
  v
Phase 1: Extract stereo frames + ZED tracking poses + intrinsics  (pyzed)
  |
  v
Phase 2: Generate frames_meta.json  (cuSFM KeyframesMetadataCollection format)
  |
  v
Phase 3: Run cuSFM  (CUDA-accelerated SfM with stereo + pose priors)
  |
  v
Phase 4: Arrange COLMAP output  (images/ + sparse/0/ layout for frgs)
  |
  v
frgs reconstruct <output-dir> -o scene.ply
```

## Coordinate conventions

The ZED SDK uses a right-handed Y-up world frame (OpenGL convention).  COLMAP
and the fVDB viewer expect Z-up.  The script applies a 90-degree rotation around
the X-axis to all world-space quantities (poses and depth points) so that cameras
that were held level during capture point toward the X-Y horizon in the output,
and "up" in the real world maps to +Z.

## Environment details

The container is built on `nvcr.io/nvidia/tensorrt:24.12-py3` (Ubuntu 24.04,
CUDA 12, TensorRT 10.x) -- the same base pyCuSFM uses -- with ZED SDK 5.1
installed on top.
