# ZED2 SVO to COLMAP Conversion

Convert a ZED2 `.svo` recording into a COLMAP dataset compatible with
`frgs reconstruct`.  By default the script exports ZED tracking poses
directly.  Pass `--refine` to enable CUDA-accelerated pose refinement via
NVIDIA cuSFM (stereo-aware, with ZED tracking-pose priors).

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
   # Default: export ZED tracking poses directly (no refinement)
   python create_colmap_from_zed2.py \
       --svo-path /data/recording.svo \
       --output-dir /data/colmap_output

   # Optional: enable cuSFM pose refinement
   python create_colmap_from_zed2.py \
       --svo-path /data/recording.svo \
       --output-dir /data/colmap_output \
       --refine
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

# Run the conversion (default: direct export, no refinement)
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
| `--refine` | off | Enable cuSFM pose refinement |
| `--min-inter-frame-distance` | `0.06` | cuSFM keyframe distance threshold (meters, requires `--refine`) |
| `--min-inter-frame-rotation` | `1.5` | cuSFM keyframe rotation threshold (degrees, requires `--refine`) |
| `--skip-cusfm` | off | With `--refine`, stop after metadata generation |
| `--with-depth` | off | With `--refine`, replace cuSFM points with ZED depth |
| `--low-light` | off | Optimize for under-exposed captures (NEURAL depth, relaxed confidence, CLAHE enhancement) |
| `--blur-threshold` | off (`50.0` with `--low-light`) | Skip frames with Laplacian variance below this value (lower = blurrier) |
| `--verbose` | off | Debug logging |

## Pipeline overview

```
ZED2 .svo
  |
  v
Phase 1: Extract stereo frames + ZED tracking poses + intrinsics  (pyzed)
  |
  |--- (default) -----> Write COLMAP sparse from ZED poses directly
  |                        |
  |--- (--refine) -----> Phase 2: Generate frames_meta.json
  |                        |
  |                      Phase 3: Run cuSFM  (CUDA-accelerated SfM)
  |                        |
  v                        v
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
