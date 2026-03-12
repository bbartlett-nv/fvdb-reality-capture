# SplatKing LiDAR Capture to COLMAP Conversion

Convert a [SplatKing](https://radiancefields.com/splatking) LiDAR-mode capture
into a COLMAP dataset compatible with `frgs reconstruct`.  The script reads
ARKit camera poses and intrinsics directly from the capture metadata -- no
COLMAP feature matching or SfM is required.

## Prerequisites

- Python 3.11+
- numpy, scipy, Pillow, tqdm
- A SplatKing LiDAR-mode capture folder (e.g. `PhotoSeries_YYYYMMDD_HHMMSS/`)
- Docker with NVIDIA Container Toolkit (`nvidia-docker`) -- only when using `--refine`

## Quick start

```bash
# Default: export ARKit poses directly (no refinement)
python create_colmap_from_splatking.py \
    --input-dir /path/to/PhotoSeries_YYYYMMDD_HHMMSS \
    --output-dir /path/to/colmap_output
```

Then train a 3DGS model:

```bash
frgs reconstruct /path/to/colmap_output -o scene.ply
```

## Quick start (devcontainer -- for `--refine`)

The `--refine` flag requires pyCuSFM, which needs CUDA 12 and Ubuntu 24.04.
The easiest way to get this environment is via the included devcontainer.

1. Open this folder (`scripts/convert_splatking/`) in Cursor or VS Code.
2. Configure the data mount -- copy the example env file and set your path:

   ```bash
   cp .devcontainer/.env.example .devcontainer/.env
   # Edit .devcontainer/.env and set DATA_DIR to the directory with your captures
   ```

3. When prompted, choose **Reopen in Container**.  The first build takes
   a while as it installs pyCuSFM and dependencies.
4. Inside the container terminal, run:

   ```bash
   python create_colmap_from_splatking.py \
       --input-dir /data/PhotoSeries_YYYYMMDD_HHMMSS \
       --output-dir /data/colmap_output \
       --refine
   ```

5. Back on the host, train a 3DGS model:

   ```bash
   frgs reconstruct /path/to/colmap_output -o scene.ply
   ```

## Quick start (standalone Docker)

If you prefer not to use devcontainers, build and run directly:

```bash
# Build from the convert_splatking/ directory
docker compose -f .devcontainer/docker-compose.yml build

# Run the conversion with cuSFM refinement
docker compose -f .devcontainer/docker-compose.yml run --rm devcontainer \
    python create_colmap_from_splatking.py \
        --input-dir /data/PhotoSeries_YYYYMMDD_HHMMSS \
        --output-dir /data/colmap_output \
        --refine
```

## Script options

```
python create_colmap_from_splatking.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | (required) | Path to SplatKing session folder |
| `--output-dir` | (required) | Output COLMAP dataset directory |
| `--quality-threshold` | `0.5` | Minimum qualityScore (0-1) to include a frame |
| `--with-depth` | off | Back-project per-frame depth maps for denser points3D |
| `--depth-stride` | `4` | Subsampling stride for depth back-projection (requires `--with-depth`) |
| `--refine` | off | Enable cuSFM pose refinement (requires `cusfm_cli` on PATH) |
| `--min-inter-frame-distance` | `0.06` | cuSFM keyframe distance threshold in meters (requires `--refine`) |
| `--min-inter-frame-rotation` | `1.5` | cuSFM keyframe rotation threshold in degrees (requires `--refine`) |
| `--skip-cusfm` | off | With `--refine`, stop after metadata generation |
| `--cusfm-extra-args` | `[]` | Additional arguments for cusfm_cli (requires `--refine`) |
| `--verbose` | off | Debug logging |

## Pipeline overview

```
SplatKing LiDAR capture folder
  |
  v
Phase 1: Parse splatpack.json, filter by quality + tracking state
  |
  |--- (default) -----> Write COLMAP sparse from ARKit poses directly
  |                        |
  |--- (--refine) -----> Phase 1b: Copy rotated images to work directory
  |                        |
  |                      Phase 2: Generate frames_meta.json (ARKit pose priors)
  |                        |
  |                      Phase 3: Run cuSFM  (CUDA-accelerated SfM)
  |                        |
  v                        v
Arrange COLMAP output  (images/ + sparse/0/ layout for frgs)
  |
  v
frgs reconstruct <output-dir> -o scene.ply
```

## Input folder structure

A SplatKing LiDAR-mode capture exports a flat directory containing:

```
PhotoSeries_YYYYMMDD_HHMMSS/
    splatpack.json                       # Session manifest (schema v2)
    capture_started.json                 # Session start metadata
    photo_series.json                    # Photo series index
    lidar_pointcloud_world_xyz.bin       # Accumulated LiDAR point cloud (float32 XYZ)
    wide_YYYYMMDD_HHMMSS.jpg            # Per-frame JPEG image (1920x1440)
    wide_YYYYMMDD_HHMMSS.json           # Per-frame ARKit metadata
    wide_YYYYMMDD_HHMMSS_depth.bin      # Per-frame depth map (256x192, float32 meters)
    ...
```

## Point cloud colorization

The LiDAR point cloud (`lidar_pointcloud_world_xyz.bin`) contains only XYZ
geometry with no color data.  By default, the script automatically colorizes
the point cloud by projecting each 3D point into the camera images and
sampling RGB values.  Colors are averaged across all frames that observe a
given point, producing robust per-point colors for visualization in
`frgs show-data` and for reconstruction with `frgs reconstruct`.

## Viewing the dataset

To preview the converted dataset before training:

```bash
frgs show-data /path/to/colmap_output
```

If the scene appears upside down in the viewer, use the `-fu` (flip up) flag:

```bash
frgs show-data /path/to/colmap_output -fu
```

This is a known behavior of the PCA-based scene normalization used by the
viewer -- it can pick an ambiguous sign for the vertical axis.  The `-fu` flag
flips it to the correct orientation.  This does not affect `frgs reconstruct`.

## Coordinate conventions

SplatKing captures use ARKit's right-handed Y-up world frame with an
OpenGL-style camera convention (Y-up, -Z forward).  The script converts to:

- **World frame**: Z-up (90-degree rotation around X), matching COLMAP and the
  fVDB viewer convention.
- **Camera convention**: OpenCV/COLMAP (Y-down, Z forward).

This matches the coordinate conversion used by the ZED2 conversion script
(`scripts/convert_zed2/`), ensuring consistent orientation in the fVDB viewer.

## Depth options

By default, only the session-level LiDAR point cloud
(`lidar_pointcloud_world_xyz.bin`) is used for `points3D.bin`.  This provides
a sparse but reliable initial point cloud.

Pass `--with-depth` to also back-project per-frame depth maps (256x192,
float32 meters from the LiDAR sensor) into world coordinates.  This produces
a significantly denser point cloud.  Use `--depth-stride` to control
subsampling (default 4 = every 4th pixel in each dimension).
