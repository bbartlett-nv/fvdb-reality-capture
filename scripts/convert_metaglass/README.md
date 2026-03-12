# Ray-Ban Meta Glasses Video to COLMAP Conversion

Convert a Ray-Ban Meta glasses video recording into a COLMAP dataset
compatible with `frgs reconstruct`.  The script extracts frames from the
video, runs COLMAP feature extraction and matching, then uses GLOMAP (or
COLMAP) for sparse reconstruction.

## Prerequisites

- **ffmpeg** (on PATH)
- **COLMAP** (on PATH) -- for feature extraction and matching
- **GLOMAP** (on PATH) -- for global SfM mapping (or pass `--mapper colmap` to
  fall back to incremental COLMAP)
- Python packages: `opencv-python`, `numpy`, `matplotlib` (optional, for blur
  histograms)

## Quick start

```bash
python create_colmap_from_metaglass.py \
    /path/to/recording.mp4 \
    --output-dir /path/to/colmap_dataset

# Or extract frames only (no COLMAP/GLOMAP):
python create_colmap_from_metaglass.py \
    /path/to/recording.mp4 \
    --output-dir /path/to/colmap_dataset \
    --extract-only
# Frames are in /path/to/colmap_dataset/images_raw/

# Then train a 3DGS model:
frgs reconstruct /path/to/colmap_dataset -o scene.ply
```

## Script options

```
python create_colmap_from_metaglass.py --help
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `video_path` | (required) | Path to the input video file (MP4/MOV) |
| `--output-dir` | (required) | Output COLMAP dataset directory |
| `--fps` | `2.0` | Target frame extraction rate (frames per second) |
| `--image-format` | `jpeg` | Image format for extracted frames |
| `--blur-threshold` | off | Skip frames with Laplacian variance below this value |
| `--colmap-camera-model` | `OPENCV` | COLMAP camera model (OPENCV recommended for wide-angle Meta glasses lens) |
| `--matcher` | `sequential` | COLMAP matching strategy (`sequential`, `exhaustive`, or `vocab_tree`) |
| `--mapper` | `glomap` | Mapper for sparse reconstruction (`glomap` or `colmap`) |
| `--colmap-path` | auto | Explicit path to the colmap binary |
| `--glomap-path` | auto | Explicit path to the glomap binary |
| `--ffmpeg-path` | auto | Explicit path to the ffmpeg binary |
| `--extract-only` | off | Stop after frame extraction; images are written to `<output-dir>/images_raw/` |
| `--verbose` | off | Debug logging |

## Pipeline overview

```
Ray-Ban Meta video (MP4/MOV)
  |
  v
Phase 1: Extract frames at target FPS  (ffmpeg)
          Optional blur filtering       (OpenCV Laplacian)
  |
  v
Phase 2: Feature extraction             (colmap feature_extractor, SIFT)
  |
  v
Phase 3: Feature matching               (colmap sequential_matcher)
          Sparse reconstruction          (glomap mapper / colmap mapper)
  |
  v
Phase 4: Arrange COLMAP output           (images/ + sparse/0/)
  |
  v
frgs reconstruct <output-dir> -o scene.ply
```

## Tips for best results

- **Low FPS extraction**: The default 2 fps is a good starting point. Too many
  frames slow down matching without improving reconstruction. For small rooms,
  1 fps may suffice.
- **Exhaustive matching**: For small datasets (< 200 frames), use
  `--matcher exhaustive` for better matching coverage.
- **Camera model**: The Ray-Ban Meta glasses have a wide-angle lens. The default
  `OPENCV` model handles radial and tangential distortion. Do not use `PINHOLE`
  unless you have undistorted the video first.
- **Motion blur**: Head-mounted cameras are prone to motion blur during fast head
  turns. Use `--blur-threshold 50` to filter blurry frames.
- **Reconstruction tuning**: When running `frgs reconstruct`, consider
  `--tx.normalization-type pca` and `--tx.image-downsample-factor 2` for
  indoor scenes.
