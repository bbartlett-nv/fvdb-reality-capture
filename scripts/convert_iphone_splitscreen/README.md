# iPhone Split-Screen (LiDAR + RGB) to COLMAP Conversion

Convert an iPhone split-screen recording (LiDAR visualization on the left,
RGB camera on the right) into a COLMAP dataset compatible with
`frgs reconstruct`.

The script crops the RGB portion of each frame and discards the LiDAR
visualization (which is a rendered overlay, not raw depth data).  It then
runs COLMAP feature extraction and matching, followed by GLOMAP (or COLMAP)
for sparse reconstruction.

## Prerequisites

- **ffmpeg** (on PATH)
- **COLMAP** (on PATH) -- for feature extraction and matching
- **GLOMAP** (on PATH) -- for global SfM mapping (or pass `--mapper colmap` to
  fall back to incremental COLMAP)
- Python packages: `opencv-python`, `numpy`, `matplotlib` (optional, for blur
  histograms)

## Quick start

```bash
# Default: crop right half of split-screen as RGB
python create_colmap_from_iphone_splitscreen.py \
    /path/to/screen_recording.mp4 \
    --output-dir /path/to/colmap_dataset

# Or extract frames only (no COLMAP/GLOMAP):
python create_colmap_from_iphone_splitscreen.py \
    /path/to/screen_recording.mp4 \
    --output-dir /path/to/colmap_dataset \
    --extract-only
# Cropped frames are in /path/to/colmap_dataset/images_raw/

# Then train a 3DGS model:
frgs reconstruct /path/to/colmap_dataset -o scene.ply
```

## Script options

```
python create_colmap_from_iphone_splitscreen.py --help
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `video_path` | (required) | Path to the input screen recording (MP4/MOV) |
| `--output-dir` | (required) | Output COLMAP dataset directory |
| `--fps` | `2.0` | Target frame extraction rate (frames per second) |
| `--image-format` | `jpeg` | Image format for extracted frames |
| `--crop` | `right_half` | Which portion to keep: `right_half`, `left_half`, `auto`, or `manual` |
| `--crop-rect` | auto | Manual crop rectangle as `x,y,w,h` (required when `--crop manual`) |
| `--blur-threshold` | off | Skip frames with Laplacian variance below this value |
| `--colmap-camera-model` | `PINHOLE` | COLMAP camera model (PINHOLE works well for iPhone) |
| `--matcher` | `sequential` | COLMAP matching strategy (`sequential`, `exhaustive`, or `vocab_tree`) |
| `--mapper` | `glomap` | Mapper for sparse reconstruction (`glomap` or `colmap`) |
| `--colmap-path` | auto | Explicit path to the colmap binary |
| `--glomap-path` | auto | Explicit path to the glomap binary |
| `--ffmpeg-path` | auto | Explicit path to the ffmpeg binary |
| `--extract-only` | off | Stop after frame extraction; images are written to `<output-dir>/images_raw/` |
| `--verbose` | off | Debug logging |

## Crop configuration

The split-screen layout has the LiDAR visualization on the left and the RGB
camera feed on the right.  By default (`--crop right_half`), the script takes
the exact right half of the video frame.

| Mode | Behavior |
|------|----------|
| `right_half` | Crop the right 50% of the frame (default) |
| `left_half` | Crop the left 50% (use if your layout is reversed) |
| `auto` | Sample a frame and detect the vertical split boundary |
| `manual` | Use the rectangle specified by `--crop-rect x,y,w,h` |

If the split is not exactly at the midpoint (e.g., due to UI chrome or
uneven layout), use `--crop auto` or `--crop manual --crop-rect x,y,w,h`.

## Pipeline overview

```
iPhone split-screen recording (MP4/MOV)
  |
  v
Phase 1: Extract frames at target FPS    (ffmpeg)
          Crop to RGB portion             (ffmpeg crop filter)
          Optional blur filtering         (OpenCV Laplacian)
  |
  v
Phase 2: Feature extraction               (colmap feature_extractor, SIFT)
  |
  v
Phase 3: Feature matching                 (colmap sequential_matcher)
          Sparse reconstruction            (glomap mapper / colmap mapper)
  |
  v
Phase 4: Arrange COLMAP output             (images/ + sparse/0/)
  |
  v
frgs reconstruct <output-dir> -o scene.ply
```

## Limitations

- **Screen recording quality**: The video is a screen recording, not a direct
  camera capture.  This means it has been re-encoded at the display's refresh
  rate and resolution, with additional compression.  Feature matching quality
  will be lower than from a direct camera recording.
- **LiDAR data is not usable**: The left side of the split screen shows a
  rendered LiDAR visualization, not raw depth values.  There is no way to
  extract metric depth from this visualization.  The script discards it entirely.
- **Halved resolution**: Since only half the frame is usable, effective image
  resolution is reduced.
- **No ARKit poses**: The phone computed camera poses internally via ARKit, but
  these are not accessible from a screen recording.  Full COLMAP SfM is needed.
- **For better results**: Consider using **Record3D** (free iOS app) which
  exports ARKit camera poses, depth maps, and intrinsics directly in a format
  convertible to COLMAP -- avoiding all of the above limitations.

## Tips for best results

- **Low FPS extraction**: Start with 2 fps.  Screen recordings often have
  redundant frames; extracting too many wastes time without improving quality.
- **Verify the crop**: Run with `--verbose` and check the first few extracted
  frames to confirm the crop captures the full RGB view without UI elements.
- **Blur filtering**: Use `--blur-threshold 50` if the recording contains
  motion blur from fast camera movement.
- **Reconstruction tuning**: When running `frgs reconstruct`, consider
  `--tx.normalization-type pca` and `--tx.image-downsample-factor 2`.
