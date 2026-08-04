#!/usr/bin/env bash

# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Train the proven 3x3 San Francisco reconstruction with one independent FRGS
# process per GPU, then stream/core-clip the nine PLYs into one global PLY.
#
# Run this script as the batch shell (or directly from an salloc shell), not from
# a resource-holding outer srun. Example; add the site-specific account,
# partition, QOS, and container options used by your cluster:
#
#   sbatch --nodes=1 --ntasks=8 --gpus-per-node=8 --cpus-per-task=10 \
#       ./frgs_sf_footprints_colmap_enu_usgs_multigpu_chunks.sh \
#       reconstruct sf_hybrid_3x3_20260804
#
# On clusters that spell the GPU request as GRES, use --gres=gpu:8 instead.
# A rerun with the same run ID validates and reuses completed chunk PLYs.

set -euo pipefail

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
    echo "This launcher requires Bash 5.1 or newer (wait -n -p support)." >&2
    exit 1
fi

# Internal entry point used inside each Slurm step/process group. It holds a
# per-chunk advisory lock, asserts one visible GPU, and proves the exact worker
# environment can use the same bounded hybrid loader and implementation as the
# immutable plan before execing FRGS.
if [[ "${1:-}" == "__frgs_worker" ]]; then
    if (($# < 7)); then
        echo "Internal worker invocation is incomplete." >&2
        exit 2
    fi
    readonly INTERNAL_PLAN_PATH="$2"
    readonly INTERNAL_DATASET_PATH="$3"
    readonly INTERNAL_CHUNK_INDEX="$4"
    readonly INTERNAL_CHUNK_ID="$5"
    readonly INTERNAL_WORK_DIR="$6"
    readonly INTERNAL_PYTHON_BIN="$7"
    shift 7

    exec {INTERNAL_CHUNK_LOCK_FD}>"${INTERNAL_WORK_DIR}/.${INTERNAL_CHUNK_ID}.worker.lock"
    if ! flock -n "${INTERNAL_CHUNK_LOCK_FD}"; then
        echo "Another process is already training ${INTERNAL_CHUNK_ID}." >&2
        exit 75
    fi

    "${INTERNAL_PYTHON_BIN}" - \
        "${INTERNAL_PLAN_PATH}" "${INTERNAL_DATASET_PATH}" "${INTERNAL_CHUNK_INDEX}" <<'PY'
import json
import pathlib
import sys

import numpy as np
import torch

import fvdb_reality_capture
from fvdb_reality_capture.cli.frgs._reconstruct import _training_implementation_identity
from fvdb_reality_capture.sfm_scene import TracklessColmapSceneSource, probe_trackless_colmap_binary

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
dataset_path = pathlib.Path(sys.argv[2]).expanduser().resolve()
chunk_index = int(sys.argv[3])
payload = plan["payload"]
chunk = payload["chunks"][chunk_index]

if pathlib.Path(payload["dataset_path"]).resolve() != dataset_path:
    raise SystemExit("Worker dataset does not match the immutable plan")
if _training_implementation_identity() != payload["implementation"]:
    raise SystemExit("Worker Python implementation/runtime does not match the immutable plan")
package_path = pathlib.Path(fvdb_reality_capture.__file__).resolve()
print(f"FRGS worker package: {package_path}")

if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"Each worker must see exactly one CUDA GPU, but this worker sees {torch.cuda.device_count()}."
    )
probe = torch.empty(1, device="cuda")
print(f"FRGS worker GPU: {torch.cuda.get_device_name(0)}")
del probe

supported, reason = probe_trackless_colmap_binary(dataset_path)
if not supported:
    raise SystemExit(f"Bounded hybrid loader is unavailable in the worker environment: {reason}")
source = TracklessColmapSceneSource(dataset_path)
layout = {
    "point_count": source.point_count,
    "tracked_point_count": source.tracked_point_count,
    "track_observation_count": source.track_observation_count,
    "trackless_suffix_count": source.trackless_suffix_count,
}
if layout != payload["source"]["layout"] or source.fingerprint != payload["source"]["fingerprint"]:
    raise SystemExit("Worker hybrid source identity/layout does not match the immutable plan")
selected_count, strictly_increasing = source.count_points_and_check_strict_id_order(
    np.asarray(chunk["worker_train_bbox"], dtype=np.float32),
    bounds_mode="open",
)
if not strictly_increasing or selected_count != chunk["worker_open_initial_count"]:
    raise SystemExit(
        f"{chunk['id']} bounded-loader preflight mismatch: selected={selected_count:,}, "
        f"planned={chunk['worker_open_initial_count']:,}, ordered={strictly_increasing}"
    )
print(
    f"FRGS worker preflight passed: required bounded single-bbox hybrid loader; "
    f"{chunk['id']} selects {selected_count:,} points; {reason}"
)
PY
    exec "$@"
fi

usage() {

    cat <<'EOF'
Usage:
  frgs_sf_footprints_colmap_enu_usgs_multigpu_chunks.sh reconstruct [RUN_ID]
  frgs_sf_footprints_colmap_enu_usgs_multigpu_chunks.sh plan RUN_ID
  frgs_sf_footprints_colmap_enu_usgs_multigpu_chunks.sh prewarm RUN_ID
  frgs_sf_footprints_colmap_enu_usgs_multigpu_chunks.sh merge RUN_ID

Modes:
  reconstruct  Validate the immutable plan, prewarm crop masks, train every
               missing chunk in parallel, and merge all nine chunks.
  plan         Validate the hybrid COLMAP source and create/verify the plan.
  prewarm      Create/verify the plan and serially populate crop-mask caches.
  merge        Validate all nine final chunk PLYs and run only the merge.

RUN_ID defaults to FRGS_RUN_ID. A new timestamp is used only for reconstruct.
Reuse the printed RUN_ID to resume a partially completed launch.

Useful environment overrides:
  FRGS_CLUSTER_USER_ROOT   Cluster user root.
  FRGS_REPO_ROOT           fVDB Reality Capture checkout.
  FRGS_DATASET_PATH        Hybrid COLMAP dataset.
  FRGS_OUTPUT_PATH         Final merged PLY path.
  FRGS_WORK_DIR            Immutable plan and final chunk PLY directory.
  FRGS_LOG_PATH            FRGS writer logs and intermediate PLYs.
  FRGS_PYTHON              Python executable (default: /opt/venv/bin/python).
  FRGS_WORKER_BACKEND      auto, srun, or visible (default: auto).
  FRGS_MAX_PARALLEL        Concurrent workers; defaults to visible GPU count.
  FRGS_CPUS_PER_WORKER     CPUs requested by each srun step (default: 10).
  FRGS_MIN_GPU_MEMORY_GIB  Hard minimum per GPU (default: 60 GiB).
  FRGS_ALLOW_LOW_MEMORY_GPU Set to 1 to override the GPU-memory minimum.
  FRGS_GPU_TOKENS          Comma-separated CUDA_VISIBLE_DEVICES tokens for the
                           visible backend; normally inherited automatically.
  FRGS_SAVE_CHECKPOINTS    Set to 1 to save large optimizer checkpoints.
  FRGS_HASH_IMAGE_CONTENTS Set to 0 to skip the plan-time full image-byte hash.
  FRGS_EXPECTED_SOURCE_SHA256 Override the pinned hybrid points3D.bin digest.
  FRGS_FORCE_MASK_PREWARM  Set to 1 to revalidate every crop-mask cache.

Intermediate model PLYs remain enabled. For chunk N they are written below:
  FRGS_LOG_PATH/<run-name>_chunk_NNNN_<attempt>/ply/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    usage
    exit 0
fi

readonly LAUNCH_MODE="${1:-reconstruct}"
case "${LAUNCH_MODE}" in
    reconstruct | plan | prewarm | merge) ;;
    *)
        usage >&2
        exit 2
        ;;
esac

RUN_ID="${2:-${FRGS_RUN_ID:-}}"
if [[ -z "${RUN_ID}" && "${LAUNCH_MODE}" == "reconstruct" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${RUN_ID}" ]]; then
    echo "${LAUNCH_MODE} requires RUN_ID (or FRGS_RUN_ID)." >&2
    exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "RUN_ID may contain only letters, digits, periods, underscores, and hyphens: ${RUN_ID}" >&2
    exit 2
fi
readonly RUN_ID

readonly CLUSTER_USER_ROOT="${FRGS_CLUSTER_USER_ROOT:-/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_computervision/users/bbartlett}"
readonly REPO_ROOT="${FRGS_REPO_ROOT:-${CLUSTER_USER_ROOT}/fvdb-reality-capture}"
readonly DATASET_PATH="${FRGS_DATASET_PATH:-${CLUSTER_USER_ROOT}/data/sf_footprints_colmap_usgs_hybrid_015m_4cluster}"
readonly PYTHON_BIN="${FRGS_PYTHON:-/opt/venv/bin/python}"
readonly LOG_PATH="${FRGS_LOG_PATH:-${CLUSTER_USER_ROOT}/frgs_logs}"
readonly RUN_STEM="${FRGS_RUN_NAME:-sf_footprints_usgs_hybrid015_metric_facade_every95_ds1_max24m_sh3p40_24mperchunk_3x3x1_ref160_ep300_multigpu_${RUN_ID}}"
readonly OUTPUT_PATH="${FRGS_OUTPUT_PATH:-${CLUSTER_USER_ROOT}/${RUN_STEM}.ply}"
readonly WORK_DIR="${FRGS_WORK_DIR:-${OUTPUT_PATH%.ply}.multigpu_chunks}"
readonly PLAN_PATH="${WORK_DIR}/launcher_plan.json"
readonly MASK_MARKER_PATH="${WORK_DIR}/crop_masks.ready.json"
readonly WORKER_LOG_DIR="${WORK_DIR}/worker_logs"
readonly ATTEMPT_DIR="${WORK_DIR}/attempt_outputs"
readonly QUARANTINE_DIR="${WORK_DIR}/quarantine"
readonly MAX_GAUSSIANS=24000000
readonly LAUNCHER_PIPELINE_VERSION=2

if [[ "${OUTPUT_PATH}" != *.ply ]]; then
    echo "FRGS_OUTPUT_PATH must end in .ply: ${OUTPUT_PATH}" >&2
    exit 2
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "Repository directory does not exist: ${REPO_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Dataset directory does not exist: ${DATASET_PATH}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
# Invoke the FRGS entry point through this exact interpreter and repository
# PYTHONPATH; do not trust an unrelated console-script installation on PATH.
readonly FRGS_EXECUTABLE_PATH="$(readlink -f "${PYTHON_BIN}")"
if [[ ! -d "$(dirname "${OUTPUT_PATH}")" ]]; then
    echo "Output parent directory does not exist: $(dirname "${OUTPUT_PATH}")" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}" "${WORKER_LOG_DIR}" "${ATTEMPT_DIR}" "${QUARANTINE_DIR}" "${LOG_PATH}"

# One launcher owns a run. Worker processes close this descriptor before exec,
# so a killed launcher never leaves the advisory lock held by orphan trainers.
exec {LAUNCH_LOCK_FD}>"${WORK_DIR}/.launcher.lock"
if ! flock -n "${LAUNCH_LOCK_FD}"; then
    echo "Another launcher owns ${WORK_DIR}. Refusing a concurrent launch." >&2
    exit 1
fi
readonly LAUNCH_LOCK_FD

cd "${REPO_ROOT}"

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
read -r LAUNCHER_SHA256 _ < <(sha256sum "${SCRIPT_PATH}")
readonly LAUNCHER_SHA256
read -r FRGS_EXECUTABLE_SHA256 _ < <(sha256sum "${FRGS_EXECUTABLE_PATH}")
readonly FRGS_EXECUTABLE_SHA256

export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1
export CUDA_LOG_FILE=stdout

# These are exactly the settings used by the successful desktop 3x3 run. Each
# standalone worker receives its own training-halo bbox below. PLY saving stays
# enabled; only optimizer checkpoints are disabled by default because nine
# 24M-Gaussian optimizer states are substantially larger than the PLY snapshots.
TRAINING_ARGS=(
    --device cuda
    --verbose
    --use-every-n-as-val -1
    --chunked-colmap-load require
    --tx.normalization-type none
    --tx.image-downsample-factor 1
    --tx.min-points-per-image -1
    --cfg.remove-gaussians-outside-scene-bbox
    --cfg.mask-rasterization-mode bbox
    --cfg.batch-size 1
    --cfg.crops-per-image 1
    --cfg.tile-size 16
    --cfg.eps-2d 0.3
    --cfg.near-plane 0.01
    --cfg.far-plane 10000000000
    --cfg.min-radius-2d 0
    --cfg.no-antialias
    --cfg.projection-method auto
    --cfg.render-backend image_space
    --cfg.random-bkgd
    --cfg.max-epochs 300
    --cfg.refine-start-epoch 10
    --cfg.refine-stop-epoch 160
    --cfg.refine-every-epoch 5.0
    --cfg.freeze-scales-after-refinement-stop
    --cfg.prune-after-refinement-stop
    --cfg.sh-degree 3
    --cfg.increase-sh-degree-every-epoch 40
    --cfg.ssim-lambda 0.2
    --opt.spatial-scale-mode ABSOLUTE_UNITS
    --opt.spatial-scale-multiplier 1.0
    --opt.means-lr 0.005
    --opt.log-scales-lr 0.003
    --opt.quats-lr 0.002
    --opt.shN-lr 0.000125
    --opt.insertion-grad-2d-threshold-mode PERCENTILE_EVERY_ITERATION
    --opt.insertion-grad-2d-threshold 0.95
    --opt.insertion-scale-3d-threshold 0.075
    --opt.insertion-split-factor 2
    --opt.deletion-opacity-threshold 0.005
    --opt.deletion-scale-3d-threshold 2.0
    --opt.reset-opacities-every-n-refinements 0
    --opt.use-scales-for-deletion-after-n-refinements 1000000
    --opt.opacity-updates-use-revised-formulation
    --opt.max-gaussians "${MAX_GAUSSIANS}"
    --cfg.save-at-percent 20 40 60 80
    --cfg.no-optimize-camera-poses
    --cfg.scene-bbox-support-sigma 3.0
)
if [[ "${FRGS_SAVE_CHECKPOINTS:-0}" != "1" ]]; then
    TRAINING_ARGS+=(--io.no-save-checkpoints)
fi
readonly -a TRAINING_ARGS

export LAUNCH_REPO_ROOT="${REPO_ROOT}"
export LAUNCH_DATASET_PATH="${DATASET_PATH}"
export LAUNCH_OUTPUT_PATH="${OUTPUT_PATH}"
export LAUNCH_WORK_DIR="${WORK_DIR}"
export LAUNCH_PLAN_PATH="${PLAN_PATH}"
export LAUNCH_LOG_PATH="${LOG_PATH}"
export LAUNCH_RUN_STEM="${RUN_STEM}"
export LAUNCHER_SHA256
export LAUNCH_PYTHON_BIN="$(readlink -f "${PYTHON_BIN}")"
export LAUNCH_FRGS_EXECUTABLE_PATH="${FRGS_EXECUTABLE_PATH}"
export LAUNCH_FRGS_EXECUTABLE_SHA256="${FRGS_EXECUTABLE_SHA256}"
export LAUNCH_MAX_GAUSSIANS="${MAX_GAUSSIANS}"
export LAUNCH_HASH_IMAGE_CONTENTS="${FRGS_HASH_IMAGE_CONTENTS:-1}"
export LAUNCH_EXPECTED_SOURCE_POINTS="${FRGS_EXPECTED_SOURCE_POINTS:-89654246}"
export LAUNCH_EXPECTED_SOURCE_SHA256="${FRGS_EXPECTED_SOURCE_SHA256:-29ab9be28b87804b4e9217ccffc338c2e975f55e629d0307bde57f91fa16410c}"
export LAUNCH_EXPECTED_TRACKED_POINTS="${FRGS_EXPECTED_TRACKED_POINTS:-107709}"
export LAUNCH_EXPECTED_TRACK_OBSERVATIONS="${FRGS_EXPECTED_TRACK_OBSERVATIONS:-412626}"
export LAUNCH_EXPECTED_TRACKLESS_SUFFIX="${FRGS_EXPECTED_TRACKLESS_SUFFIX:-89546537}"
export LAUNCHER_PIPELINE_VERSION
export LAUNCH_EXPECTED_CHUNK_COUNTS="${FRGS_EXPECTED_CHUNK_COUNTS:-5843422,9771723,8891184,10723642,19306988,14172772,10476929,15079368,11026415}"

ensure_plan() {
    echo "Validating the hybrid loader, source identity, 3x3 geometry, and initialization counts."
    "${PYTHON_BIN}" - "${TRAINING_ARGS[@]}" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

from fvdb_reality_capture.cli.frgs._reconstruct import (
    _scene_resume_fingerprint,
    _training_implementation_identity,
)
from fvdb_reality_capture.sfm_scene import TracklessColmapSceneSource, probe_trackless_colmap_binary
from fvdb_reality_capture.spatial_chunking import plan_spatial_chunks
from fvdb_reality_capture.transforms import DownsampleImages

DOMAIN_BBOX = (-714.046, -587.823, -100.0, 487.600, 609.579, 150.0)
GRID = (3, 3, 1)
OVERLAP = 0.10


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_expected_counts(text):
    if not text.strip():
        return None
    return tuple(int(value) for value in text.split(","))


dataset_path = pathlib.Path(os.environ["LAUNCH_DATASET_PATH"]).expanduser().resolve()
output_path_unresolved = pathlib.Path(os.environ["LAUNCH_OUTPUT_PATH"]).expanduser()
output_path = output_path_unresolved.parent.resolve() / output_path_unresolved.name
work_dir = pathlib.Path(os.environ["LAUNCH_WORK_DIR"]).expanduser().resolve()
plan_path = pathlib.Path(os.environ["LAUNCH_PLAN_PATH"]).expanduser().resolve()
log_path = pathlib.Path(os.environ["LAUNCH_LOG_PATH"]).expanduser().resolve()
max_gaussians = int(os.environ["LAUNCH_MAX_GAUSSIANS"])
hash_image_contents = os.environ["LAUNCH_HASH_IMAGE_CONTENTS"] != "0"

supported, probe_reason = probe_trackless_colmap_binary(dataset_path)
if not supported:
    raise SystemExit(f"The bounded hybrid COLMAP loader is unavailable: {probe_reason}")

source = TracklessColmapSceneSource(dataset_path)
expected_layout = {
    "point_count": int(os.environ["LAUNCH_EXPECTED_SOURCE_POINTS"]),
    "tracked_point_count": int(os.environ["LAUNCH_EXPECTED_TRACKED_POINTS"]),
    "track_observation_count": int(os.environ["LAUNCH_EXPECTED_TRACK_OBSERVATIONS"]),
    "trackless_suffix_count": int(os.environ["LAUNCH_EXPECTED_TRACKLESS_SUFFIX"]),
}
actual_layout = {
    "point_count": source.point_count,
    "tracked_point_count": source.tracked_point_count,
    "track_observation_count": source.track_observation_count,
    "trackless_suffix_count": source.trackless_suffix_count,
}
if actual_layout != expected_layout:
    raise SystemExit(
        "Hybrid COLMAP layout mismatch. The cluster checkout may lack the true-hybrid loader, or the source is not "
        f"the verified model. Expected {expected_layout}; found {actual_layout}."
    )

ordered_count, strictly_increasing = source.count_points_and_check_strict_id_order()
if ordered_count != source.point_count or not strictly_increasing:
    raise SystemExit(
        "The standalone bbox fast path requires all COLMAP point IDs to be strictly increasing; "
        f"count={ordered_count:,}, source={source.point_count:,}, ordered={strictly_increasing}."
    )

chunks = plan_spatial_chunks(DOMAIN_BBOX, GRID, OVERLAP)
initial_counts = tuple(source.count_points(np.asarray(chunk.train_bbox), bounds_mode="closed") for chunk in chunks)
expected_counts = parse_expected_counts(os.environ["LAUNCH_EXPECTED_CHUNK_COUNTS"])
if expected_counts is not None and initial_counts != expected_counts:
    raise SystemExit(
        "Native closed-bound chunk counts do not match the verified 3x3 model. "
        f"Expected {expected_counts}; found {initial_counts}."
    )
if max(initial_counts) > max_gaussians:
    raise SystemExit(
        f"Largest initialization has {max(initial_counts):,} points, above the per-chunk "
        f"max_gaussians={max_gaussians:,}."
    )

# The standalone single-bbox fast path uses strict/open float32 crop bounds,
# whereas native chunk loading uses closed bounds. Moving each face outward by
# one float32 ULP preserves closed-bound points. This is less than a tenth of a
# millimeter here and does not materially alter projected crop masks.
chunk_documents = []
for chunk, initial_count in zip(chunks, initial_counts, strict=True):
    worker_bbox = np.asarray(chunk.train_bbox, dtype=np.float32).copy()
    worker_bbox[:3] = np.nextafter(worker_bbox[:3], np.float32(-np.inf))
    worker_bbox[3:] = np.nextafter(worker_bbox[3:], np.float32(np.inf))
    worker_initial_count = source.count_points(worker_bbox, bounds_mode="open")
    chunk_documents.append(
        {
            "index": chunk.index,
            "id": chunk.id,
            "grid_index": list(chunk.grid_index),
            "core_bbox": list(chunk.core_bbox),
            "train_bbox": list(chunk.train_bbox),
            "worker_train_bbox": [float(value) for value in worker_bbox],
            "inclusive_max": list(chunk.inclusive_max),
            "native_closed_initial_count": initial_count,
            "worker_open_initial_count": worker_initial_count,
            "artifact_path": str(work_dir / f"{chunk.id}.ply"),
            "receipt_path": str(work_dir / f"{chunk.id}.receipt.json"),
        }
    )
    if worker_initial_count > max_gaussians:
        raise SystemExit(
            f"{chunk.id} standalone worker selects {worker_initial_count:,} points, above "
            f"max_gaussians={max_gaussians:,}."
        )

source_full_sha256 = source.full_fingerprint
expected_source_sha256 = os.environ["LAUNCH_EXPECTED_SOURCE_SHA256"].strip()
if expected_source_sha256 and source_full_sha256 != expected_source_sha256:
    raise SystemExit(
        "Hybrid points3D.bin SHA-256 mismatch. "
        f"Expected {expected_source_sha256}; found {source_full_sha256}."
    )

metadata_scene = source.metadata_scene(np.asarray(DOMAIN_BBOX, dtype=np.float64))
metadata_scene = DownsampleImages(image_downsample_factor=1, rescaled_jpeg_quality=95)(metadata_scene)
scene_fingerprint = _scene_resume_fingerprint(
    metadata_scene,
    include_points=False,
    include_image_contents=hash_image_contents,
)

payload = {
    "schema": "frgs_multigpu_chunk_launcher_v1",
    "launcher_pipeline_version": int(os.environ["LAUNCHER_PIPELINE_VERSION"]),
    "dataset_path": str(dataset_path),
    "output_path": str(output_path),
    "merge_receipt_path": str(work_dir / "merged_output.receipt.json"),
    "work_dir": str(work_dir),
    "log_path": str(log_path),
    "run_stem": os.environ["LAUNCH_RUN_STEM"],
    "domain_bbox": list(DOMAIN_BBOX),
    "grid": list(GRID),
    "overlap_pct": OVERLAP,
    "max_gaussians": max_gaussians,
    "selection_bridge": "native_closed_via_standalone_open_float32_nextafter_v1",
    "training_cli_args": sys.argv[1:],
    "merge_metadata_config": {
        "image_downsample_factor": 1,
        "eps_2d": 0.3,
        "near_plane": 0.01,
        "far_plane": 1.0e10,
        "min_radius_2d": 0.0,
        "antialias": False,
        "tile_size": 16,
        "projection_method": "auto",
        "render_backend": "image_space",
    },
    "source": {
        "loader": "mixed_track_binary_colmap_partial_v2",
        "probe_reason": probe_reason,
        "fingerprint": source.fingerprint,
        "full_sha256": source_full_sha256,
        "layout": actual_layout,
        "metadata_and_images": scene_fingerprint,
        "full_image_contents_hashed": hash_image_contents,
    },
    "implementation": _training_implementation_identity(),
    "worker_runtime": {
        "python_path": os.environ["LAUNCH_PYTHON_BIN"],
        "entry_point": "python -c fvdb_reality_capture.cli.frgs:frgs",
        "python_executable_path": os.environ["LAUNCH_FRGS_EXECUTABLE_PATH"],
        "python_executable_sha256": os.environ["LAUNCH_FRGS_EXECUTABLE_SHA256"],
    },
    "chunks": chunk_documents,
}
signature = canonical_sha256(payload)
document = {
    "schema_version": 1,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "signature_sha256": signature,
    "launcher_file_sha256": os.environ["LAUNCHER_SHA256"],
    "payload": payload,
}

if plan_path.exists():
    try:
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Existing launcher plan is unreadable: {plan_path}: {error}") from error
    if existing.get("signature_sha256") != signature or existing.get("payload") != payload:
        raise SystemExit(
            "Existing launcher plan is incompatible with the current source, code, paths, or training settings: "
            f"{plan_path}\nexisting signature: {existing.get('signature_sha256')}\ncurrent signature:  {signature}\n"
            "Choose the original checkout/settings or use a new RUN_ID/work directory."
        )
    print(f"Verified existing immutable plan: {plan_path}")
else:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{plan_path.name}.", suffix=".tmp", dir=plan_path.parent, delete=False
        ) as temporary_file:
            temporary_path = pathlib.Path(temporary_file.name)
            json.dump(document, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, plan_path)
        temporary_path = None
        directory_fd = os.open(plan_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"Created immutable plan: {plan_path}")

print(f"Hybrid source: {source.point_count:,} points; {probe_reason}")
for chunk, count, chunk_document in zip(chunks, initial_counts, chunk_documents, strict=True):
    worker_count = chunk_document["worker_open_initial_count"]
    delta = worker_count - count
    print(
        f"  {chunk.id} grid={chunk.grid_index}: native={count:,}; standalone={worker_count:,}; delta={delta:+,}"
    )
print(f"Halo-inclusive initialization total: {sum(initial_counts):,}; maximum: {max(initial_counts):,}")
print(f"Plan signature: {signature}")
PY
}

prewarm_crop_masks() {
    if [[ "${FRGS_FORCE_MASK_PREWARM:-0}" != "1" && -r "${MASK_MARKER_PATH}" ]]; then
        if "${PYTHON_BIN}" - "${PLAN_PATH}" "${MASK_MARKER_PATH}" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
marker = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
raise SystemExit(0 if marker.get("plan_signature_sha256") == plan.get("signature_sha256") else 1)
PY
        then
            echo "Crop-mask prewarm already completed for this plan."
            return
        fi
    fi

    echo "Serially generating/validating nine crop-mask caches before GPU training."
    "${PYTHON_BIN}" - "${PLAN_PATH}" "${MASK_MARKER_PATH}" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

from fvdb_reality_capture.sfm_scene import TracklessColmapSceneSource
from fvdb_reality_capture.transforms import CropScene, DownsampleImages

plan_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
marker_path = pathlib.Path(sys.argv[2]).expanduser().resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
payload = plan["payload"]
source = TracklessColmapSceneSource(payload["dataset_path"])
scene = source.metadata_scene(np.asarray(payload["domain_bbox"], dtype=np.float64))
scene = DownsampleImages(
    image_downsample_factor=int(payload["merge_metadata_config"]["image_downsample_factor"]),
    rescaled_jpeg_quality=95,
)(scene)

for chunk in payload["chunks"]:
    bbox = np.asarray(chunk["worker_train_bbox"], dtype=np.float64)
    print(f"Prewarming {chunk['id']} crop masks for {bbox.tolist()}", flush=True)
    CropScene(bbox=bbox)(scene)

marker = {
    "schema_version": 1,
    "plan_signature_sha256": plan["signature_sha256"],
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{marker_path.name}.", suffix=".tmp", dir=marker_path.parent, delete=False
    ) as temporary_file:
        temporary_path = pathlib.Path(temporary_file.name)
        json.dump(marker, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, marker_path)
    temporary_path = None
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
print(f"Crop-mask prewarm complete: {marker_path}")
PY
}

configure_cuda_compatibility() {
    local driver_version=""
    local version_text=""
    if [[ -r /sys/module/nvidia/version ]]; then
        read -r driver_version < /sys/module/nvidia/version
    elif [[ -r /proc/driver/nvidia/version ]]; then
        version_text="$(</proc/driver/nvidia/version)"
        if [[ "${version_text}" =~ Kernel[[:space:]]Module[[:space:]]+([0-9]+(\.[0-9]+)+) ]]; then
            driver_version="${BASH_REMATCH[1]}"
        fi
    fi
    if [[ -z "${driver_version}" ]]; then
        echo "Unable to read the host NVIDIA driver version from sysfs or procfs." >&2
        return 1
    fi

    local driver_major="${driver_version%%.*}"
    if [[ ! "${driver_major}" =~ ^[0-9]+$ ]]; then
        echo "Unable to parse NVIDIA driver version: ${driver_version}" >&2
        return 1
    fi
    echo "NVIDIA driver: ${driver_version}"

    # The pinned cluster image uses CUDA 13. Select its bundled forward-
    # compatibility library only on supported pre-R580 data-center drivers.
    if ((driver_major < 535)); then
        echo "CUDA 13 forward compatibility requires a supported R535+ data-center driver." >&2
        return 1
    elif ((driver_major < 580)); then
        local compat_path="/usr/local/cuda/compat"
        if [[ ! -r "${compat_path}/libcuda.so.1" ]]; then
            echo "CUDA 13 requires an R580+ driver or the cuda-compat-13-0 package." >&2
            return 1
        fi
        export LD_LIBRARY_PATH="${compat_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo "Using CUDA 13 forward-compatibility libraries for driver ${driver_version}."
    fi
}

cuda_preflight() {
    local preflight_output
    preflight_output="$("${PYTHON_BIN}" - <<'PY'
import os
import sys

import fvdb  # noqa: F401 - initialize the fVDB CUDA extensions
import torch

count = torch.cuda.device_count()
minimum_memory_gib = float(os.environ.get("FRGS_MIN_GPU_MEMORY_GIB", "60"))
allow_low_memory = os.environ.get("FRGS_ALLOW_LOW_MEMORY_GPU", "0") == "1"
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
print(f"Visible CUDA devices: {count}")
if count < 1:
    raise SystemExit("No CUDA devices are visible inside this allocation/container.")
for index in range(count):
    try:
        probe = torch.empty(1, device=f"cuda:{index}")
    except Exception as error:
        raise SystemExit(f"cuda:{index} allocation preflight failed: {error}") from None
    properties = torch.cuda.get_device_properties(index)
    gib = properties.total_memory / (1024**3)
    print(f"  cuda:{index}: {properties.name}; {gib:.1f} GiB")
    if gib < minimum_memory_gib:
        message = (
            f"cuda:{index} has {gib:.1f} GiB, below FRGS_MIN_GPU_MEMORY_GIB={minimum_memory_gib:.1f}; "
            "a 24M full-resolution worker may run out of memory."
        )
        if allow_low_memory:
            print(f"WARNING: {message}", file=sys.stderr)
        else:
            raise SystemExit(f"{message} Set FRGS_ALLOW_LOW_MEMORY_GPU=1 to override.")
    del probe
print(f"FRGS_VISIBLE_GPU_COUNT={count}")
PY
)"
    printf '%s\n' "${preflight_output}"
    VISIBLE_GPU_COUNT="${preflight_output##*FRGS_VISIBLE_GPU_COUNT=}"
    if [[ ! "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ || "${VISIBLE_GPU_COUNT}" -lt 1 ]]; then
        echo "Unable to parse the CUDA preflight device count." >&2
        return 1
    fi
    export VISIBLE_GPU_COUNT
}

declare -A WORKER_BBOXES=()
load_worker_bboxes() {
    local bbox_lines
    bbox_lines="$("${PYTHON_BIN}" - "${PLAN_PATH}" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for chunk in plan["payload"]["chunks"]:
    values = " ".join(format(float(value), ".17g") for value in chunk["worker_train_bbox"])
    print(f"{chunk['index']}\t{values}")
PY
)"
    local index bbox
    while IFS=$'\t' read -r index bbox; do
        [[ -n "${index}" ]] || continue
        WORKER_BBOXES["${index}"]="${bbox}"
    done <<< "${bbox_lines}"
    if ((${#WORKER_BBOXES[@]} != 9)); then
        echo "Expected nine worker bboxes in ${PLAN_PATH}; found ${#WORKER_BBOXES[@]}." >&2
        return 1
    fi
}

# Machine-readable helper records are tagged because fVDB's native libraries can
# emit CUDA diagnostics on stdout. Untagged lines are forwarded as diagnostics.
# Reuse requires a plan-bound receipt and a fresh full SHA-256 verification.
pending_chunks() {
    "${PYTHON_BIN}" - "${PLAN_PATH}" "${QUARANTINE_DIR}" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys

from fvdb_reality_capture.tools import validate_gaussian_ply_file

HASH_BLOCK_BYTES = 8 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def quarantine(paths, chunk, reason):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    moved = []
    for sequence, candidate in enumerate(paths):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        destination = quarantine_dir / (
            f"{candidate.name}.invalid.{timestamp}.{os.getpid()}.{sequence}"
        )
        os.replace(candidate, destination)
        moved.append(str(destination))
    print(
        f"Quarantined untrusted/invalid {chunk['id']} files {moved}: {reason}",
        file=sys.stderr,
    )


plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = plan["payload"]
plan_signature = plan["signature_sha256"]
quarantine_dir = pathlib.Path(sys.argv[2]).expanduser().resolve()
max_gaussians = int(payload["max_gaussians"])

for chunk in payload["chunks"]:
    path = pathlib.Path(chunk["artifact_path"])
    receipt_path = pathlib.Path(chunk["receipt_path"])
    if not path.exists() and not path.is_symlink():
        if receipt_path.exists() or receipt_path.is_symlink():
            quarantine((receipt_path,), chunk, "receipt exists without its artifact")
        print(f"FRGS_PENDING_INDEX={chunk['index']}")
        continue

    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact must be a regular non-symlink file")
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("matching completion receipt is missing or is not a regular file")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        count = validate_gaussian_ply_file(path)
        if not 1 <= count <= max_gaussians:
            raise ValueError(f"final Gaussian count {count:,} is outside [1, {max_gaussians:,}]")
        size = path.stat().st_size
        artifact_sha256 = file_sha256(path)
        expected = {
            "schema_version": 1,
            "plan_signature_sha256": plan_signature,
            "chunk_id": chunk["id"],
            "chunk_index": chunk["index"],
            "artifact_path": str(path.resolve()),
            "artifact_size": size,
            "artifact_sha256": artifact_sha256,
            "initial_count": chunk["worker_open_initial_count"],
            "final_count": count,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(
                    f"receipt field {key!r} mismatch: expected {value!r}, found {receipt.get(key)!r}"
                )
    except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        quarantine((path, receipt_path), chunk, error)
        print(f"FRGS_PENDING_INDEX={chunk['index']}")
    else:
        print(
            f"Reusing {chunk['id']}: {count:,} Gaussians; SHA-256 {artifact_sha256}; {path}",
            file=sys.stderr,
        )
PY
}

parse_pending_output() {
    local output="$1"
    local line
    PENDING_INDICES=()
    while IFS= read -r line; do
        if [[ "${line}" =~ ^FRGS_PENDING_INDEX=([0-8])$ ]]; then
            PENDING_INDICES+=("${BASH_REMATCH[1]}")
        elif [[ -n "${line}" ]]; then
            printf '%s\n' "${line}" >&2
        fi
    done <<< "${output}"
}

validate_attempt() {
    local attempt_path="$1"
    local raw_output line record=""
    if ! raw_output="$("${PYTHON_BIN}" - "${attempt_path}" "${MAX_GAUSSIANS}" <<'PY'
import hashlib
import os
import pathlib
import sys

from fvdb_reality_capture.tools import validate_gaussian_ply_file

path = pathlib.Path(sys.argv[1]).expanduser().resolve()
max_gaussians = int(sys.argv[2])
if path.is_symlink() or not path.is_file():
    raise SystemExit(f"Attempt artifact must be a regular non-symlink file: {path}")
count = validate_gaussian_ply_file(path)
if not 1 <= count <= max_gaussians:
    raise SystemExit(f"Attempt Gaussian count {count:,} is outside [1, {max_gaussians:,}]")

before = path.stat()
digest = hashlib.sha256()
with path.open("rb") as artifact:
    for block in iter(lambda: artifact.read(8 * 1024 * 1024), b""):
        digest.update(block)
    os.fsync(artifact.fileno())
after = path.stat()
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if identity(before) != identity(after):
    raise SystemExit(f"Attempt artifact changed while it was being hashed: {path}")
print(f"FRGS_ATTEMPT_RECORD={count}\t{digest.hexdigest()}\t{after.st_size}")
PY
)"; then
        [[ -n "${raw_output}" ]] && printf '%s\n' "${raw_output}" >&2
        return 1
    fi
    while IFS= read -r line; do
        if [[ "${line}" == FRGS_ATTEMPT_RECORD=* ]]; then
            record="${line#FRGS_ATTEMPT_RECORD=}"
        elif [[ -n "${line}" ]]; then
            printf '%s\n' "${line}" >&2
        fi
    done <<< "${raw_output}"
    if [[ -z "${record}" ]]; then
        echo "Attempt validator did not emit a tagged result." >&2
        return 1
    fi
    printf '%s\n' "${record}"
}

write_chunk_receipt() {
    local chunk_index="$1"
    local artifact_path="$2"
    local final_count="$3"
    local artifact_sha256="$4"
    local artifact_size="$5"
    "${PYTHON_BIN}" - \
        "${PLAN_PATH}" "${chunk_index}" "${artifact_path}" "${final_count}" \
        "${artifact_sha256}" "${artifact_size}" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
chunk_index = int(sys.argv[2])
artifact_path = pathlib.Path(sys.argv[3]).expanduser().resolve()
final_count = int(sys.argv[4])
artifact_sha256 = sys.argv[5]
artifact_size = int(sys.argv[6])
chunk = plan["payload"]["chunks"][chunk_index]
receipt_path = pathlib.Path(chunk["receipt_path"])

if receipt_path.exists() or receipt_path.is_symlink():
    raise SystemExit(f"Refusing to replace existing chunk receipt: {receipt_path}")
if artifact_path != pathlib.Path(chunk["artifact_path"]).resolve():
    raise SystemExit("Published artifact path does not match the immutable plan")
if artifact_path.is_symlink() or not artifact_path.is_file():
    raise SystemExit(f"Published artifact is not a regular file: {artifact_path}")
if artifact_path.stat().st_size != artifact_size:
    raise SystemExit("Published artifact size changed before receipt creation")

receipt = {
    "schema_version": 1,
    "plan_signature_sha256": plan["signature_sha256"],
    "chunk_id": chunk["id"],
    "chunk_index": chunk["index"],
    "artifact_path": str(artifact_path),
    "artifact_size": artifact_size,
    "artifact_sha256": artifact_sha256,
    "initial_count": chunk["worker_open_initial_count"],
    "final_count": final_count,
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{receipt_path.name}.", suffix=".tmp",
        dir=receipt_path.parent, delete=False
    ) as temporary_file:
        temporary_path = pathlib.Path(temporary_file.name)
        json.dump(receipt, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.link(temporary_path, receipt_path)
    temporary_path.unlink()
    temporary_path = None
    directory_fd = os.open(receipt_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
print(f"Published plan-bound receipt: {receipt_path}")
PY
}

merge_chunks() {
    echo "Validating and streaming all nine core-owned chunk PLYs into ${OUTPUT_PATH}."
    "${PYTHON_BIN}" - "${PLAN_PATH}" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

from fvdb_reality_capture.cli.frgs._reconstruct import Reconstruct
from fvdb_reality_capture.sfm_scene import TracklessColmapSceneSource
from fvdb_reality_capture.tools import GaussianPlyMergeSource, merge_gaussian_ply_files, validate_gaussian_ply_file
from fvdb_reality_capture.transforms import DownsampleImages


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt_no_replace(path, document):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = pathlib.Path(temporary_file.name)
            json.dump(document, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.link(temporary_path, path)
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = plan["payload"]
plan_signature = plan["signature_sha256"]
output_path = pathlib.Path(payload["output_path"])
merge_receipt_path = pathlib.Path(payload["merge_receipt_path"])

sources = []
chunk_bindings = []
for chunk in payload["chunks"]:
    artifact_path = pathlib.Path(chunk["artifact_path"])
    receipt_path = pathlib.Path(chunk["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    count = validate_gaussian_ply_file(artifact_path, expected_vertex_count=receipt["final_count"])
    if receipt.get("plan_signature_sha256") != plan_signature:
        raise SystemExit(f"Receipt for {chunk['id']} is not bound to this plan")
    if receipt.get("artifact_path") != str(artifact_path.resolve()):
        raise SystemExit(f"Receipt path mismatch for {chunk['id']}")
    if receipt.get("artifact_size") != artifact_path.stat().st_size:
        raise SystemExit(f"Receipt size mismatch for {chunk['id']}")
    binding = {
        "chunk_id": chunk["id"],
        "artifact_sha256": receipt["artifact_sha256"],
        "artifact_size": receipt["artifact_size"],
        "final_count": count,
    }
    chunk_bindings.append(binding)
    print(f"Validated {chunk['id']}: {count:,} Gaussians; SHA-256 {receipt['artifact_sha256']}")
    sources.append(
        GaussianPlyMergeSource(
            artifact_path,
            core_bbox=tuple(chunk["core_bbox"]),
            inclusive_max=tuple(chunk["inclusive_max"]),
        )
    )
chunk_bindings_sha256 = canonical_sha256(chunk_bindings)

if output_path.exists() or output_path.is_symlink():
    if output_path.is_symlink():
        raise SystemExit(
            f"Merged output path is a symlink and will not be trusted or replaced: {output_path}. "
            "Move it aside before retrying."
        )
    if merge_receipt_path.is_symlink() or not merge_receipt_path.is_file():
        raise SystemExit(
            f"Merged output exists without a plan-bound receipt: {output_path}. Move it aside before retrying."
        )
    receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
    count = validate_gaussian_ply_file(output_path)
    output_sha256 = file_sha256(output_path)
    expected = {
        "schema_version": 1,
        "plan_signature_sha256": plan_signature,
        "chunk_bindings_sha256": chunk_bindings_sha256,
        "output_path": str(output_path.resolve()),
        "output_size": output_path.stat().st_size,
        "output_sha256": output_sha256,
        "output_count": count,
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise SystemExit(
            f"Merged output receipt mismatch in {mismatches}: {merge_receipt_path}. "
            "Move the stale output and receipt aside before retrying."
        )
    print(f"Merged output already exists and is plan-bound: {output_path} ({count:,} Gaussians)")
    raise SystemExit(0)
if merge_receipt_path.exists() or merge_receipt_path.is_symlink():
    raise SystemExit(f"Stale merged receipt exists without its output: {merge_receipt_path}. Move it aside before retrying.")

render = payload["merge_metadata_config"]
source = TracklessColmapSceneSource(payload["dataset_path"])
metadata_scene = source.metadata_scene(np.asarray(payload["domain_bbox"], dtype=np.float64))
metadata_scene = DownsampleImages(
    image_downsample_factor=int(render["image_downsample_factor"]),
    rescaled_jpeg_quality=95,
)(metadata_scene)

command = Reconstruct(dataset_path=pathlib.Path(payload["dataset_path"]), out_path=output_path)
command.cfg.eps_2d = float(render["eps_2d"])
command.cfg.near_plane = float(render["near_plane"])
command.cfg.far_plane = float(render["far_plane"])
command.cfg.min_radius_2d = float(render["min_radius_2d"])
command.cfg.antialias = bool(render["antialias"])
command.cfg.tile_size = int(render["tile_size"])
command.cfg.projection_method = render["projection_method"]
command.cfg.render_backend = render["render_backend"]

result = merge_gaussian_ply_files(
    sources,
    output_path,
    metadata=command._merged_reconstruction_metadata(metadata_scene),
)
validated_count = validate_gaussian_ply_file(output_path, expected_vertex_count=result.output_gaussians)
output_sha256 = file_sha256(output_path)
receipt = {
    "schema_version": 1,
    "plan_signature_sha256": plan_signature,
    "chunk_bindings_sha256": chunk_bindings_sha256,
    "chunk_bindings": chunk_bindings,
    "output_path": str(output_path.resolve()),
    "output_size": output_path.stat().st_size,
    "output_sha256": output_sha256,
    "output_count": validated_count,
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
write_receipt_no_replace(merge_receipt_path, receipt)
print(
    f"Merged {result.input_gaussians:,} trained Gaussians into {validated_count:,} core-owned Gaussians; "
    f"filtered {result.filtered_gaussians:,} halo Gaussians."
)
print(f"Final output: {output_path}; SHA-256 {output_sha256}")
print(f"Plan-bound merged receipt: {merge_receipt_path}")
PY
}


echo "Run ID:      ${RUN_ID}"
echo "Mode:        ${LAUNCH_MODE}"
echo "Dataset:     ${DATASET_PATH}"
echo "Output:      ${OUTPUT_PATH}"
echo "Chunk work:  ${WORK_DIR}"
echo "Writer logs: ${LOG_PATH}"

# Configure CUDA 13 compatibility before planner imports on actual training
# launches. Plan/merge-only modes remain usable in CPU-only environments.
if [[ "${LAUNCH_MODE}" == "reconstruct" ]]; then
    configure_cuda_compatibility
fi

ensure_plan
if [[ "${LAUNCH_MODE}" == "plan" ]]; then
    exit 0
fi

if [[ "${LAUNCH_MODE}" == "prewarm" || "${LAUNCH_MODE}" == "reconstruct" ]]; then
    prewarm_crop_masks
fi
if [[ "${LAUNCH_MODE}" == "prewarm" ]]; then
    exit 0
fi

PENDING_OUTPUT="$(pending_chunks)"
parse_pending_output "${PENDING_OUTPUT}"

if [[ "${LAUNCH_MODE}" == "merge" ]]; then
    if ((${#PENDING_INDICES[@]} > 0)); then
        echo "Cannot merge; missing/invalid chunk indices: ${PENDING_INDICES[*]}" >&2
        exit 1
    fi
    merge_chunks
    exit 0
fi

if ((${#PENDING_INDICES[@]} == 0)); then
    echo "All final chunk PLYs are already valid; proceeding directly to merge."
    merge_chunks
    exit 0
fi

cuda_preflight
load_worker_bboxes

WORKER_BACKEND="${FRGS_WORKER_BACKEND:-auto}"
if [[ "${WORKER_BACKEND}" == "auto" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" && ( -z "${SLURM_STEP_ID:-}" || "${SLURM_STEP_ID}" == "batch" ) ]] \
        && command -v srun >/dev/null 2>&1; then
        WORKER_BACKEND="srun"
    else
        WORKER_BACKEND="visible"
    fi
fi
if [[ "${WORKER_BACKEND}" != "srun" && "${WORKER_BACKEND}" != "visible" ]]; then
    echo "FRGS_WORKER_BACKEND must be auto, srun, or visible; got ${WORKER_BACKEND}." >&2
    exit 2
fi
if [[ "${WORKER_BACKEND}" == "srun" && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "The srun backend requires an active Slurm allocation (SLURM_JOB_ID is unset)." >&2
    exit 1
fi
if [[ "${WORKER_BACKEND}" == "visible" ]] && ! command -v setsid >/dev/null 2>&1; then
    echo "The visible backend requires setsid so the complete worker process group can be terminated safely." >&2
    exit 1
fi

GPU_TOKENS=()
if [[ "${WORKER_BACKEND}" == "visible" ]]; then
    if [[ -n "${FRGS_GPU_TOKENS:-}" ]]; then
        IFS=',' read -r -a GPU_TOKENS <<< "${FRGS_GPU_TOKENS}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a GPU_TOKENS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        for ((index = 0; index < VISIBLE_GPU_COUNT; index++)); do
            GPU_TOKENS+=("${index}")
        done
    fi
    if ((${#GPU_TOKENS[@]} < 1)); then
        echo "No CUDA_VISIBLE_DEVICES tokens are available for the visible backend." >&2
        exit 1
    fi
fi

MAX_PARALLEL="${FRGS_MAX_PARALLEL:-${VISIBLE_GPU_COUNT}}"
if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FRGS_MAX_PARALLEL must be a positive integer: ${MAX_PARALLEL}" >&2
    exit 2
fi
if ((MAX_PARALLEL > VISIBLE_GPU_COUNT)); then
    MAX_PARALLEL="${VISIBLE_GPU_COUNT}"
fi
if [[ "${WORKER_BACKEND}" == "visible" && "${MAX_PARALLEL}" -gt "${#GPU_TOKENS[@]}" ]]; then
    MAX_PARALLEL="${#GPU_TOKENS[@]}"
fi
if ((MAX_PARALLEL > 9)); then
    MAX_PARALLEL=9
fi
readonly MAX_PARALLEL

readonly CPUS_PER_WORKER="${FRGS_CPUS_PER_WORKER:-10}"
if [[ ! "${CPUS_PER_WORKER}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FRGS_CPUS_PER_WORKER must be a positive integer: ${CPUS_PER_WORKER}" >&2
    exit 2
fi
if ((CPUS_PER_WORKER < 9)) && [[ "${FRGS_ALLOW_CPU_OVERSUBSCRIPTION:-0}" != "1" ]]; then
    echo "Each worker uses one trainer plus eight DataLoader workers; FRGS_CPUS_PER_WORKER must be at least 9." >&2
    echo "Set FRGS_ALLOW_CPU_OVERSUBSCRIPTION=1 only if the cluster CPU policy requires it." >&2
    exit 2
fi
readonly WORKER_BACKEND CPUS_PER_WORKER
readonly ATTEMPT_ID="$(date +%Y%m%d_%H%M%S)_${BASHPID}"

echo "Worker backend: ${WORKER_BACKEND}; visible GPUs: ${VISIBLE_GPU_COUNT}; concurrency: ${MAX_PARALLEL}"
if [[ "${WORKER_BACKEND}" == "visible" ]]; then
    echo "GPU tokens: ${GPU_TOKENS[*]}"
fi
echo "Intermediate PLYs are enabled; optimizer checkpoints: $([[ "${FRGS_SAVE_CHECKPOINTS:-0}" == "1" ]] && echo enabled || echo disabled)"

declare -A NEED_CHUNK=()
for index in "${PENDING_INDICES[@]}"; do
    NEED_CHUNK["${index}"]=1
done

# Start dense/long chunks first so the small final chunk is most likely to fit
# into the tail of the eight-GPU schedule.
PRIORITY_ORDER=(4 7 5 8 3 6 1 2 0)
QUEUE=()
for index in "${PRIORITY_ORDER[@]}"; do
    if [[ -n "${NEED_CHUNK[${index}]:-}" ]]; then
        QUEUE+=("${index}")
    fi
done

declare -a ACTIVE_PIDS=()
declare -a FREE_SLOTS=()
declare -a FAILED_CHUNKS=()
declare -A PID_CHUNK=()
declare -A PID_SLOT=()
declare -A PID_ATTEMPT=()
declare -A PID_LOG=()
for ((index = 0; index < MAX_PARALLEL; index++)); do
    FREE_SLOTS+=("${index}")
done

start_worker() {
    local chunk_index="$1"
    local slot="$2"
    local chunk_id
    printf -v chunk_id 'chunk_%04d' "${chunk_index}"
    local run_name="${RUN_STEM}_${chunk_id}_${ATTEMPT_ID}"
    local attempt_path="${ATTEMPT_DIR}/${chunk_id}.${ATTEMPT_ID}.ply"
    local log_path="${WORKER_LOG_DIR}/${chunk_id}.${ATTEMPT_ID}.log"
    local -a bbox_values=()
    read -r -a bbox_values <<< "${WORKER_BBOXES[${chunk_index}]}"
    if ((${#bbox_values[@]} != 6)); then
        echo "Invalid bbox for ${chunk_id}: ${WORKER_BBOXES[${chunk_index}]}" >&2
        return 1
    fi
    if [[ -e "${attempt_path}" || -L "${attempt_path}" ]]; then
        echo "Attempt output already exists: ${attempt_path}" >&2
        return 1
    fi

    local -a command=(
        "${PYTHON_BIN}" -c 'from fvdb_reality_capture.cli.frgs import frgs; raise SystemExit(frgs())'
        reconstruct "${DATASET_PATH}"
        -o "${attempt_path}"
        --run-name "${run_name}"
        --io.log-path "${LOG_PATH}"
        --tx.crop-bbox "${bbox_values[@]}"
        "${TRAINING_ARGS[@]}"
    )

    local -a worker_command=(
        "${SCRIPT_PATH}" __frgs_worker
        "${PLAN_PATH}" "${DATASET_PATH}" "${chunk_index}" "${chunk_id}" "${WORK_DIR}" "${PYTHON_BIN}"
        "${command[@]}"
    )

    if [[ "${WORKER_BACKEND}" == "srun" ]]; then
        srun \
            --exclusive \
            --exact \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task="${CPUS_PER_WORKER}" \
            --gpus-per-task=1 \
            --cpu-bind=cores \
            --gpu-bind=closest \
            env \
                PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
                OMP_NUM_THREADS=1 \
                MKL_NUM_THREADS=1 \
                OPENBLAS_NUM_THREADS=1 \
                NUMEXPR_NUM_THREADS=1 \
                "${worker_command[@]}" \
            >"${log_path}" 2>&1 {LAUNCH_LOCK_FD}>&- &
    else
        local gpu_token="${GPU_TOKENS[${slot}]}"
        setsid env \
            CUDA_VISIBLE_DEVICES="${gpu_token}" \
            PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 \
            "${worker_command[@]}" \
            >"${log_path}" 2>&1 {LAUNCH_LOCK_FD}>&- &
    fi

    local pid=$!
    ACTIVE_PIDS+=("${pid}")
    PID_CHUNK["${pid}"]="${chunk_index}"
    PID_SLOT["${pid}"]="${slot}"
    PID_ATTEMPT["${pid}"]="${attempt_path}"
    PID_LOG["${pid}"]="${log_path}"
    echo "Started ${chunk_id} as PID ${pid} (slot ${slot}); log: ${log_path}"
    echo "  Intermediate PLYs: ${LOG_PATH}/${run_name}/ply/"
}

remove_active_pid() {
    local completed_pid="$1"
    local -a remaining=()
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
        [[ "${pid}" != "${completed_pid}" ]] && remaining+=("${pid}")
    done
    ACTIVE_PIDS=("${remaining[@]}")
}

reap_one_worker() {
    local finished_pid=""
    local worker_status=0
    if wait -n -p finished_pid "${ACTIVE_PIDS[@]}"; then
        worker_status=0
    else
        worker_status=$?
    fi
    if [[ -z "${finished_pid}" || -z "${PID_CHUNK[${finished_pid}]:-}" ]]; then
        echo "Could not identify a completed worker process." >&2
        return 1
    fi

    local chunk_index="${PID_CHUNK[${finished_pid}]}"
    local slot="${PID_SLOT[${finished_pid}]}"
    local attempt_path="${PID_ATTEMPT[${finished_pid}]}"
    local log_path="${PID_LOG[${finished_pid}]}"
    local chunk_id final_path receipt_path gaussian_count gaussian_sha256 gaussian_size attempt_validation
    printf -v chunk_id 'chunk_%04d' "${chunk_index}"
    final_path="${WORK_DIR}/${chunk_id}.ply"
    receipt_path="${WORK_DIR}/${chunk_id}.receipt.json"
    remove_active_pid "${finished_pid}"
    FREE_SLOTS+=("${slot}")

    if ((worker_status != 0)); then
        echo "${chunk_id} failed with exit status ${worker_status}; see ${log_path}" >&2
        FAILED_CHUNKS+=("${chunk_id}")
    elif ! grep -Fq "Single-scene COLMAP bbox fast path selected" "${log_path}"; then
        echo "${chunk_id} did not confirm the bounded single-bbox loader; refusing its output: ${log_path}" >&2
        FAILED_CHUNKS+=("${chunk_id}")
    elif ! attempt_validation="$(validate_attempt "${attempt_path}")"; then
        echo "${chunk_id} produced an invalid attempt PLY; preserved at ${attempt_path}; see ${log_path}" >&2
        FAILED_CHUNKS+=("${chunk_id}")
    elif [[ -e "${final_path}" || -L "${final_path}" || -e "${receipt_path}" || -L "${receipt_path}" ]]; then
        echo "Refusing to replace an unexpectedly existing final chunk artifact/receipt: ${final_path}" >&2
        FAILED_CHUNKS+=("${chunk_id}")
    else
        IFS=$'\t' read -r gaussian_count gaussian_sha256 gaussian_size <<< "${attempt_validation}"
        if [[ ! "${gaussian_count}" =~ ^[0-9]+$ || ! "${gaussian_sha256}" =~ ^[0-9a-f]{64}$ \
            || ! "${gaussian_size}" =~ ^[0-9]+$ ]]; then
            echo "Malformed validation record for ${chunk_id}: ${attempt_validation}" >&2
            FAILED_CHUNKS+=("${chunk_id}")
        else
            # A same-filesystem hard link is an atomic no-replace publication.
            # The receipt fsync below durably records both the link and provenance.
            ln -- "${attempt_path}" "${final_path}"
            unlink -- "${attempt_path}"
            write_chunk_receipt \
                "${chunk_index}" "${final_path}" "${gaussian_count}" "${gaussian_sha256}" "${gaussian_size}"
            echo "Completed ${chunk_id}: ${gaussian_count} Gaussians; SHA-256 ${gaussian_sha256}; ${final_path}"
        fi
    fi

    unset 'PID_CHUNK['"${finished_pid}"']'
    unset 'PID_SLOT['"${finished_pid}"']'
    unset 'PID_ATTEMPT['"${finished_pid}"']'
    unset 'PID_LOG['"${finished_pid}"']'
}

cleanup_active_workers() {
    local reason="$1"
    ((${#ACTIVE_PIDS[@]} > 0)) || return 0
    echo "${reason}; terminating ${#ACTIVE_PIDS[@]} active worker process(es)." >&2
    local pid attempt running
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ "${WORKER_BACKEND}" == "visible" ]]; then
            kill -TERM -- "-${pid}" 2>/dev/null || true
        else
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    for attempt in {1..10}; do
        running=0
        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                running=$((running + 1))
            fi
        done
        ((running == 0)) && break
        sleep 1
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            if [[ "${WORKER_BACKEND}" == "visible" ]]; then
                kill -KILL -- "-${pid}" 2>/dev/null || true
            else
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        fi
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    ACTIVE_PIDS=()
}

handle_launcher_signal() {
    local signal_name="$1"
    trap - HUP INT TERM
    cleanup_active_workers "Received ${signal_name}"
    if [[ "${signal_name}" == "INT" ]]; then
        exit 130
    fi
    exit 143
}

cleanup_on_launcher_exit() {
    local exit_status=$?
    trap - EXIT
    if ((${#ACTIVE_PIDS[@]} > 0)); then
        cleanup_active_workers "Launcher exiting unexpectedly with status ${exit_status}"
    fi
    exit "${exit_status}"
}

trap cleanup_on_launcher_exit EXIT
trap 'handle_launcher_signal HUP' HUP
trap 'handle_launcher_signal INT' INT
trap 'handle_launcher_signal TERM' TERM

queue_position=0
while ((queue_position < ${#QUEUE[@]} || ${#ACTIVE_PIDS[@]} > 0)); do
    while ((queue_position < ${#QUEUE[@]} && ${#FREE_SLOTS[@]} > 0)); do
        free_position=$((${#FREE_SLOTS[@]} - 1))
        slot="${FREE_SLOTS[${free_position}]}"
        unset 'FREE_SLOTS['"${free_position}"']'
        FREE_SLOTS=("${FREE_SLOTS[@]}")
        start_worker "${QUEUE[${queue_position}]}" "${slot}"
        queue_position=$((queue_position + 1))
    done
    if ((${#ACTIVE_PIDS[@]} > 0)); then
        reap_one_worker
    fi
done
trap - HUP INT TERM

if ((${#FAILED_CHUNKS[@]} > 0)); then
    echo "Chunk training failed for: ${FAILED_CHUNKS[*]}" >&2
    echo "Valid completed chunks were retained. Rerun with RUN_ID=${RUN_ID} to train only missing chunks." >&2
    exit 1
fi

PENDING_OUTPUT="$(pending_chunks)"
parse_pending_output "${PENDING_OUTPUT}"
if ((${#PENDING_INDICES[@]} > 0)); then
    echo "Post-training validation still reports missing chunk indices: ${PENDING_INDICES[*]}" >&2
    exit 1
fi

merge_chunks
echo "Multi-GPU reconstruction complete: ${OUTPUT_PATH}"
