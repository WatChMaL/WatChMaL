#!/bin/bash
#SBATCH -A hyperk
#SBATCH -p gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH -J multiring-seg-smoke
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err

# Multi-ring segmentation SMOKE TEST (1 file, 1 epoch, no wandb) inside the
# apptainer image that ships spconv. Run from the repo root:
#     mkdir -p logs            # SBATCH needs logs/ to exist
#     # set SMOKE_IMAGE / SMOKE_DATA below (or export them), then:
#     sbatch tutorial/launch/caverns/container/submit_multiring_smoke_cclyon.sh
#
# Launches the canonical example config tutorial/config/caverns/main/multiring_segmentation_train.yaml
# with smoke overrides (1 epoch, 1 file). Diagnostics are off by default, so the
# optional `diagnostic_multiring` submodule is NOT required.

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME:-} (${SLURM_JOB_ID:-}) on $(hostname)"
nvidia-smi || true

# Resolve this repo's root so the container binds YOUR checkout, not a hard path.
# Under sbatch, SLURM runs a spooled COPY of this script, so the script location
# is unreliable -> use SLURM_SUBMIT_DIR (run `sbatch` FROM THE REPO ROOT, which
# the relative logs/ SBATCH output above already requires). Direct-bash runs
# fall back to the script location (<repo>/tutorial/launch/caverns/container -> ../../../..).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
echo "Binding repo: ${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/main.py" ]]; then
    echo "ERROR: ${REPO_ROOT} does not look like the repo root (no main.py)." >&2
    echo "       Run sbatch from the NeuralNetworks_Software root." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# EDIT ME (or export SMOKE_IMAGE / SMOKE_DATA before sbatch) -----------------
#   SMOKE_IMAGE : apptainer .sif that ships spconv + this framework's deps
#   SMOKE_DATA  : host directory holding your multi-ring HDF5 file(s)
# The defaults below are the known-working reference dataset/image; point them
# at your own paths on the cluster.
# ---------------------------------------------------------------------------
IMAGE="${SMOKE_IMAGE:-/sps/t2k/melbaz/env/ml_image.sif}"
HOST_DATA="${SMOKE_DATA:-/sps/t2k/melbaz/Simulation/output}"
# HDF5 filename pattern searched (recursively) under the data dir.
FILE_PATTERN="${SMOKE_FILE_PATTERN:-wcsim_output_multihit_with_digi_hit_and_trigger.h5}"

WORKDIR="/workspace/work/ml"                       # repo mount point in container
DATA_MOUNT="/workspace/work/data"                  # data mount point in container
BIND_CODE="${REPO_ROOT}:${WORKDIR}"                # your code -> /workspace/work/ml
BIND_DATA="${HOST_DATA}:${DATA_MOUNT}"             # your data -> /workspace/work/data

echo "Image: ${IMAGE}"
echo "Data : ${HOST_DATA}  ->  ${DATA_MOUNT}"

# wandb is disabled for the smoke test (launch_wandb=false below). Offline-safe.
# For a real run: set WANDB_API_KEY to YOUR key and pass launch_wandb=true.
export WANDB_MODE=offline

# Build the in-container command (outer shell expands the paths below). Uses the
# same launch pattern as run_main_cclyon.sh: --config-path points at main/, and
# hydra.searchpath adds the tutorial/config/caverns root so the group defaults resolve.
# stats_cache_path is redirected into the (writable) repo checkout so feature
# normalization stats can be written even when the data dir is read-only.
PYCMD="SPCONV_ALGO=native HYDRA_FULL_ERROR=1 python main.py \
    --config-path ${WORKDIR}/tutorial/config/caverns/main --config-name multiring_segmentation_train \
    hydra.searchpath=[${WORKDIR}/tutorial/config/caverns] \
    launch_wandb=false \
    tasks.train.epochs=1 \
    data.dataset.params.num_batches=1 \
    data.dataset.params.base_dir=${DATA_MOUNT} \
    data.dataset.params.file_name_pattern=${FILE_PATTERN} \
    data.dataset.params.stats_cache_path=${WORKDIR}/logs/feature_stats_smoke.npz"

srun apptainer exec --nv \
  --bind "$BIND_CODE" \
  --bind "$BIND_DATA" \
  --pwd  "$WORKDIR" \
  "$IMAGE" \
  bash -lc "$PYCMD"
