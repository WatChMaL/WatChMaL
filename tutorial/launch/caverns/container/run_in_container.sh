#!/bin/bash
# Container/image settings. Run from anywhere; binds YOUR repo + Mathias's data.
set -euo pipefail

# Root of NeuralNetworks_Software — auto-detected from this script's location
# (<repo>/tutorial/launch/caverns/container -> four levels up; override with NEUNET_ROOT).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEUNET_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# RUN_SCRIPT="${SCRIPT_DIR}/train_sr_in_container.sh"
RUN_SCRIPT="${SCRIPT_DIR}/train_mr_in_container.sh"
BIND_CODE="${REPO_ROOT}:/workspace/work/ml" 

# -- Data --
# - t2k partition
# BIND_DATA=/sps/t2k/melbaz/Simulation/output:/workspace/work/data
# BIND_DATA=/sps/hyperk/Datasets/graph_datasets:/workspace/work/data
# - hyperk partition
BIND_DATA=/sps/hyperk/Datasets/mr_smoke_datasets:/workspace/work/data/1_4rings_random_vertex_mix_muon_electron_2

# -- Split index --
# split_path (train/val/test index list) used by the sr tutorial datasets lives outside BIND_DATA, 
# so it needs its own mount
# - t2k partition
# BIND_INDEX=/sps/t2k/eleblevec/NeuNetSoft/index_lists:/workspace/work/index_lists
# - hyperk partition
BIND_INDEX=/sps/hyperk/Datasets/index_list:/workspace/work/index_list

# -- Container images --
# - t2k partition
# /sps/t2k/melbaz/env/ml_image.sif                         -> ref. image for multi-ring tasks. Also work for single ring tasks
# /sps/t2k/eleblevec/containers/pytorch_pyg_cu130_v1.1.sif -> Newer image (updated to cuda 13.0 & torch 2.11). Not suitable for multi-ring tasks. Not suitable for V100 (H100 GPUs only)
# - hyperk partition
# /sps/hyperk/zhu/CAVERNS/env/ml_image.sif                 -> image for multi-ring tasks. Also work for single ring tasks
IMAGE=/sps/hyperk/zhu/CAVERNS/env/ml_image.sif


# ---- wandb control (consumed by the in-container script) ----
export LAUNCH_WANDB=false          # true | false
export WANDB_MODE=online         # online | offline  (only used when LAUNCH_WANDB=true)

# Your wandb API key file (only read when LAUNCH_WANDB=true and WANDB_MODE=online)
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_key/wandb_api_key.txt}"
export WANDB_API_KEY=""
if [[ "${LAUNCH_WANDB}" == "true" && "${WANDB_MODE}" == "online" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

apptainer exec --nv \
  --bind "$BIND_CODE" \
  --bind "$BIND_DATA" \
  --bind "$BIND_INDEX" \
  --env LAUNCH_WANDB="${LAUNCH_WANDB}" \
  --env WANDB_MODE="${WANDB_MODE}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --pwd /workspace/work/ml \
  "$IMAGE" \
  bash -l "$RUN_SCRIPT" "$@"