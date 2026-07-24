#!/bin/bash
# Container launcher for the CAVERNS tutorial. Run from anywhere; binds YOUR repo +
# reference data into an apptainer image and runs the single-ring or multi-ring
# in-container training script.
#
# Usage:
#   bash run_in_container.sh [--hk|--t2k] [--mr|--sr] [--dry] [extra hydra overrides...]
#
#     --hk | --hyperk   hyperk-partition images + data paths (default)
#     --t2k             t2k-partition images + data paths
#     --mr              multi-ring segmentation fit (default)
#     --sr              single-ring (GAT) fit
#     --dry             Hydra dry run (`-c job`): print the composed config and exit,
#                       no training, no data needed
#
#   Any further arguments are forwarded verbatim to main.py, e.g.:
#     bash run_in_container.sh --t2k --mr gpu_list=[0,1] tasks.train.epochs=10
set -euo pipefail

# Root of the repo — auto-detected from this script's location
# (<repo>/tutorial/launch/caverns/container -> four levels up; override with NEUNET_ROOT).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEUNET_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

usage() {
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- parse flags; everything else is collected as hydra overrides ----
PARTITION="hk"          # hk | t2k
TASK="mr"               # mr | sr
DRY_RUN=0               # 1 -> append Hydra's `-c job` (print config, no training)
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hk|--hyperk) PARTITION="hk" ;;
    --t2k)         PARTITION="t2k" ;;
    --mr)          TASK="mr" ;;
    --sr)          TASK="sr" ;;
    --dry)         DRY_RUN=1 ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; HYDRA_ARGS+=("$@"); break ;;
    *)             HYDRA_ARGS+=("$1") ;;
  esac
  shift
done

# Dry run: forwarded to the in-container script and on to main.py as `-c job`.
if [[ "$DRY_RUN" == "1" ]]; then
  HYDRA_ARGS+=("-c" "job")
fi

echo "[run_in_container] partition=${PARTITION}  task=${TASK}  dry_run=${DRY_RUN}"

BIND_CODE="${REPO_ROOT}:/workspace/work/ml"

# ---- task -> in-container script ----
case "$TASK" in
  mr) RUN_SCRIPT="${SCRIPT_DIR}/train_mr_in_container.sh" ;;
  sr) RUN_SCRIPT="${SCRIPT_DIR}/train_sr_in_container.sh" ;;
  *)  echo "ERROR: unknown task '$TASK' (use --mr or --sr)" >&2; exit 1 ;;
esac

# ---- partition + task -> image, data bind, (sr only) index bind ----
# Mount targets are fixed by the shipped configs, do NOT change them here:
#   mr -> base_dir          /workspace/work/data/1_4rings_random_vertex_mix_muon_electron_2
#   sr -> graph_folder_path /workspace/work/data/debug/...
#         split_path        /workspace/work/index_lists/...   (note the trailing 's')
BIND_INDEX=""           # only the single-ring datasets read a split index list
case "${PARTITION}_${TASK}" in
  t2k_mr)
    # ml_image.sif: ref. image for multi-ring (also runs single-ring).
    IMAGE=/sps/t2k/melbaz/env/ml_image.sif
    BIND_DATA=/sps/t2k/melbaz/Simulation/output:/workspace/work/data
    ;;
  t2k_sr)
    # pyg_cu130 is H100-only (cuda 13.0 / torch 2.11); use ml_image.sif above if on V100.
    IMAGE=/sps/t2k/eleblevec/containers/pytorch_pyg_cu130_v1.1.sif
    BIND_DATA=/sps/hyperk/Datasets/graph_datasets:/workspace/work/data
    BIND_INDEX=/sps/t2k/eleblevec/NeuNetSoft/index_lists:/workspace/work/index_lists
    echo "WARNING: by default --t2k --sr runs on the cuda 13.0 image (${IMAGE})," >&2
    echo "         which is NOT compatible with V100 GPUs (H100 only)." >&2
    echo "         If you are using V100, set IMAGE=/sps/t2k/melbaz/env/ml_image.sif" >&2
    echo "         in the 't2k_sr)' case of this script (run_in_container.sh)." >&2
    ;;
  hk_mr)
    IMAGE=/sps/hyperk/zhu/CAVERNS/env/ml_image.sif
    BIND_DATA=/sps/hyperk/Datasets/mr_smoke_datasets:/workspace/work/data/1_4rings_random_vertex_mix_muon_electron_2
    ;;
  hk_sr)
    # hyperk ml_image.sif also runs single-ring tasks.
    IMAGE=/sps/hyperk/zhu/CAVERNS/env/ml_image.sif
    BIND_DATA=/sps/hyperk/Datasets/graph_datasets:/workspace/work/data
    BIND_INDEX=/sps/hyperk/Datasets/index_list:/workspace/work/index_lists
    ;;
  *) echo "ERROR: unknown partition/task combo '${PARTITION}_${TASK}'" >&2; exit 1 ;;
esac

# ---- wandb control (consumed by the in-container script) ----
export LAUNCH_WANDB=false          # true | false
export WANDB_MODE=online         # online | offline  (only used when LAUNCH_WANDB=true)

# Your wandb API key file (only read when LAUNCH_WANDB=true and WANDB_MODE=online)
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$HOME/.wandb_key/wandb_api_key.txt}"
export WANDB_API_KEY=""
if [[ "${LAUNCH_WANDB}" == "true" && "${WANDB_MODE}" == "online" ]]; then
  export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
fi

# ---- assemble binds (index only for single-ring) ----
BINDS=(--bind "$BIND_CODE" --bind "$BIND_DATA")
if [[ -n "$BIND_INDEX" ]]; then
  BINDS+=(--bind "$BIND_INDEX")
fi

apptainer exec --nv \
  "${BINDS[@]}" \
  --env LAUNCH_WANDB="${LAUNCH_WANDB}" \
  --env WANDB_MODE="${WANDB_MODE}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --pwd /workspace/work/ml \
  "$IMAGE" \
  bash -l "$RUN_SCRIPT" ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"}
