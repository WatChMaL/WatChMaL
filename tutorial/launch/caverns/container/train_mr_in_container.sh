#!/bin/bash

# Hydra path management: cwd is the bound repo root (/workspace/work/ml, set by
# the launcher's --pwd), so no host/cluster path is needed here.
hydra_searchpath="$(pwd)/tutorial/config/caverns"

# ---- matplotlib / spconv / hydra ----
export MPLCONFIGDIR=/tmp/mpl
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"

export SPCONV_ALGO=native
export HYDRA_FULL_ERROR=1


## -- STOP -- 
# Runs INSIDE ml_image.sif. cwd is /workspace/work/ml.
set -euo pipefail

## You can jump to python main.py ("Launch the training")
## You shouldn't have to change anything below here.
# ---- wandb, driven by the launcher's env vars ----
LAUNCH_WANDB="${LAUNCH_WANDB:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${LAUNCH_WANDB}" == "true" ]]; then
    # HOME is read-only in the image, so point wandb at a writable dir
    export WANDB_DIR=/workspace/work/ml/wandb
    export WANDB_CONFIG_DIR=$WANDB_DIR/config
    export WANDB_DATA_DIR=$WANDB_DIR/data
    export WANDB_CACHE_DIR=$WANDB_DIR/cache
    mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

    export WANDB_MODE                       # online | offline

    if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
        echo "ERROR: LAUNCH_WANDB=true, WANDB_MODE=online, but WANDB_API_KEY is empty." >&2
        echo "       Set it in run_in_container.sh, or switch to WANDB_MODE=offline." >&2
        exit 1
    fi    
    HYDRA_WANDB="launch_wandb=true"
else
    HYDRA_WANDB="launch_wandb=false"        # wandb fully off; WANDB_MODE ignored
fi

# Launch the training 
python main.py --config-path $hydra_searchpath/main --config-name multiring_segmentation_train \
    hydra.searchpath=[$hydra_searchpath] \
    ${HYDRA_WANDB} \
    gpu_list=[0] \
    +data.dataset.params.stats_cache_path=/workspace/work/ml/feature_stats_4rings.npz \
    "$@"
