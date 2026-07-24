#!/bin/bash
# Runs INSIDE the watchmal-core base image. cwd is /workspace/work/ml (set by the
# launcher's --pwd). Small ResNet SMOKE: train + test, <=5 epochs, resnet18.
#
# By default this runs a REAL training (needs Monte-Carlo data bound). Pass --dry
# to instead run Hydra's `-c job` DRY RUN (prints the composed config, no training),
# which needs no data and leaves the data configs as placeholders.

# Hydra path management: cwd is the bound repo root, so no host path is needed.
config_path="$(pwd)/tutorial/config/watchmal"

# ---- matplotlib / hydra writable dirs ----
export MPLCONFIGDIR=/tmp/mpl
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"
export HYDRA_FULL_ERROR=1


## -- STOP --
set -euo pipefail

## You shouldn't have to change anything below here.
# Real training by default (needs data bound). Pass  --dry  as the FIRST argument
# (or export SMOKE_DRY=1) to only compose + print the config and exit — no training,
# no data needed.
DRY_RUN=""
if [[ "${1:-}" == "--dry" ]]; then DRY_RUN="-c job"; shift; fi
if [[ "${SMOKE_DRY:-0}" == "1" ]]; then DRY_RUN="-c job"; fi

# resnet_train composes: data=iwcd, model=resnet18 (small), engine=classifier,
# train -> restore_best_state -> evaluate(test). Epochs capped at 5 for the smoke.
python main.py \
    --config-path "$config_path" \
    --config-name resnet_train \
    gpu_list=[0] \
    tasks.train.epochs=5 \
    ${DRY_RUN} \
    "$@"
