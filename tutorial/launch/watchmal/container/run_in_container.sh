#!/bin/bash
# Container launcher for the WATCHMAL core — small ResNet smoke (train + test).
# Mirrors tutorial/launch/caverns/container/run_in_container.sh, but for the
# watchmal core and its official base image.
#
# By default it runs a REAL training (needs Monte-Carlo data bound below). Pass
# --dry to instead run Hydra's `-c job` DRY RUN: it composes and prints the resolved
# config, then exits WITHOUT training — so it needs NO data and the data paths in
# tutorial/config/watchmal/data/*.yaml stay as placeholders.
#
# Usage:
#   bash run_in_container.sh                        # real run (needs data bound below)
#   bash run_in_container.sh --dry                  # dry run (print config, no data)
#   # any extra args are forwarded verbatim to main.py
set -euo pipefail

# Root of the repo — auto-detected from this script's location
# (<repo>/tutorial/launch/watchmal/container -> four levels up; override with NEUNET_ROOT).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEUNET_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

RUN_SCRIPT="${SCRIPT_DIR}/train_resnet_smoke_in_container.sh"
BIND_CODE="${REPO_ROOT}:/workspace/work/ml"

# -- Container image --
# Official base image used for the watchmal core (ships in /sps/hyperk/containers/ml).
IMAGE="${WATCHMAL_IMAGE:-/sps/hyperk/containers/ml/container_base_ml_v4.0.0.sif}"

# -- Monte-Carlo data (only needed for a --real run) --
# You do NOT have/need it for the default dry run. When you do, bind it here and
# point tutorial/config/watchmal/data/*.yaml at the mount point.
# BIND_DATA=/path/to/your/data:/workspace/work/data

BINDS=(--bind "$BIND_CODE")
[[ -n "${BIND_DATA:-}" ]] && BINDS+=(--bind "$BIND_DATA")

apptainer exec --nv \
  "${BINDS[@]}" \
  --pwd /workspace/work/ml \
  "$IMAGE" \
  bash -l "$RUN_SCRIPT" "$@"
