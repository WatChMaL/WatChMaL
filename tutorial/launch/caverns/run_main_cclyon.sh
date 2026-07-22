#!/bin/bash

# ============================================================================
# Tutorial: Run Training Interactively (CC-Lyon cluster)
# ============================================================================
# This script runs training directly in your current terminal session.
# Use this for quick tests, debugging, or when you want to see output in real-time.
#
# Usage:
#   1. Edit the settings below (config name, paths, etc.)
#   2. Make sure you're in the correct conda environment
#   3. Run: ./run_main_cclyon.sh
#   4. Press Ctrl+C to stop if needed
# ============================================================================

# ============================================================================
# Training Configuration - EDIT THESE
# ============================================================================

# Path to WatChMaL repository — auto-detected from this script's location, so it
# works on any cluster / any checkout (also from your launch/ copy).
# Override by exporting NEUNET_ROOT if you keep the script outside the repo.
NeuNet_folder_path="${NEUNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Config tree to compose from: 'tutorial/config/caverns' (shipped examples)
# or 'config' (your own workspace, see the main README part 1 - 4.)
config_folder=tutorial/config/caverns

# Config file name (without .yaml extension)
# Examples:
#   - gcn_classification
#   - gat_classification
#   - gat_vertex_regression
#   - wcte_mpmt_gat_classification
# For multiring_segmentation_train see the container config part.
config_name=gat_classification

# GPU configuration
# For single GPU: 'gpu_list=[0]'
# For multiple GPUs: 'gpu_list=[0,1]'
# For CPU: 'gpu_list=[]'
gpu_list='gpu_list=[0]'

# Master port for distributed training (only needed for multi-GPU)
# master_port='MASTER_PORT=12357'

# Hydra search path (derived — usually don't need to change)
hydra_searchpath=${NeuNet_folder_path}/${config_folder}

# Pytorch configuration
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Execution Code - DO NOT MODIFY BELOW
# ============================================================================

echo "=========================================="
echo "Running training interactively"
echo "Config: $config_name"
echo "GPU: $gpu_list"
echo "=========================================="

cd $NeuNet_folder_path
export HYDRA_FULL_ERROR=1


# add -c job aat the end to launch a dry run
# (main config will be displayed & no training will be performed), 
python \
    main.py \
    --config-path=${hydra_searchpath}/main \
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list \
    $master_port \
    # launch_wandb=False \
