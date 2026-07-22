#!/bin/bash

# ============================================================================
# Tutorial: Restore and Evaluate Trained Model
# ============================================================================
# This script restores a trained model checkpoint and runs evaluation.
# Use this after training to evaluate your model on test data.
#
# Note: the GNN tutorial mains (gcn_classification, gat_*, wcte_*) already
# restore the best checkpoint and evaluate at the end of training. Use this
# script for STANDALONE evaluation configs:
#   - multiring_segmentation_test (shipped example — needs the spconv
#     container, see tutorial/launch/caverns/container/README.md)
#   - or your own restore config in your config/main/ workspace
#
# Usage:
#   1. Edit the settings below (config name, paths, etc.)
#   2. Make sure you're in the correct conda environment
#   3. Run: ./restore_execute.sh
# ============================================================================

# ============================================================================
# Restore Configuration - EDIT THESE
# ============================================================================

# Path to WatChMaL repository — auto-detected from this script's location
# (override by exporting NEUNET_ROOT).
NeuNet_folder_path="${NEUNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# Config tree to compose from: 'tutorial/config/caverns' (shipped) or 'config/caverns' (yours)
config_folder=tutorial/config/caverns

# Config file name (without .yaml extension)
# This should be a config that restores a trained run and evaluates it,
# e.g. your own copy of multiring_segmentation_test pointing at your run.
config_name=multiring_segmentation_test

# GPU configuration
# For single GPU: 'gpu_list=[0]'
# For CPU: 'gpu_list=[]'
gpu_list='gpu_list=[0]'
# gpu_list='gpu_list=[]'  # Uncomment for CPU mode

# Master port for distributed training (only needed for multi-GPU)
master_port='MASTER_PORT=12353'

# Hydra search path (derived — usually don't need to change)
hydra_searchpath=${NeuNet_folder_path}/${config_folder}

# ============================================================================
# Execution Code - DO NOT MODIFY BELOW
# ============================================================================

echo "=========================================="
echo "Restoring and evaluating model"
echo "Config: $config_name"
echo "=========================================="

cd $NeuNet_folder_path
export HYDRA_FULL_ERROR=1

# Activate conda environment (adjust path and name, or activate beforehand)
# source /path/to/miniconda3/bin/activate <your_env>

python \
    main.py \
    --config-path=${hydra_searchpath}/main \
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list \
    $master_port
