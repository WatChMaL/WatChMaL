#!/bin/bash

# ============================================================================
# Tutorial: Submit Training Job to SLURM (CC-Lyon cluster)
# ============================================================================
# This script submits a training job to the SLURM batch system.
# The job will run in the background and you can monitor it via SLURM commands.
#
# Usage:
#   1. Edit the settings below (config name, paths, etc.)
#   2. Submit the job: sbatch submit_main_cclyon.sh
#   3. Check status: squeue -u $USER
#   4. View logs: tail -f /path/to/logfile.log
# ============================================================================

# SLURM options - adjust these for your needs
## SBATCH --mail-user=your.email@example.com    # Where to send mail notifications
#SBATCH --mail-type=ALL                       # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Job configuration
#SBATCH --job-name=tutorial_training          # Job name (appears in squeue)
#SBATCH --output=logs/tutorial_training_%j.log  # Log file, relative to where you run sbatch
                                                # -> run from the repo root and `mkdir -p logs` first

# Resource requirements
#SBATCH --partition=gpu_v100                       # Partition: gpu, htc, etc.
#SBATCH --ntasks=1                           # Number of parallel processes
#SBATCH --cpus-per-task=5                    # CPUs per task
#SBATCH --gres=gpu:1                   # GPU: v100:1, a100:1, etc.
#SBATCH --mem=50G                            # Memory required
#SBATCH --time=0-01:00:00                    # Time limit (days-hours:minutes:seconds)

# ============================================================================
# Training Configuration - EDIT THESE
# ============================================================================
CONDA_ENV_NAME=pt28_cuda129
MINICONDA_PATH=/sps/t2k/eleblevec/miniconda3/

# Path to WatChMaL repository — auto-detected: SLURM_SUBMIT_DIR when submitted
# with sbatch (so RUN `sbatch` FROM THE REPO ROOT), script location otherwise.
# Override by exporting NEUNET_ROOT before sbatch if you submit from elsewhere.
NeuNet_folder_path="${NEUNET_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"

# Config tree to compose from: 'tutorial/config/caverns' (shipped examples)
# or 'config' (your own workspace, see the main README part 1 - 4.)
spe_folder_name=tutorial/config/caverns

# Config file name (without .yaml extension)
# Examples:
#   - gcn_classification
#   - gat_classification
#   - gat_vertex_regression
config_name=gat_vertex_regression

# GPU configuration
# For single GPU: 'gpu_list=[0]'
# For multiple GPUs: 'gpu_list=[0,1]'
# For CPU: 'gpu_list=[]'
gpu_list='gpu_list=[0]'

# Indicate master port in case of multi gpus training
# MASTER_PORT=12357

# Hydra search path (derived — usually don't need to change)
hydra_searchpath=${NeuNet_folder_path}/${spe_folder_name}

# ============================================================================
# Execution Code - DO NOT MODIFY BELOW
# ============================================================================

echo "=========================================="
echo "Submitting training job to SLURM"
echo "Config: $config_name"
echo "Job will be submitted to partition: gpu"
echo "=========================================="

cd $NeuNet_folder_path
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV_NAME}

python \
    main.py \
    --config-path=$hydra_searchpath/main \
    --config-name=$config_name \
    hydra.searchpath=[$hydra_searchpath] \
    $gpu_list \
    $master_port \
