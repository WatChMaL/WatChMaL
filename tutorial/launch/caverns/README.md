# Tutorial Launch Scripts - Quick Start Guide

🐳 **Using containers ?** For apptainer-based launching — required for multi-ring  
segmentation (`spconv` only ships in an image) and for the smoke test — see the  
[container README](container/README.md). and the `container` folder.

> These are the shipped reference scripts — don't edit them in place. Copy them to your own  
> `launch/` folder with `bash setup/make_dirs.sh` and customize there (see the main README,  
> [Part 1, step 4: Create your own workspace](../README.md#4-create-your-own-workspace)).



## Contents

- [Overview](#overview)
  - [Interactive runs (no SLURM)](#interactive-runs-no-slurm)
  - [Job submission (SLURM)](#job-submission-slurm)
- [1. Interactive training](#1-interactive-training-run_main_cclyonsh)
- [2. Batch submission (SLURM)](#2-batch-submission-submit_main_cclyonsh)
- [3. Restore and evaluate](#3-restore-and-evaluate-restore_executesh)
- [Common configuration options](#common-configuration-options)
- [Troubleshooting](#troubleshooting)



## Overview

The scripts come in two flavours: **interactive** (direct bash, real-time output) and
**job submission** (`sbatch`, queued by SLURM).

### Interactive runs (no SLURM)


| Script               | Purpose              | Execution   | When to Use                                  |
| -------------------- | -------------------- | ----------- | -------------------------------------------- |
| `run_main_cclyon.sh` | Interactive training | Direct bash | Quick tests, debugging, real-time monitoring |
| `restore_execute.sh` | Model evaluation     | Direct bash | Evaluate trained models                      |




### Job submission (SLURM)


| Script                  | Purpose                | Execution | When to Use                         |
| ----------------------- | ---------------------- | --------- | ----------------------------------- |
| `submit_main_cclyon.sh` | Batch training (SLURM) | `sbatch`  | Long training runs, background jobs |




## 1. Interactive Training: `run_main_cclyon.sh`

**Use this for**: Quick tests, debugging, or when you want to see output in real-time.

> 🐳 **No conda/python environnement ?** You can run the same training inside a prebuilt apptainer image instead — no local environment needed. See [Interactive container training](container/README.md#interactive-container-training-run_in_containersh).



### Setup

1. Make the script executable:
  ```bash
   chmod +x run_main_cclyon.sh
  ```
2. Edit the configuration section in the script:
  ```bash
   # Set your config name
   config_name=gcn_classification

   # Set GPU (or CPU)
   gpu_list='gpu_list=[0]'  # GPU
   # gpu_list='gpu_list=[]'  # CPU
  ```
3. Activate your conda environment (if not already active):
  *You can check your conda env names by running* `conda env list` *in your terminal*
  ```bash
   conda activate <your-env-name>
  ```



### Run

```bash
./run_main_cclyon.sh
```



### What happens

- Runs training in your current terminal
- Output appears in real-time
- Press `Ctrl+C` to stop
- Good for debugging and short runs



### Tips

- Use CPU mode (`gpu_list=[]`) for quick tests
- Check GPU availability: `nvidia-smi` (in interactive mode only)
- Monitor GPU usage: `watch -n 1 nvidia-smi`

---



## 2. Batch Submission: `submit_main_cclyon.sh`

**Use this for**: Long training runs that you want to run in the background.

### Setup

1. Make the script executable:
  ```bash
   chmod +x submit_main_cclyon.sh
  ```
2. Edit the SLURM options (top of script):
  ```bash
   #SBATCH --job-name=tutorial_training
   #SBATCH --output=/path/to/logs/tutorial_training_%j.log
   #SBATCH --gres=gpu:v100:1
   #SBATCH --mem=50G
   #SBATCH --time=0-02:00:00  # 2 hours
  ```
3. Edit the training configuration:
  ```bash
   config_name=gcn_classification
   gpu_list='gpu_list=[0]'
  ```



### Submit

```bash
sbatch submit_main_cclyon.sh
```



### Monitor

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# View log file (replace %j with actual job ID)
tail -f /path/to/logs/tutorial_training_<job_id>.log

# Cancel job if needed
scancel <job_id>
```


## 3. Restore and Evaluate: `restore_execute.sh`

**Use this for**: Evaluating a trained model on test data.

### Prerequisites

- You must have a trained model checkpoint
- You need a restore config that points to your checkpoint



### Setup

1. Make the script executable:
  ```bash
   chmod +x restore_execute.sh
  ```
2. Edit the configuration:
  ```bash
   # Set the restore config name
   config_name=multiring_segmentation_test

   # Set GPU (or CPU)
   gpu_list='gpu_list=[0]'
  ```
3. Ensure your restore config points to the correct checkpoint:
  - The shipped example is `multiring_segmentation_test` (see its header; needs the spconv container)
  - The GNN tutorial mains already restore + evaluate at the end of training
  - For your own runs, create a restore config in your `config/main/` workspace



### Run

```bash
./restore_execute.sh
```



### What happens

- Loads the trained model checkpoint
- Runs evaluation on test data
- Generates metrics and plots
- Output appears in terminal



## Common Configuration Options



### Config Names

Shipped example configs live in `tutorial/config/caverns/main/`:

- **GNN (Hyper-K 20-inch PMT graphs)**: `gcn_classification`, `gat_classification`, `gat_vertex_regression`
- **GNN (WCTE mPMT graphs)**: `wcte_mpmt_gat_classification`
- **Multi-ring segmentation**: `multiring_segmentation_train` (train), `multiring_segmentation_test` (evaluate)



### GPU Settings

```bash
# Single GPU
gpu_list='gpu_list=[0]'

# Multiple GPUs
gpu_list='gpu_list=[0,1]'

# CPU only
gpu_list='gpu_list=[]'
```



### Paths

- **Repository path**: your `NeuralNetworks_Software` checkout (e.g. `/sps/t2k/<user>/NeuralNetworks_Software`)
- **Main config folder**: `tutorial/config/caverns/main` (passed to `--config-path`)
- **Hydra search path**: the `tutorial/config/caverns` root (passed as `hydra.searchpath=[...]`) so the
config groups referenced by the main config resolve

---



## Troubleshooting



### "Permission denied"

```bash
chmod +x <script_name>.sh
```



### "Config not found"

- Check that `spe_folder_name` matches your config folder
- Check that `config_name` matches your config file (without .yaml)



### "GPU not available"

- Check GPU availability: `nvidia-smi`
- Use CPU mode: `gpu_list='gpu_list=[]'`
- Check SLURM partition: `sinfo`



### "Conda environment not found"

- Activate environment: `conda activate <your-conda-env-name>`
- Or adjust path in script: `source /path/to/miniconda3/bin/activate <env_name>`  
*You can check available conda env with* `conda info --envs`



### "Out of memory"

- Reduce batch size in training config
- Request more memory in SLURM: `#SBATCH --mem=100G` (max on CC Lyon is 192GB for V100 nodes)

