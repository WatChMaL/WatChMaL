# Container Launch Scripts (apptainer)

Container-based launching on CC-Lyon. Available images (paths, sizes, what they ship):
[docs/cclyon_available_containers.md](../../docs/cclyon_available_containers.md).

For conda-based (non-container) launching, see the [launch README](../../README.md).

## Contents

- [Overview](#overview)
  - [Interactive runs (no SLURM)](#interactive-runs-no-slurm)
  - [Job submission (SLURM)](#job-submission-slurm)
- [Choosing an image: single-ring vs multi-ring](#choosing-an-image-single-ring-vs-multi-ring)
- [Interactive container training: `run_in_container.sh`](#interactive-container-training-run_in_containersh)
- [Quick smoke test: `submit_multiring_smoke_cclyon.sh`](#quick-smoke-test-submit_multiring_smoke_cclyonsh)

## Overview

The container scripts come in two flavours: **interactive** (direct bash, on a node you
already hold) and **job submission** (`sbatch`, queued by SLURM).

### Interactive runs (no SLURM)

| Script                     | Purpose                                                                       | Execution          | When to Use                                                       |
| -------------------------- | ----------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------- |
| `run_in_container.sh`      | Host launcher: binds your repo + data, picks the image + task, runs one of the in-container scripts below | Direct bash        | Interactive container runs (single- or multi-ring)               |
| `train_sr_in_container.sh` | In-container half for a **single-ring** task (GAT classification by default)  | (not run directly) | Selected via `RUN_SCRIPT` in `run_in_container.sh`               |
| `train_mr_in_container.sh` | In-container half for **multi-ring** segmentation                             | (not run directly) | Selected via `RUN_SCRIPT` in `run_in_container.sh`               |

### Job submission (SLURM)

| Script                             | Purpose                            | Execution | When to Use                                               |
| ---------------------------------- | ---------------------------------- | --------- | --------------------------------------------------------- |
| `submit_multiring_smoke_cclyon.sh` | Multi-ring segmentation smoke test | `sbatch`  | Validate the image + this checkout + data path end-to-end |

---

## Choosing an image: single-ring vs multi-ring

There is no single image that covers everything — the multi-ring and single-ring stacks
need different CUDA / library builds. Pick the image (and matching task script) for your
model family; full list in
[docs/cclyon_available_containers.md](../../docs/cclyon_available_containers.md).

| Task                          | In-container script        | Image                                                      | Notes                                                                 |
| ----------------------------- | -------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| **Multi-ring** (sparse 3D CNN) | `train_mr_in_container.sh` | `/sps/t2k/melbaz/env/ml_image.sif`                         | Ships **spconv**. Older CUDA/torch. Also runs single-ring models.    |
| **Single-ring** (GAT/GCN, PyG) | `train_sr_in_container.sh` | either — see the GPU note below                            | PyG image is PyTorch 2.11 + CUDA 13.0 + **PyG**. No spconv.          |

You select the pair inside `run_in_container.sh` (the `RUN_SCRIPT`, `IMAGE`, and
`BIND_DATA` lines are commented in pairs — flip them together).

> ⚠️ **Single-ring: the image depends on which GPU you get.** CC-Lyon's **V100** nodes do
> not support CUDA 13.0, so `pytorch_pyg_cu130_v1.1.sif` will **not** run there.
>
> | Your GPU | Single-ring image to use                                                       |
> | -------- | ------------------------------------------------------------------------------ |
> | **V100** | `/sps/t2k/melbaz/env/ml_image.sif` only (older CUDA — also handles single-ring) |
> | **H100** | either image works; prefer `pytorch_pyg_cu130_v1.1.sif` (newer torch + PyG)    |
>
> This is why `run_in_container.sh` ships with `RUN_SCRIPT=train_sr_in_container.sh` (single-ring)
> but `IMAGE=/sps/t2k/melbaz/env/ml_image.sif` — that pairing is the one that runs
> everywhere. Switch the image to the PyG one only once you know you are on an H100.
>
> **Multi-ring is unaffected**: `ml_image.sif` is the only image shipping `spconv`, so it
> is always the multi-ring choice regardless of GPU.

---

## Interactive container training: `run_in_container.sh`

**Use this for**: interactive single- or multi-ring training inside an apptainer image
(e.g. on a node you already have).

Two-script pattern:

- `run_in_container.sh` runs on the **host**: it binds your repo + data (+ index list for
  single-ring) into the image, chooses the image and the in-container task script, and
  controls wandb via env vars (`LAUNCH_WANDB`, `WANDB_MODE`, key file).
- `train_sr_in_container.sh` / `train_mr_in_container.sh` run **inside** the image: they
  set matplotlib/hydra (and `SPCONV_ALGO` for multi-ring) writable dirs, then launch
  `main.py` with the right config (`gat_classification_container` for single-ring,
  `multiring_segmentation_train` for multi-ring).

To switch task, edit the paired settings near the top of `run_in_container.sh`:

```bash
# --- single-ring (default) ---
RUN_SCRIPT="${SCRIPT_DIR}/train_sr_in_container.sh"
BIND_DATA=/sps/hyperk/Datasets/graph_datasets:/workspace/work/data
IMAGE=/sps/t2k/melbaz/env/ml_image.sif                          # V100-safe default
# IMAGE=/sps/t2k/eleblevec/containers/pytorch_pyg_cu130_v1.1.sif  # H100 only

# --- multi-ring: comment the three lines above and use these instead ---
# RUN_SCRIPT="${SCRIPT_DIR}/train_mr_in_container.sh"
# BIND_DATA="/sps/t2k/melbaz/Simulation/output:/workspace/work/data"
# IMAGE=/sps/t2k/melbaz/env/ml_image.sif
```

The repo location is auto-detected from the script's own path (override with
`NEUNET_ROOT`). The single-ring datasets also read a train/val/test index list from a
second bind, `BIND_INDEX` (`/sps/t2k/eleblevec/NeuNetSoft/index_lists` ->
`/workspace/work/index_lists`); the `*_container.yaml` dataset configs already point at
those in-container mount points (see
[docs/cclyon_user_specific_paths.md](../../docs/cclyon_user_specific_paths.md)).

Run it, optionally appending extra Hydra overrides:

```bash
./run_in_container.sh                       # single-ring (default) or multi-ring, per your edits
./run_in_container.sh tasks.train.epochs=10 # extra Hydra overrides are forwarded to main.py
```

---

## Quick smoke test: `submit_multiring_smoke_cclyon.sh`

**Use this for**: a fast end-to-end check that the apptainer image, this checkout, and your data
path all work together — before launching a real run.

Runs multi-ring segmentation for **1 epoch on 1 file**, wandb offline, diagnostics disabled (so the
optional `diagnostic_multiring` submodule is **not** needed). It launches
`tutorial/config/caverns/main/multiring_segmentation_train.yaml` inside the apptainer image.

```bash
mkdir -p logs                         # SBATCH needs logs/ to exist
# Point the script at YOUR image + data (edit the defaults or export them):
export SMOKE_IMAGE=/path/to/your/ml_image.sif
export SMOKE_DATA=/path/to/dir/with/multiring/h5
sbatch tutorial/launch/caverns/container/submit_multiring_smoke_cclyon.sh
```

Success = job exits 0, having trained one epoch and written a checkpoint; check
`logs/slurm-multiring-seg-smoke-<jobid>.out`.
