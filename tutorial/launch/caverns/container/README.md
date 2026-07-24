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
| `run_in_container.sh`      | Host launcher: `--hk`/`--t2k` picks the partition (images + data paths), `--mr`/`--sr` picks the task; binds your repo + data and runs the matching in-container script | Direct bash        | Interactive container runs (single- or multi-ring)               |
| `train_sr_in_container.sh` | In-container half for a **single-ring** task (GAT classification by default)  | (not run directly) | Selected by `--sr` in `run_in_container.sh`                       |
| `train_mr_in_container.sh` | In-container half for **multi-ring** segmentation                             | (not run directly) | Selected by `--mr` in `run_in_container.sh`                       |

### Job submission (SLURM)

| Script                             | Purpose                            | Execution | When to Use                                               |
| ---------------------------------- | ---------------------------------- | --------- | --------------------------------------------------------- |
| `submit_multiring_smoke_cclyon.sh` | Multi-ring segmentation smoke test | `sbatch`  | Validate the image + this checkout + data path end-to-end |

---

## Choosing an image: single-ring vs multi-ring

You no longer edit the script to pick an image — `run_in_container.sh` selects it from two
flags: `--hk`/`--t2k` (partition = which cluster's images + reference data) and
`--mr`/`--sr` (task). There is no single image that covers everything: multi-ring needs
`spconv`, single-ring needs PyG. Full image list in
[docs/cclyon_available_containers.md](../../docs/cclyon_available_containers.md).

| Flags        | In-container script        | Image                                                      | Notes                                                             |
| ------------ | -------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| `--hk --mr`  | `train_mr_in_container.sh` | `/sps/hyperk/zhu/CAVERNS/env/ml_image.sif`                 | Ships **spconv**. Also runs single-ring.                         |
| `--hk --sr`  | `train_sr_in_container.sh` | `/sps/hyperk/zhu/CAVERNS/env/ml_image.sif`                 | Same image; runs single-ring too.                               |
| `--t2k --mr` | `train_mr_in_container.sh` | `/sps/t2k/melbaz/env/ml_image.sif`                         | Ships **spconv**. Older CUDA/torch. Also runs single-ring.       |
| `--t2k --sr` | `train_sr_in_container.sh` | `/sps/t2k/eleblevec/containers/pytorch_pyg_cu130_v1.1.sif` | PyTorch 2.11 + CUDA 13.0 + **PyG**. **H100 only** (see below).   |

Defaults are `--hk --mr` (so a bare `bash run_in_container.sh` runs multi-ring on the
hyperk image).

> ⚠️ **`--t2k --sr` defaults to the CUDA 13.0 image, which does NOT run on V100.** CC-Lyon's
> **V100** nodes do not support CUDA 13.0, so `pytorch_pyg_cu130_v1.1.sif` fails there. The
> script prints this warning whenever you pass `--t2k --sr`.
>
> If you are on a **V100**, edit the `t2k_sr)` case in `run_in_container.sh` and set:
>
> ```bash
> IMAGE=/sps/t2k/melbaz/env/ml_image.sif   # older CUDA — also handles single-ring
> ```
>
> On **H100** the default PyG image is preferred (newer torch + PyG). The `--hk` images and
> **all multi-ring** runs are unaffected (`ml_image.sif` is the only one shipping `spconv`).

---

## Interactive container training: `run_in_container.sh`

**Use this for**: interactive single- or multi-ring training inside an apptainer image
(e.g. on a node you already have).

Two-script pattern:

- `run_in_container.sh` runs on the **host**: parses `--hk`/`--t2k` and `--mr`/`--sr`,
  binds your repo + data (+ index list for single-ring) into the chosen image, and controls
  wandb via env vars (`LAUNCH_WANDB`, `WANDB_MODE`, key file).
- `train_sr_in_container.sh` / `train_mr_in_container.sh` run **inside** the image: they
  set matplotlib/hydra (and `SPCONV_ALGO` for multi-ring) writable dirs, then launch
  `main.py` with the right config (`gat_classification_container` for single-ring,
  `multiring_segmentation_train` for multi-ring).

Usage:

```bash
bash run_in_container.sh [--hk|--t2k] [--mr|--sr] [extra hydra overrides...]

  --hk | --hyperk   hyperk-partition images + data paths (default)
  --t2k             t2k-partition images + data paths
  --mr              multi-ring segmentation fit (default)
  --sr              single-ring (GAT) fit
```

Examples (flags can be in any order; anything else is forwarded verbatim to `main.py`):

```bash
bash run_in_container.sh                              # = --hk --mr
bash run_in_container.sh --t2k --sr                   # single-ring, t2k PyG image (H100)
bash run_in_container.sh --hk --mr gpu_list=[0,1]     # multi-ring on 2 GPUs
bash run_in_container.sh --t2k --mr tasks.train.epochs=10
```

The repo location is auto-detected from the script's own path (override with
`NEUNET_ROOT`). The single-ring datasets also read a train/val/test index list from a
second bind, `BIND_INDEX` (mounted to `/workspace/work/index_lists`), added automatically
only for `--sr`; multi-ring does not use it. The `*_container.yaml` dataset configs already
point at those in-container mount points (see
[docs/cclyon_user_specific_paths.md](../../docs/cclyon_user_specific_paths.md)).

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
