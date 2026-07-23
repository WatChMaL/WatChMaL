# WatChMaL | CAVERNS

Framework for training, testing and using machine-learning models for Water Cherenkov
detectors: graph neural networks (classification / regression) and a sparse 3D CNN for
multi-ring segmentation. Configuration is composed with [Hydra](https://hydra.cc); runs
are driven by a single entry point, `main.py`.

## Contents

**[Part 1 — Quickstart](#part-1--quickstart)** — get the software running

1. [Install](#1-install)
2. [Smoke run : validate the software end-to-end](#2-smoke-run)
3. [Run a tutorial example](#3-run-a-tutorial-example)
4. ⭐ **[Create your own workspace](#4-create-your-own-workspace)** : all
  your experiments should **live in your own `config/` + `launch/` folders**, be careful not to overwrite the ones in the tutorial trees

**[Part 2 — Configuration guide](#part-2--configuration-guide)** — what you need to build your own experiments

- [How a run is assembled](#how-a-run-is-assembled)
- [Anatomy of a main config](#anatomy-of-a-main-config)
- [The config groups](#the-config-groups)
- [Hydra survival kit](#hydra-survival-kit)

More focused docs:

- `tutorial/launch/caverns/README.md` — the launch scripts, no container (interactive, SLURM, evaluate)
- `tutorial/launch/caverns/container/README.md` — container-based launching (smoke test, multi-ring)
- `watchmal/model/README.md` — the available models and how to add your own
- `tutorial/config/caverns/data/dataset/README.md` — the dataset classes: what each is for and why
- `docs/cclyon_available_containers.md` — ready-to-use apptainer images on CC-Lyon
- `docs/cclyon_available_detectors.md` — detectors, models and reference datasets on CC-Lyon
- `docs/cclyon_user_specific_paths.md` — every user-specific path in the tutorials: change it or keep it?

---

# Part 1 — Quickstart

## 1. Install


### Containers **(recommended)**

You can use the containers availables instead of create your own python env. They are shared across CC Lyon / Compute Canada / IPMU / Sukap clusters.  
Depending onw which cluster you are working you might need to contact :

- Compute Canada & IPMU : "Patrick de Perio" [pdeperio@IPMU.JP](mailto:pdeperio@IPMU.JP);
- CC Lyon : "Benjamin QUILAIN" [benjamin.quilain@llr.in2p3.fr](mailto:benjamin.quilain@llr.in2p3.fr);
- Sukap : [shinmasu@km.icrr.u-tokyo.ac.jp](mailto:shinmasu@km.icrr.u-tokyo.ac.jp);

### Local python env

```bash
git clone <this repo> && cd WatChMaL
pip install -r setup/requirements.txt  
```

Depending on what you run:

- **Graph models (GNN examples)** also need [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
- **Multi-ring segmentation** needs `spconv` (CUDA-specific). On CC-Lyon it ships inside an
apptainer image — use the launch scripts below rather than installing it yourself.
- **Multi-ring diagnostics plots** come from an optional submodule — *not* needed for
training or the smoke test:
  ```bash
  git submodule update --init submodules/diagnostic_multiring
  ```
  Without it, keep `tasks.train.diagnostic.enabled: []` (the default).

## 2. Smoke run

The fastest way to check that the code, the container image and your data path all work
together. It trains multi-ring segmentation for **1 epoch on 1 file**, wandb off,
diagnostics off — no wandb key, no submodule needed.

### CC Lyon

The script's defaults already point at the reference image and dataset, so there is
nothing to set — just submit it:

```bash
cd WatChMaL                           # if not already there
mkdir -p logs                                        # SLURM needs logs/ to exist

sbatch tutorial/launch/caverns/container/submit_multiring_smoke_cclyon.sh
```

Submit **from the repo root**: the script binds your checkout via `SLURM_SUBMIT_DIR`, and
its `logs/` output path is relative.

### Other clusters

Same command, but tell it where your image and data are:

```bash
cd WatChMaL
mkdir -p logs                                        # SLURM needs logs/ to exist

export SMOKE_IMAGE=/path/to/ml_image.sif             # apptainer image shipping spconv
export SMOKE_DATA=/path/to/dir/with/multiring/h5     # dir holding your multi-ring HDF5 file(s)

sbatch tutorial/launch/caverns/container/submit_multiring_smoke_cclyon.sh
```



**Success** = the job exits 0 after training one epoch and writing a checkpoint. Check
`logs/slurm-multiring-seg-smoke-<jobid>.out`.

## 3. Run a tutorial example

### Pick an example

Shipped examples live in `tutorial/config/caverns/main/`. Each one has a `--config-name`; the
graph examples also have a **`_container` variant** whose dataset paths are already
written as the in-container mount points (see [option 1](#option-1--container-recommended)).


| Example                            | `--config-name`                | `_container` variant                 | Set your data paths in                                                 |
| ---------------------------------- | ------------------------------ | ------------------------------------ | ---------------------------------------------------------------------- |
| GCN classification                 | `gcn_classification`           | —                                    | `tutorial/config/caverns/data/dataset/20inch_pmt_knn5_classification.yaml`    |
| Vanilla GAT classification         | `gat_classification`           | `gat_classification_container`       | same as above                                                          |
| Vanilla GAT vertex regression      | `gat_vertex_regression`        | `gat_vertex_regression_container`    | `tutorial/config/caverns/data/dataset/20inch_pmt_knn5_vertex_regression.yaml` |
| WCTE mPMT GAT classification       | `wcte_mpmt_gat_classification` | —                                    | `tutorial/config/caverns/data/dataset/wcte_mpmt_classification.yaml`          |
| Multi-ring segmentation (train)    | `multiring_segmentation_train` | container-only (needs `spconv`)      | override `data.dataset.params.base_dir=...` on the command line        |
| Multi-ring segmentation (evaluate) | `multiring_segmentation_test`  | container-only (needs `spconv`)      | see the header of that file                                            |


The shipped defaults point at shared reference data on CC-Lyon — fine for a first run
there. Everywhere else, see
[docs/cclyon_user_specific_paths.md](../docs/cclyon_user_specific_paths.md) for the full list
of paths to change.

### Pick a way to launch it

Three options. All three end up calling the same entry point, `main.py`, with the same
Hydra arguments — they differ only in what you need installed and in how much you type.


| # | Option                                                                | You need                                                  | Start from                                     |
| - | --------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------- |
| 1 | [**Container**](#option-1--container-recommended) *(recommended)*     | an apptainer image — **no local python env**              | `tutorial/launch/caverns/container/run_in_container.sh` |
| 2 | [**Local env + launch script**](#option-2--local-python-env--launch-script) | a python env with torch + CUDA (+ PyG for graph models)   | `tutorial/launch/caverns/run_main_cclyon.sh`           |
| 3 | [**Plain `python main.py`**](#option-3--plain-python-mainpy)          | the same env as option 2                                  | the command line, by hand                      |

---

#### Option 1 — Container *(recommended)*

Nothing to install: the image ships the whole stack. This is the **only** way to run
multi-ring segmentation, since `spconv` only exists inside an image.

It is a two-script pattern:

- `tutorial/launch/caverns/container/run_in_container.sh` runs on the **host** — it binds your
repo, your data and the index list into the image, picks the image, and chooses which
in-container script to run. Your checkout is auto-detected (override with `NEUNET_ROOT`).
- `train_sr_in_container.sh` (single-ring) / `train_mr_in_container.sh` (multi-ring) run
**inside** the image — they set the writable matplotlib/wandb dirs (`$HOME` is read-only
in the images), `SPCONV_ALGO=native` for multi-ring, then call `main.py`.

Edit the paired settings near the top of `run_in_container.sh` — `RUN_SCRIPT`, `BIND_DATA`
and `IMAGE` must be flipped **together**, because the single-ring and multi-ring stacks
need different images:

```bash
# --- single-ring (GAT/GCN, PyG) ---
RUN_SCRIPT="${SCRIPT_DIR}/train_sr_in_container.sh"
BIND_DATA=/sps/hyperk/Datasets/graph_datasets:/workspace/work/data
IMAGE=/sps/t2k/melbaz/env/ml_image.sif                            # runs on V100 and H100
# IMAGE=/sps/t2k/eleblevec/containers/pytorch_pyg_cu130_v1.1.sif  # newer, but H100 only

# --- multi-ring (sparse 3D CNN) ---
# RUN_SCRIPT="${SCRIPT_DIR}/train_mr_in_container.sh"
# BIND_DATA="/sps/t2k/melbaz/Simulation/output:/workspace/work/data"
# IMAGE=/sps/t2k/melbaz/env/ml_image.sif                          # ships spconv
```

⚠️ For single-ring, the image depends on your GPU: CC-Lyon's **V100** nodes do not support
CUDA 13.0, so the PyG image only runs on **H100**. `ml_image.sif` works on both — hence the
default. See [Choosing an image](launch/caverns/container/README.md#choosing-an-image-single-ring-vs-multi-ring).

Then run it — extra Hydra overrides are forwarded to `main.py`:

```bash
bash tutorial/launch/caverns/container/run_in_container.sh
bash tutorial/launch/caverns/container/run_in_container.sh tasks.train.epochs=10 gpu_list=[0]
```

The config name is set inside the in-container script (`gat_classification_container` for
single-ring, `multiring_segmentation_train` for multi-ring). Those `_container` configs
already point at `/workspace/work/data` and `/workspace/work/index_lists`, so **if you keep
the default binds there is no path to edit**. wandb is off by default (`LAUNCH_WANDB=false`
in the launcher).

Details, image table and the smoke test: [`tutorial/launch/caverns/container/README.md`](launch/caverns/container/README.md)
and [`docs/cclyon_available_containers.md`](../docs/cclyon_available_containers.md).

---

#### Option 2 — Local python env + launch script

Use this when you already have a working conda/python env (torch + CUDA, plus PyG for the
graph models). The script wraps the `main.py` command so you only edit a few variables at
the top; the repo root is auto-detected from the script's own location.

```bash
conda activate <your-env>            # check names with: conda env list
bash tutorial/launch/caverns/run_main_cclyon.sh
```

What to edit at the top of the script:

```bash
config_folder=tutorial/config/caverns  # or 'config/caverns' once you have your own workspace (step 4)
config_name=gat_classification       # any --config-name from the table above
gpu_list='gpu_list=[0]'              # [0]=one GPU, [0,1]=DDP, []=CPU
```

Use the **non-`_container`** config names here, and set your dataset paths in the file
listed in the table first.

For a long run, submit the batch version instead — `sbatch` needs `logs/` to exist and
resolves the repo from `SLURM_SUBMIT_DIR`, so **submit from the repo root**:

```bash
mkdir -p logs
sbatch tutorial/launch/caverns/submit_main_cclyon.sh
```

To evaluate an existing checkpoint, use `tutorial/launch/caverns/restore_execute.sh`. Full details:
[`tutorial/launch/caverns/README.md`](launch/caverns/README.md).

---

#### Option 3 — Plain `python main.py`

What the scripts above do for you, typed by hand. `--config-path` selects the folder of
main configs, `--config-name` picks one, and `hydra.searchpath` tells Hydra where the
config *groups* live (see [Part 2](#part-2--configuration-guide)):

```bash
python main.py \
    --config-path tutorial/config/caverns/main --config-name gcn_classification \
    hydra.searchpath=[tutorial/config/caverns]
```

Any config value can be overridden on the command line, e.g. `gpu_list=[0]
tasks.train.epochs=50`. On a cluster, prefer **absolute** paths in both `--config-path` and
`hydra.searchpath`.

---

Tip: `-c job` **prints the composed config and exits** — always do this before a real
launch. Option 1 forwards its arguments, so `bash run_in_container.sh -c job` works
directly; option 2 does not, so append `-c job` to the `python main.py` call at the bottom
of the script (there is already a comment marking the spot); option 3 takes it on the
command line.

## 4. Create your own workspace

`tutorial/config/` and `tutorial/launch/` are the shipped, tracked references — **don't
edit them**. Your experiments live in your own `config/` and `launch/` folders, which are
gitignored (personal, never pushed). Bootstrap them from the tutorials:

```bash
bash setup/make_dirs.sh     # copies tutorial/config/ -> config/  and  tutorial/launch/ -> launch/
```

Then:

1. Rename/edit a main config, e.g. `config/main/my_experiment.yaml`.
2. Adapt the group configs it references (`config/data/...`, `config/model/...`, ...).
3. Launch — same pattern, pointing at *your* folders:
  ```bash
   python main.py \
       --config-path config/main --config-name my_experiment \
       hydra.searchpath=[config]
  ```
4. On a cluster, edit the paths at the top of the scripts in `launch/` and submit those
  instead (see `tutorial/launch/caverns/README.md`; the full list of paths to adapt is in
   [docs/cclyon_user_specific_paths.md](../docs/cclyon_user_specific_paths.md)).

Part 2 explains what all the pieces mean.

---

# Part 2 — Configuration guide

## How a run is assembled

`main.py` performs the same steps for every job:

1. Hydra composes the full config (main config + `defaults` list + command-line overrides).
2. Global setup: `gpu_list` (empty = CPU, one entry = single GPU, several = DDP
  multiprocessing using `MASTER_PORT`), `seed`, optional wandb init (`launch_wandb`).
3. The **model** is instantiated from `config.model` (`_target`_ → Python class).
4. The **engine** is instantiated from `config.engine` — it owns the training/evaluation
  logic and receives the model.
5. For each entry under `**tasks`**, its `data_loaders`, `optimizers`, `scheduler`, `loss`
  and `early_stopping` sub-configs are handed to the engine.
6. Each remaining task key is called as an engine method: `tasks.train` → `engine.train(...)`,
  `tasks.evaluate` → `engine.evaluate(...)`, etc.

Each run writes into Hydra's run directory (`outputs/<date>/<time>/` by default);
checkpoints and logs land there.

## Anatomy of a main config

A main config sets a few top-level options, then composes everything else from **config
groups** via the `defaults` list. Annotated example
(`tutorial/config/caverns/main/gcn_classification.yaml`):

```yaml
core: caverns          # which engine core to run: caverns (default) or watchmal
MASTER_PORT: 12357     # port for distributed (multi-GPU) runs
seed: 1234             # REQUIRED - no seed is generated for you
gpu_list: []           # [] = CPU, [0] = one GPU, [0,1] = DDP on two GPUs
kind: gnn              # data/model family
launch_wandb: True     # wandb logging on/off (needs wandb/ group configured)

defaults:
    - wandb: tutorial                        # -> wandb/tutorial.yaml
    - model: gcn_classifier                  # -> model/gcn_classifier.yaml
    - engine: gnn_classifier                 # -> engine/gnn_classifier.yaml

    - data/dataset@data.dataset: 20inch_pmt_knn5_classification
    - data/transforms@data.transforms: 20inch_pmt_classification

    - tasks/train: gnn_train
    - sampler@tasks.train.data_loaders.train.sampler_config: subset_random
    - sampler@tasks.train.data_loaders.validation.sampler_config: subset_sequential
    - loss@tasks.train.loss: cross_entropy
    - optimizers@tasks.train.optimizers: adam
    - scheduler@tasks.train.scheduler: reduce_lr_on_plateau
    - scheduler@tasks.train.early_stopping: early_stopping

    - tasks/restore_best_state: restore_best # reload best checkpoint...
    - tasks/evaluate: gnn_evaluate           # ...then evaluate it
    - sampler@tasks.evaluate.data_loaders.test.sampler_config: subset_sequential

    - _self_                                 # this file wins over the groups
```

Reading a `defaults` line:

- `- model: gcn_classifier` — fill the `model` section from `model/gcn_classifier.yaml`
(found on the search path).
- `- loss@tasks.train.loss: cross_entropy` — the `@` **repackages**: take
`loss/cross_entropy.yaml` but place its content at `tasks.train.loss` instead of `loss`.
This is how one shared `loss/` folder serves any task.
- `- _self_` — apply this file's own keys last, so they override the groups.

## The config groups


| Group                                                          | Sets                                                | Shipped options                                             |
| -------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| `main/`                                                        | entry points — one file per experiment              | the 5 tutorials                                             |
| `engine/`                                                      | engine class (`_target_`) that runs the tasks       | `gnn_classifier`, `gnn_regressor`, `multiring/segmentation` |
| `data/dataset/`                                                | dataset class + paths + preprocessing               | in-memory graph tutorials, `multiring_sparse3d`             |
| `data/transforms/`                                             | data transformations / augmentation                 | classification, regression                                  |
| `model/`                                                       | network architecture (`_target_` + hyperparameters) | GCN, vanilla GAT, multi-ring segmentation                   |
| `tasks/train/`, `tasks/evaluate/`, `tasks/restore_best_state/` | what the job does: epochs, batch sizes, loaders...  | tutorials + `segmentation`                                  |
| `loss/`                                                        | loss function                                       | CE, BCE, MSE, Huber, smooth, `set_ce_dice`                  |
| `optimizers/`                                                  | optimizer                                           | Adam variants                                               |
| `scheduler/`                                                   | LR schedule / early stopping                        | cosine, plateau, stepLR, early stopping                     |
| `sampler/`                                                     | how loaders sample the dataset splits               | random, sequential                                          |
| `wandb/`                                                       | wandb project/entity/tags                           | tutorial, multiring                                         |


Anywhere a config has a `_target_` key, Hydra instantiates that Python class/function with
the sibling keys as arguments. To use a different optimizer or model, point `_target_` at
any class (yours, or straight from PyTorch) — no framework code changes needed.

## Hydra survival kit

Everything below composes in this order: config groups → main config (`_self_`) →
command line (strongest).

```bash
# Preview the composed config without running anything
python main.py -c job --config-path config/main --config-name my_experiment hydra.searchpath=[config]

# Override any value
python main.py ... tasks.train.epochs=50 gpu_list=[0,1]

# Add a key that doesn't exist yet (+) or delete one (~)
python main.py ... +data.dataset.params.stats_cache_path=/tmp/stats.npz
python main.py ... ~tasks.evaluate

# Swap a whole config group from the command line
python main.py ... optimizers@tasks.train.optimizers=adam loss@tasks.train.loss=mse

# Full stack traces when composition fails
HYDRA_FULL_ERROR=1 python main.py ...
```

Why `hydra.searchpath` is always needed: `--config-path` only tells Hydra where the *main*
config is. The groups it references (`model/...`, `loss/...`) are looked up on the search
path, so you pass the root of your config tree: `hydra.searchpath=[config]` (or
`[tutorial/config/caverns]` for the shipped examples). On clusters, prefer absolute paths in both.

Full Hydra docs: [https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/) (this repo uses `version_base=1.1`), and
`python main.py --hydra-help`.
