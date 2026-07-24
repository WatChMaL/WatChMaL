# User-specific paths in the tutorial files

The shipped tutorial configs and launch scripts contain absolute CC-Lyon paths pointing
at **shared reference resources** (data, images — readable by `t2k`/`hyperk` group
members, verified 2026-07): leave those alone for a first run, change them when you
switch to your own data.

The **repo paths themselves are auto-detected** by the launch scripts — no editing needed:

- direct-bash scripts derive the repo root from their own location (works from your
`launch/` copy too; override by exporting `NEUNET_ROOT`);
- `sbatch` scripts use `SLURM_SUBMIT_DIR`, so **run** `sbatch` **from the repo root** (their
relative `logs/` output requires that anyway — `mkdir -p logs` first);
- inside the container, the repo is found at the bound working directory.

Owners: `eleblevec` = Erwan, `mferey` = Mathieu, `melbaz` = Mathias.

Remember: make any edits in **your own** `config/` **+** `launch/` **copies**
(see [Create your own workspace](../README.md#4-create-your-own-workspace)), not in the
tutorial trees.

## Contents

- [Must change before running](#must-change-before-running)
- [Reference resources — keep for a first run on CC-Lyon](#reference-resources--keep-for-a-first-run-on-cc-lyon)

## Must change before running

For **CC Lyon** :  

- only wandb-related settings remain user-specific, and only if you enable logging (`launch_wandb: True`):

Additionnal (for other cluster) : 

- *To be defined* 




| File                                            | Key / variable   | Currently points at                  | Purpose                                                                       |
| ----------------------------------------------- | ---------------- | ------------------------------------ | ----------------------------------------------------------------------------- |
| `tutorial/config/caverns/wandb/tutorial.yaml`          | `entity`         | Mathieu's wandb account              | where runs are logged → **your entity** (or `null` for your default login)    |
| `tutorial/config/caverns/wandb/wcte.yaml`              | `entity`         | Mathieu's wandb account              | idem                                                                          |
| `tutorial/launch/caverns/container/run_in_container.sh` | `WANDB_KEY_FILE` | `$HOME/.wandb_key/wandb_api_key.txt` | your wandb API key — only read with `LAUNCH_WANDB=true` + `WANDB_MODE=online` |




## Reference resources — keep for a first run on CC-Lyon

These are working defaults for the shipped examples: shared data and images, verified
group-readable. Change them only when you switch to your own data.


| File                                                                   | Key / variable                 | Points at                                                               | Purpose                                                                                                    |
| ---------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `tutorial/config/caverns/data/dataset/20inch_pmt_knn5_classification.yaml`    | `split_path`                   | `/sps/t2k/eleblevec/NeuNetSoft/index_lists/...npz`                      | train/val/test indices matched to the 15 debug graph folders below                                         |
| same + `..._vertex_regression.yaml`                                    | `graph_folder_path`            | `/sps/hyperk/Datasets/graph_datasets/debug/...knn5`                     | shared Hyper-K debug graph samples (10 events per folder)                                                  |
| `tutorial/config/caverns/data/dataset/20inch_pmt_knn5_vertex_regression.yaml` | `split_path`                   | same `.npz` as above                                                    | idem                                                                                                       |
| `tutorial/config/caverns/data/dataset/wcte_mpmt_classification.yaml`          | `split_path`                   | `/sps/t2k/mferey/CAVERNS/.../index_list/...npz`                         | indices for the WCTE reference sample                                                                      |
| `tutorial/config/caverns/data/dataset/wcte_mpmt_classification.yaml`          | `pyg_data_folder_path`         | `/sps/hyperk/mferey/Data/WCTE_v2/Graphs/...`                            | WCTE reference graph pairs (PMT + mPMT)                                                                    |
| `tutorial/launch/caverns/container/submit_multiring_smoke_cclyon.sh`           | `IMAGE` / `HOST_DATA` defaults | `/sps/t2k/melbaz/env/ml_image.sif`, `/sps/t2k/melbaz/Simulation/output` | smoke-test image + multi-ring data — overridable without editing via `SMOKE_IMAGE` / `SMOKE_DATA`          |
| `tutorial/launch/caverns/container/run_in_container.sh`                        | `IMAGE`, `BIND_DATA`           | same image + data as above                                              | container image (see [available containers](cclyon_available_containers.md)) and multi-ring reference data |


Note: `tutorial/config/caverns/data/dataset/multiring_sparse3d.yaml` contains only
`/path/to/...` placeholders — the launch scripts override them, or you set
`data.dataset.params.base_dir=...` on the command line.