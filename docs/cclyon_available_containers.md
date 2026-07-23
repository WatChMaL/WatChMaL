# CC-Lyon: available apptainer images

Ready-to-use `.sif` images on `/sps/t2k` for running this framework on the CC-Lyon
cluster (paths verified 2026-07). Which one you need depends on the model family —
see [cclyon_available_detectors.md](cclyon_available_detectors.md).


| Image                                                                                    | Size | Ships                                                                                       | Use for                                                                                                                         |
| ---------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `/sps/t2k/melbaz/env/ml_image.sif`                                                       | 11G  | PyTorch + **spconv**                                                                        | Multi-ring segmentation (sparse 3D CNN). Default image of the smoke test and of the multi-ring path of `tutorial/launch/caverns/container/run_in_container.sh` |
| `/sps/t2k/eleblevec/containers/pytorch_pyg_cu130_v1.1.sif`                               | 6.1G | PyTorch 2.11 + CUDA 13.0 + **PyG** (pyg_lib, scatter, sparse, cluster) + hydra/wandb/uproot + seaborn | Graph models (GCN, GAT, mPMT GAT). **H100 only** (no V100). Build recipe: `mini-Caverns-toolsbox/env_settings/containers/pytorch_pyg_cu130.def`; seaborn added on top via `containers/add_seaborn_v1.1.sh` |
| `/sps/t2k/eleblevec/cours/cmhk_ml_tutorial/container_base_ml_v4.0.0.sif`                 | 14G  | nic 4.0 base ML stack                                                                       | General-purpose fallback                                                                                                        |
| `/sps/t2k/eleblevec/cours/cmhk_ml_tutorial/container_base_ml_vpyg24.05.sif`              | 13G  | nic 4.0 modified with PyG 2.4                                                               | Graph models — ⚠️ **to be verified** before relying on it                                                                       |


## Usage pattern

All launch scripts follow the same shape (see the
[container launch README](../tutorial/launch/caverns/container/README.md)): bind your
checkout and your data into the image, run `main.py` from the bound repo.

```bash
apptainer exec --nv \
  --bind /path/to/WatChMaL:/workspace/work/ml \
  --bind /path/to/your/data:/workspace/work/data \
  --pwd /workspace/work/ml \
  <IMAGE.sif> \
  python main.py --config-path ... --config-name ... hydra.searchpath=[...]
```

Notes:

- `--nv` exposes the node's GPU; request one via SLURM (`#SBATCH --gres=gpu:v100:1`).
- `$HOME` is read-only inside some images — redirect writable state
(wandb dirs, matplotlib cache) as done in `tutorial/launch/caverns/container/train_mr_in_container.sh`
and `train_sr_in_container.sh`.
- For the multi-ring image, set `SPCONV_ALGO=native` (see the shipped scripts).

