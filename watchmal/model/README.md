# Models

Every model is a plain PyTorch `nn.Module`, selected from the config via `_target_`
(see the [configuration guide](../../README.md#part-2--configuration-guide)). The model
never trains itself — the **engine** (`watchmal/engine/`) drives it, so a model and an
engine are always picked as a pair in the main config.

## Graph models (PyTorch Geometric)

| File | Class | Task | Shipped config |
|---|---|---|---|
| `gcns.py` | `BaseGCN` | graph classification | `model/gcn_classifier.yaml` |
| `gat.py` | `GraphAttentionNetwork` | graph classification / regression | `model/vanilla_gat_classifier.yaml`, `model/vanilla_gat_vertex_regressor.yaml` |
| `no_conv_mlp.py` | `NoConvMLP` | MLP baseline (no message passing) | — |
| `node_encoder.py` | `NodeEncoder` | shared node-embedding building block | — |
| `mPMT_gat_pooling.py` | `HierarchicalGAT` | two-level PMT → mPMT GAT (WCTE) | `model/mpmt_gat_pooling.yaml` |
| `mPMT_gat_augmem.py` | `HierarchicalGAT` | memory-augmented variant of the above | — |
| `clsgat.py` | `GraphAttentionNetwork` | graph classification / multitask regression, prefit-token + CLS readout | — |
| `cherp.py` | `CheRP` | Set Transformer/Perceiver-style refinement of `CLSGAT.py`: token-bottleneck cross-attention, CLS readout, classification and single/multitask regression. Does not use the graph edges (still works with graph engine) | — |

Pair these with the graph engines: `engine/gnn_classifier.yaml`
(→ `watchmal.engine.graph.classification.ClassifierEngine`) or `engine/gnn_regressor.yaml`
(→ `watchmal.engine.graph.regression.RegressionEngine`). Models without a shipped config
are research models — read their docstrings and write a small `model/*.yaml` for them.

## Multi-ring segmentation (`multiring/`)

A sparse 3D CNN assembled from three parts by `watchmal.utils.build_utils.build_segmentation_model`
(see `model/multiring/segmentation_model.yaml` for the wiring):

- **Encoder** — `encoders/sparse_unet3D.SparseUNet3D`: sparse 3D U-Net (requires `spconv`).
- **Head** — `heads/query_transformer.QueryPerVoxelSoftmaxHead`: query-transformer decoder
  producing per-voxel ring assignments. `heads/per_voxel_linear.PerVoxelLinearHead` is a
  simpler alternative; `heads/activation_functions.py` provides softmax / sparsemax / entmax.
- **Wrapper** — `wrappers/model_wrapper.MultiRingModel`: glues encoder + head.

Pair with `engine/multiring/segmentation.yaml`
(→ `watchmal.engine.multiring.segmentation.MultiRingSegEngine`).

## Adding your own model

1. Create `watchmal/model/my_model.py` with an `nn.Module` whose `__init__` takes your
   hyperparameters.
2. Add a config in *your* workspace, `config/model/my_model.yaml`:
   ```yaml
   _target_: watchmal.model.my_model.MyModel
   in_channels: 5
   hidden_dim: 64
   ```
3. Reference it in your main config's `defaults` list: `- model: my_model`.
4. Keep the engine's contract in mind: your `forward` must accept the batches produced by
   the dataset/loader and return what the chosen engine expects — mirror a shipped model
   for the same task when in doubt.
