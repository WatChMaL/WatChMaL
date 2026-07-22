# Dataset configs: which class for which data, and why

Each yaml in this folder selects a **dataset class** (`dataset_parameters._target_`) and
sets its options. This page explains what each class is for, and the reasoning behind its
design — so you can pick (or extend) the right one for your own data.

## Contents

- [Choosing at a glance](#choosing-at-a-glance)
- [`PyGInMemory20inchDataset` — PMT-level graphs, fully in RAM](#pyginmemory20inchdataset--pmt-level-graphs-fully-in-ram)
- [`PyGInMemoryMPMTDataset` — paired PMT + mPMT graphs (WCTE)](#pyginmemorympmtdataset--paired-pmt--mpmt-graphs-wcte)
- [`HyperKSparseCNN3D` — HDF5 hits → sparse 3D voxels (multi-ring)](#hyperksparsecnn3d--hdf5-hits--sparse-3d-voxels-multi-ring)
- [Common config keys](#common-config-keys)

## Choosing at a glance

| Config | Dataset class | Input on disk | Feeds | Why this class |
|---|---|---|---|---|
| `20inch_pmt_knn5_classification.yaml`, `20inch_pmt_knn5_vertex_regression.yaml` | `PyGInMemory20inchDataset` | pre-built PyG graphs (`data.pt`) | GNNs (GCN, GAT) | small graph samples → load once, keep in RAM |
| `wcte_mpmt_classification.yaml` | `PyGInMemoryMPMTDataset` | pre-built graph **pairs** (`data.pt` + `data_mPMT.pt`) | hierarchical mPMT models (`HierarchicalGAT`) | model needs two granularities per event |
| `multiring_sparse3d.yaml` | `HyperKSparseCNN3D` | raw WCSim HDF5 digi-hits | sparse 3D CNN (`SparseUNet3D` + query head) | events are sparse in a huge 3D volume → voxelize on the fly for spconv |

## `PyGInMemory20inchDataset` — PMT-level graphs, fully in RAM

`watchmal.dataset.graph.pyg_in_memory_20inch_pmt.PyGInMemory20inchDataset`
(a `torch_geometric.data.InMemoryDataset`).

**What it does**: loads a fully-processed PyG graph file (`data.pt`) — one graph per event,
edges already computed at graph-generation time (e.g. KNN k=5 for the shipped examples) —
entirely into RAM, then serves slices of it. Transforms (`data/transforms/` group) are
applied on a *clone* at access time, so the cached tensors stay pristine.

**Why in-memory**:

- The tutorial graph samples are small; loading once beats re-reading files every epoch.
- On multi-GPU (DDP) runs, `main.py` builds the dataset **once** and shares it across
  worker processes (`kind: "pyg_in_memory"` triggers this) — otherwise every process would
  pay the full load time.

**Multiple folders**: give `pyg_data_folder_path` a *list* of folders and one dataset is
built per folder, wrapped in `PyGConcatDataset` — handy to mix particle types
(see `20inch_pmt_knn5_classification.yaml`).

## `PyGInMemoryMPMTDataset` — paired PMT + mPMT graphs (WCTE)

`watchmal.dataset.graph.pyg_in_memory_mpmt.PyGInMemoryMPMTDataset`.

**What it does**: serves **two graphs per event** — a PMT-level graph (`data.pt`) and an
mPMT-level graph (`data_mPMT.pt`) — returned together as
`{'pmt_data': ..., 'mpmt_data': ...}`.

**Why it exists**: detectors instrumented with multi-PMT modules (WCTE) are naturally
hierarchical; models like `HierarchicalGAT` attend *within* each mPMT and then *across*
mPMTs, so they need both granularities for the same event.

**Why two internal datasets**: PyG's `InMemoryDataset.load()` can only hold one processed
file per instance, so this class wraps **two** `PyGInMemory20inchDataset` instances (one
per file) and asserts they have the same length. Transforms are currently applied to the
PMT-level graphs only (mPMT transforms not yet supported — see
`watchmal/dataset/graph/data_utils.py`).

## `HyperKSparseCNN3D` — HDF5 hits → sparse 3D voxels (multi-ring)

`watchmal.dataset.multiring.sparse_cnn.HyperKSparseCNN3D` (referenced via
`dataset.target_3d`, instantiated by the multi-ring engine rather than by Hydra).

**What it does**: reads raw WCSim multi-hit HDF5 files (searched recursively under
`base_dir` for `file_name_pattern`), de-duplicates hits per tube, voxelizes the hit points
into a sparse 3D grid (`spconv PointToVoxel`, grid controlled by `grid.axis_limit` /
`grid.grid_size`), normalizes features (`feat_norm`, stats cached to disk), and yields
per-voxel features plus `voxel_parent_frac` soft targets for ring segmentation.

**Why sparse voxels instead of graphs**:

- A Hyper-K event lights up a tiny fraction of a huge 3D volume; a dense 3D grid would be
  almost all zeros. Sparse convolutions (spconv) only compute on occupied voxels.
- Ring **segmentation** is a per-voxel task — the natural output structure is the voxel
  grid itself, not a graph.

**Why not in-memory / not Hydra-instantiated**: files are big and per-event processing is
cheap, so each engine builds its own instance lazily (`cache_in_ram` can cache processed
events). Needs `spconv` → run inside the container image
(see [docs/cclyon_available_containers.md](../../../docs/cclyon_available_containers.md)).

**Handy knobs**: `num_batches: 1` caps the run to one file (smoke tests),
`stats_cache_path` redirects the normalization-stats cache when the data dir is read-only.

## Common config keys

Keys shared by the graph dataset configs (read by `watchmal/dataset/graph/data_utils.py`
and the engines, not by the dataset class itself):

| Key | Meaning |
|---|---|
| `kind` | `"pyg_in_memory"` → `main.py` builds the dataset once and shares it across (DDP) processes |
| `split_path` | `.npz` with `train_idxs` / `val_idxs` / `test_idxs` (see `index_list/`) |
| `fully_processed` | `True` = edges already in the graphs; then `compute_edges_parameters: null` |
| `label_set` | PDG codes in the sample (mapped to class indices by the `MapLabels` transform) |
| `target_names` | one name per target dimension (used for plots) |
| `signal_key` | which class is "signal" in plots |
| `dataset_parameters` | everything below is passed to the dataset class (`_target_`, paths, file names) |

To add your own dataset class: implement it under `watchmal/dataset/`, point
`dataset_parameters._target_` at it, and keep the keys above — then mirror one of these
configs in your own `config/data/dataset/`.
