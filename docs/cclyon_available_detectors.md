# CC-Lyon: available detectors, models and reference datasets

What can be trained out-of-the-box on CC-Lyon, per detector. Each row maps to a shipped
example config in `tutorial/config/caverns/main/` — launch pattern in the main
[README](../README.md#3-run-a-tutorial-example). Container choice:
[cclyon_available_containers.md](cclyon_available_containers.md).

## WCTE (Water Cherenkov Test Experiment)

Small Water Cherenkov detector instrumented with **multi-PMT (mPMT) modules**.  
Events are stored as *pairs* of pre-built graphs (PMT-level + mPMT-level).


| Task                  | Model                                                 | Example config                 | Reference data                         |
| --------------------- | ----------------------------------------------------- | ------------------------------ | -------------------------------------- |
| e-/mu- classification | `HierarchicalGAT` (intra-mPMT + inter-mPMT attention) | `wcte_mpmt_gat_classification` | `/sps/t2k/mferey/Data/WCTE_v2/Graphs/` |


- Dataset class: `watchmal.dataset.graph.pyg_in_memory_mpmt.PyGInMemoryMPMTDataset`
- Container: the PyG image.

# Hyper-K

## 20-inch PMTs only graphs (GNN)

PMT-level graphs built from WCSim events (KNN k=5 edges pre-computed).


| Task                      | Model                             | Example config          | Reference data (for smoke runs)               |
| ------------------------- | --------------------------------- | ----------------------- | --------------------------------------------- |
| mu-/e- classification     | `BaseGCN`                         | `gcn_classification`    | `/sps/t2k/eleblevec/Datasets/graph_datasets/` |
| mu-/e- classification     | `GraphAttentionNetwork` (vanilla) | `gat_classification`    | `/sps/t2k/eleblevec/Datasets/graph_datasets/` |
| Vertex (x,y,z) regression | `GraphAttentionNetwork` (vanilla) | `gat_vertex_regression` | `/sps/t2k/eleblevec/Datasets/graph_datasets/` |


- Dataset class: `watchmal.dataset.graph.pyg_in_memory_20inch_pmt.PyGInMemory20inchDataset`
- Container: the PyG image.

## Multi-ring segmentation (sparse 3D CNN + query transformer)

WCSim multi-hit HDF5 digi-hits, voxelized into a sparse 3D grid; per-voxel ring
assignment with a sparse 3D U-Net encoder and a query-transformer head.


| Task                         | Model                                       | Example config                 | Reference data (CC-Lyon)            |
| ---------------------------- | ------------------------------------------- | ------------------------------ | ----------------------------------- |
| Ring segmentation (train)    | `SparseUNet3D` + `QueryPerVoxelSoftmaxHead` | `multiring_segmentation_train` | `/sps/t2k/melbaz/Simulation/output` |
| Ring segmentation (evaluate) | idem, restored from a finished run          | `multiring_segmentation_test`  | `/sps/t2k/melbaz/Simulation/output` |


- Dataset class: `watchmal.dataset.multiring.sparse_cnn.HyperKSparseCNN3D`
- Container: `ml_image.sif` (ships spconv). Smoke test:
`tutorial/launch/caverns/container/submit_multiring_smoke_cclyon.sh`.
- Evaluation diagnostics need the optional `diagnostic_multiring` submodule.



# Super-K

*Work on-going.*

