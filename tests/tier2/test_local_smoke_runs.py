"""
Tier 2 — end-to-end runs on the local smoke datasets: e-/mu- separation, and vertex
regression.

Goal: everything else in Tier 1 and Tier 2 tests a piece. These two run the real thing —
`main.py` → `run.py` → engine → tracker → `outputs/` — on real detector graphs, and
assert the properties that only a finished run can show:

  * **the evaluation artefacts are complete.** `indices.npy` must hold exactly the test
    split. This is where a real bug lived: with a concatenated dataset the saved index
    was the sub-dataset's local one, so the dedup in `evaluate()` erased events —
    silently, because the metrics are accumulated *before* the dedup and stayed right.
    Only these files show it, and only over a concatenation.
  * **the run is reproducible.** With `deterministic: True`, two runs of the same config
    must agree bit for bit. Without that guarantee no future check can compare numbers,
    which is why it is asserted here rather than assumed. (Measured: without the flag
    two CPU runs already differ, because PyG's scatter reductions parallelise.)

The datasets are 200-event subsets carved off the cluster files; they are far too large
to commit, so these tests SKIP when the data is absent — which is the case on a CI
runner. They are the pre-cluster check you run by hand. The configs come from the
tracked `tutorial/` tree (copied to a temp dir and repointed), never from the private
`config/` workspace, so what is tested is what ships.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.discovery import REPO_ROOT

TUTORIAL_TREE = REPO_ROOT / "tutorial" / "config" / "caverns"


def _patch_yaml(path: Path, edits: dict) -> None:
    """Apply `{"a.b.c": value}` edits to a YAML file in place."""
    doc = yaml.safe_load(path.read_text())
    for dotted, value in edits.items():
        node = doc
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _config_tree(tmp_path: Path, edits: dict[str, dict]) -> Path:
    """Copy the shipped caverns config tree and repoint it at the local data."""
    tree = tmp_path / "config"
    shutil.copytree(TUTORIAL_TREE, tree)
    for relative, file_edits in edits.items():
        _patch_yaml(tree / relative, file_edits)
    return tree


def _run(tree: Path, config_name: str, run_dir: Path, seed: int) -> None:
    """Invoke main.py exactly as a user would, and fail loudly with its output."""
    cmd = [
        sys.executable, "main.py",
        "--config-path", str(tree / "main"), "--config-name", config_name,
        f"hydra.searchpath=[{tree}]", f"hydra.run.dir={run_dir}",
        "gpu_list=[]", "launch_wandb=False",
        "deterministic=True", f"seed={seed}",
        "tasks.train.epochs=1", "tasks.train.val_interval=10",
        "tasks.train.data_loaders.train.num_workers=0",
        "tasks.train.data_loaders.train.batch_size=10",
        "tasks.train.data_loaders.validation.num_workers=0",
        "tasks.evaluate.data_loaders.test.num_workers=0",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                            env={**dict(__import__("os").environ), "HK_BANNER": "0"})
    if result.returncode != 0:
        pytest.fail(f"run failed ({config_name}):\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def _outputs(run_dir: Path) -> Path:
    out = run_dir / "outputs"
    assert out.is_dir(), f"no outputs/ directory produced in {run_dir}"
    return out


def _assert_run_is_analysis_readable(run_dir: Path) -> None:
    out = _outputs(run_dir)
    for name in ("log_train_0.csv", "log_val.csv"):
        assert (out / name).is_file(), f"{name} missing — analysis/ cannot read this run"
    assert list(out.glob("*_BEST.pth")), "no best checkpoint was saved"

    from analysis.read import WatChMaLOutput

    train_epoch, train_loss, val_epoch, val_loss, _ = WatChMaLOutput(
        str(run_dir)
    ).read_training_log()
    assert len(np.asarray(train_loss).ravel()) > 0
    assert np.all(np.isfinite(np.asarray(train_loss).ravel())), "NaN/inf in the train loss"
    assert len(np.asarray(val_loss).ravel()) > 0


def _assert_test_set_is_complete(run_dir: Path, split_path: Path) -> None:
    """The property the concat-index bug broke."""
    out = _outputs(run_dir)
    saved = np.load(out / "indices.npy")
    expected = np.load(split_path)["test_idxs"]

    assert len(saved) == len(expected), (
        f"evaluate saved {len(saved)} rows for a {len(expected)}-event test split — "
        "test data was lost (a non-unique event index makes the dedup in evaluate() "
        "erase events, without changing any logged metric)"
    )
    assert set(saved.tolist()) == set(expected.tolist()), (
        "the saved indices are not the requested test events"
    )
    assert len(np.unique(saved)) == len(saved), "duplicate indices in indices.npy"
    for name in ("preds", "targets"):
        array = np.load(out / f"{name}.npy")
        assert array.shape[0] == len(expected), f"{name}.npy has {array.shape[0]} rows"


def _assert_identical(a: Path, b: Path) -> None:
    for name in ("log_train_0.csv", "log_val.csv"):
        assert (a / name).read_text() == (b / name).read_text(), (
            f"{name} differs between two deterministic runs of the same config"
        )
    for name in ("preds", "targets", "indices"):
        assert np.array_equal(np.load(a / f"{name}.npy"), np.load(b / f"{name}.npy")), (
            f"{name}.npy differs between two deterministic runs"
        )


# --------------------------------------------------------------------------- #
# e- / mu- separation (PID classification)
# --------------------------------------------------------------------------- #

@pytest.mark.graph
def test_pid_run_is_complete_and_reproducible(tmp_path, pid_datasets, make_split):
    """Two concatenated per-class datasets — the shape that made the index bug visible —
    run twice to pin reproducibility."""
    electron, muon = pid_datasets
    # 400 events: 0..199 e- (y=11), 200..399 mu- (y=13); the permuted split mixes them.
    split = make_split(n_events=400, n_train=60, n_val=20, n_test=20)

    tree = _config_tree(tmp_path, {
        "data/dataset/20inch_pmt_knn5_classification.yaml": {
            "split_path": str(split),
            "dataset_parameters.graph_folder_path": [str(electron), str(muon)],
        },
        # These files are q,t,x,y,z: charge is column 0 (the shipped value of 1 suits the
        # t,q,... files), and feat_norm shipped with 2 entries for a 5-feature dataset.
        "data/transforms/20inch_pmt_classification.yaml": {
            "transforms.AddFeaturesInData.charge_index": 0,
            "transforms.Normalize.feat_norm": [
                [1000, 1900, 3242.96, 3242.96, 3296.47],
                [0.01, 550, -3242.96, -3242.96, -3296.47],
            ],
        },
        "model/vanilla_gat_classifier.yaml": {"in_channels": 5},
    })

    first, second = tmp_path / "run1", tmp_path / "run2"
    _run(tree, "gat_classification", first, seed=4242)
    _assert_run_is_analysis_readable(first)
    _assert_test_set_is_complete(first, split)

    targets = np.load(first / "outputs" / "targets.npy")
    assert set(np.unique(targets).tolist()) <= {0, 1}, "MapLabels must map 13/11 to 0/1"
    assert len(np.unique(targets)) == 2, (
        "the test split should contain both classes — a single-class test set makes "
        "every classification metric meaningless"
    )
    preds = np.load(first / "outputs" / "preds.npy")
    assert preds.shape[1] == 2, f"expected 2 class scores per event, got {preds.shape}"

    _run(tree, "gat_classification", second, seed=4242)
    _assert_identical(first / "outputs", second / "outputs")


# --------------------------------------------------------------------------- #
# vertex regression
# --------------------------------------------------------------------------- #

@pytest.mark.graph
def test_vertex_regression_run_is_complete(tmp_path, vertex_dataset, make_split):
    """A different route through the same core: 3 continuous targets instead of 2
    classes, and a single (non-concatenated) dataset."""
    split = make_split(n_events=200, n_train=60, n_val=20, n_test=20)

    tree = _config_tree(tmp_path, {
        "data/dataset/20inch_pmt_knn5_vertex_regression.yaml": {
            "split_path": str(split),
            "dataset_parameters.graph_folder_path": [str(vertex_dataset)],
        },
    })

    run_dir = tmp_path / "run"
    _run(tree, "gat_vertex_regression", run_dir, seed=7)
    _assert_run_is_analysis_readable(run_dir)
    _assert_test_set_is_complete(run_dir, split)

    targets = np.load(run_dir / "outputs" / "targets.npy")
    preds = np.load(run_dir / "outputs" / "preds.npy")
    assert targets.shape[1] == 3, f"vertex targets are (x, y, z), got {targets.shape}"
    assert preds.shape == targets.shape
    assert np.all(np.isfinite(preds)), "non-finite predictions"
