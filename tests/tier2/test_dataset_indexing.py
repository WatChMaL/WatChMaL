"""
Tier 2 — T2.9/T2.10: event identity, and split-file sanity.

Goal: the index an item carries **is** the event's identity. The engine gathers it
across ranks and deduplicates the evaluation outputs on it, so if two events can carry
the same index, rows disappear from `outputs/*.npy` — with no exception, no log line,
and no change to any metric, because the metrics are accumulated before the dedup. That
is not hypothetical: it happened, over a concatenated dataset, and the only visible
symptom was 83 saved rows for a 100-event test split.

Both tests here run **without any dataset**: the concatenation logic is exercised with
two trivial in-memory stubs, so this is a CI check rather than a local-only one. The
same property is re-asserted end-to-end, on real graphs, in `test_local_smoke_runs.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Dataset

from watchmal.dataset.graph.pyg_concat import PyGConcatDataset


class _Graphish:
    """Stand-in for a PyG `Data` object: just something with an `idx` attribute."""

    def __init__(self, idx, tag):
        self.idx = idx
        self.tag = tag


class _StubDataset(Dataset):
    """A dataset that numbers its own events from 0, exactly as every real one does."""

    def __init__(self, size, tag, as_dict=False):
        self.size = size
        self.tag = tag
        self.as_dict = as_dict

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = _Graphish(index, self.tag)
        if self.as_dict:
            # the shape produced once ConvertAndToDict has run
            return {"data": item, "target": 0, "indice": index}
        return item


def _index_of(item):
    return item["indice"] if isinstance(item, dict) else item.idx


@pytest.mark.parametrize("as_dict", [False, True], ids=["data-object", "dict-form"])
def test_concatenated_events_have_unique_global_indices(as_dict):
    """The regression test for the dedup bug.

    Sub-datasets each number from 0, so concatenating them must renumber: the index an
    item reports has to equal the position it was asked for, over the whole dataset.
    Checked in both shapes an item can take, because the transform chain decides which
    one reaches the engine.
    """
    dataset = PyGConcatDataset([
        _StubDataset(4, "e-", as_dict),
        _StubDataset(4, "mu-", as_dict),
        _StubDataset(4, "e-", as_dict),   # the same source listed twice, as configs do
    ])
    assert len(dataset) == 12

    reported = [_index_of(dataset[i]) for i in range(len(dataset))]
    assert reported == list(range(12)), (
        f"each item must report its own position; got {reported}. Colliding indices make "
        "evaluate() deduplicate real events away."
    )
    assert len(set(reported)) == len(dataset), "event indices must be unique"


def test_concatenation_still_returns_the_right_events():
    """Renumbering must not shuffle: index i has to be the event the sub-dataset holds
    at that offset, or the outputs are attached to the wrong events."""
    dataset = PyGConcatDataset([_StubDataset(3, "e-"), _StubDataset(3, "mu-")])
    tags = [dataset[i].tag for i in range(len(dataset))]
    assert tags == ["e-"] * 3 + ["mu-"] * 3, tags


def test_single_dataset_indices_are_untouched():
    dataset = PyGConcatDataset([_StubDataset(5, "solo")])
    assert [dataset[i].idx for i in range(5)] == list(range(5))


# --------------------------------------------------------------------------- #
# T2.10 — split-file sanity
# --------------------------------------------------------------------------- #

def assert_split_is_well_formed(split_path, n_events: int) -> None:
    """Reusable precondition: a split file that fails this makes every metric suspect.

    An index list is written by hand (`create_index.py`, edited per run) and nothing in
    the framework validates it: an out-of-range index raises deep inside a loader, a
    duplicated one double-counts an event, and a train/test overlap inflates every
    reported number with no visible symptom at all.
    """
    data = np.load(split_path)
    splits = {key: data[key] for key in ("train_idxs", "val_idxs", "test_idxs")}

    for name, idx in splits.items():
        assert len(idx) > 0, f"{name} is empty"
        assert idx.min() >= 0 and idx.max() < n_events, (
            f"{name} has indices outside [0, {n_events}): "
            f"[{idx.min()}, {idx.max()}]"
        )
        assert len(np.unique(idx)) == len(idx), f"{name} contains duplicate indices"

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = np.intersect1d(splits[a], splits[b])
            assert overlap.size == 0, (
                f"{a} and {b} share {overlap.size} events (e.g. {overlap[:5].tolist()}) — "
                "any metric computed from this split is optimistic"
            )


def test_generated_split_is_well_formed(make_split):
    assert_split_is_well_formed(make_split(n_events=400, n_train=60, n_val=20, n_test=20),
                                n_events=400)


def test_split_checker_catches_a_bad_split(tmp_path):
    """Guard the guard: a checker that never fires is worse than none."""
    overlapping = tmp_path / "bad.npz"
    np.savez(overlapping, train_idxs=np.arange(10), val_idxs=np.arange(5, 12),
             test_idxs=np.arange(20, 25))
    with pytest.raises(AssertionError, match="share"):
        assert_split_is_well_formed(overlapping, n_events=30)

    out_of_range = tmp_path / "oor.npz"
    np.savez(out_of_range, train_idxs=np.array([0, 1]), val_idxs=np.array([2]),
             test_idxs=np.array([999]))
    with pytest.raises(AssertionError, match="outside"):
        assert_split_is_well_formed(out_of_range, n_events=10)
