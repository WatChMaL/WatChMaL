"""
Tier 2 — T2.5/T2.6/T2.7: sampling determinism, DDP model wrapping, sweep-config folding.

Three small seams the merge touched, each with the same property: when they break, the
run still completes and the numbers are merely wrong.

  * **Samplers.** The merge centralised the seed policy (one top-level seed, threaded
    into the sampler's generator). If two ranks disagree on ordering, DDP trains on
    overlapping or missing data — no error, just a quietly wrong model.
  * **`build_model`.** `find_unused_parameters` was flipped to strict (False) as a
    best-practice fix; a config asking for True must actually reach DDP, or the graph
    models that need it fail on a cluster and nowhere else.
  * **`merge_config`.** Runs on every wandb sweep, and a wrong nesting silently trains
    a different configuration than the one the sweep asked for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from watchmal.dataset.samplers.samplers import DistributedSamplerWrapper, SubsetRandomSampler
from watchmal.utils.build_utils import build_model, merge_config


# --------------------------------------------------------------------------- #
# T2.5 — sampling determinism and rank disjointness
# --------------------------------------------------------------------------- #

def _draw(indices, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return list(SubsetRandomSampler(indices, generator=generator))


def test_sampler_order_is_a_function_of_the_seed():
    indices = np.arange(50)
    assert _draw(indices, 1234) == _draw(indices, 1234), "same seed must replay exactly"
    assert _draw(indices, 1234) != _draw(indices, 4321), (
        "different seeds must give different orders, or the seed is not reaching the sampler"
    )
    assert sorted(_draw(indices, 7)) == list(indices), "every index must appear exactly once"


def _wrapped(indices, seed, rank, drop_last, world_size=2):
    """Build the wrapper the way a real rank does.

    Each rank is a separate process with its OWN sampler, constructed from the same
    run-wide seed. That detail is load-bearing: `DistributedSamplerWrapper` picks its
    slice by *position*, then resolves those positions against `list(self.sampler)` —
    so the ranks only end up with disjoint events because every rank's sampler replays
    the identical order. Sharing one sampler object between ranks in a single process
    would advance the generator twice and is not what happens in a run.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = SubsetRandomSampler(indices, generator=generator)
    return list(DistributedSamplerWrapper(sampler=sampler, seed=seed,
                                          num_replicas=world_size, rank=rank,
                                          drop_last=drop_last))


def test_distributed_wrapper_splits_without_overlap():
    """Each rank must get its own slice. Overlap means events trained on twice per epoch
    and a silently wrong gradient average — with no error anywhere."""
    indices = np.arange(40)
    per_rank = [_wrapped(indices, seed=99, rank=r, drop_last=True) for r in (0, 1)]

    assert len(per_rank[0]) == len(per_rank[1]), "ranks must take equal numbers of steps"
    assert not set(per_rank[0]) & set(per_rank[1]), "rank slices must be disjoint"
    assert set(per_rank[0]) | set(per_rank[1]) == set(indices.tolist()), (
        "an even-sized split should also be exhaustive"
    )


def test_validation_tail_policy_keeps_every_event():
    """drop_last=True is right for training (equal step counts, no intra-epoch
    duplication); for validation and inference it silently excludes a fixed tail and
    biases the metric every epoch. The wrapper must support both."""
    indices = np.arange(41)  # deliberately not divisible by the world size

    padded = set()
    for rank in (0, 1):
        padded |= set(_wrapped(indices, seed=3, rank=rank, drop_last=False))
    assert padded == set(indices.tolist()), "drop_last=False must cover every event"

    dropped = set()
    for rank in (0, 1):
        dropped |= set(_wrapped(indices, seed=3, rank=rank, drop_last=True))
    assert len(dropped) < len(indices), "drop_last=True is expected to drop the odd tail"


# --------------------------------------------------------------------------- #
# T2.6 — build_model
# --------------------------------------------------------------------------- #

MODEL_CFG = OmegaConf.create({"_target_": "torch.nn.Linear", "in_features": 4,
                              "out_features": 2})


def test_build_model_without_ddp_returns_the_bare_module():
    model, n_params = build_model(MODEL_CFG, device="cpu", use_ddp=False)
    assert isinstance(model, torch.nn.Linear)
    assert n_params == 4 * 2 + 2, "trainable-parameter count is reported to wandb"


def test_find_unused_parameters_reaches_ddp(monkeypatch):
    """Config-driven `find_unused_parameters` is the §B.4 target of the DDP unification.
    Asserted at the wrapper boundary, so it does not need two processes."""
    import watchmal.utils.build_utils as build_utils

    seen = {}

    class _FakeDDP:
        def __init__(self, module, **kwargs):
            seen.update(kwargs)
            self.module = module

    monkeypatch.setattr(build_utils, "DDP", _FakeDDP)
    build_utils.build_model(MODEL_CFG, device="cpu", use_ddp=True,
                            find_unused_parameters=True)
    assert seen.get("find_unused_parameters") is True, seen

    seen.clear()
    build_utils.build_model(MODEL_CFG, device="cpu", use_ddp=True)
    assert seen.get("find_unused_parameters") is False, (
        "the default must stay strict: True silently tolerates a parameter that gets no "
        "gradient, which is the bug you want surfaced"
    )


# --------------------------------------------------------------------------- #
# T2.7 — merge_config (wandb sweep folding)
# --------------------------------------------------------------------------- #

def test_merge_config_folds_dashed_keys_into_the_nested_config():
    cfg = OmegaConf.create({"data": {"dataset": {"batch_size": 8}}, "seed": 1})
    merged = merge_config(cfg, {"data-dataset-batch_size": 64, "seed": 7})
    assert merged.data.dataset.batch_size == 64, "a-b-c must address the nested key"
    assert merged.seed == 7


def test_merge_config_reports_unknown_keys_without_raising():
    """A sweep may carry keys this config does not have; that must not kill the run."""
    cfg = OmegaConf.create({"seed": 1})
    merged = merge_config(cfg, {"does-not-exist": 5})
    assert merged.seed == 1
    assert "does" not in merged
