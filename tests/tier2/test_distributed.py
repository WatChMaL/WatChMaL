"""
Tier 2 — T2.4: the DDP layer, exercised for real on gloo with world_size=2.

Goal: the merge unified four collective helpers that came from two different cores and
that return **two different types** — `get_reduced`/`get_gathered` hand back tensors,
`get_synchronized_metrics`/`get_synchronized_outputs` hand back python floats and numpy
arrays. Every engine family now reduces and gathers through those definitions. Getting
them crossed does not raise: it produces quietly wrong metrics, which is the failure the
draft's "2-GPU before/after check" was meant to catch, and which no amount of
single-process testing can see.

CI has no GPU, so this runs on **gloo** — that is what `ddp_setup(backend=...)` exists
for. Gloo proves the *algebra* (mean-on-rank-0, rank-ordered concatenation, who gets an
answer and who gets an empty dict); it cannot prove NCCL init, `device_id` binding or
the port offset. Those stay a cluster check.

The helpers are exercised on a stub carrying `BaseEngine`'s own unbound methods rather
than a constructed engine: no family's constructor, dataset or model is involved, so a
failure here is unambiguously the collective's fault. The set of helpers is discovered
from `BaseEngine`, so retiring one (the drafts call two of them back-compat adapters)
shrinks this test instead of breaking it.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.multiprocessing as mp

from watchmal.engine.base_engine import BaseEngine
from watchmal.utils.distributed_utils import ddp_setup

# Discovered, not listed: when an adapter is retired this simply stops being checked.
REDUCE_ADAPTERS = [n for n in ("get_reduced", "get_synchronized_metrics")
                   if hasattr(BaseEngine, n)]
GATHER_ADAPTERS = [n for n in ("get_gathered", "get_synchronized_outputs")
                   if hasattr(BaseEngine, n)]


class _Collectives:
    """Minimal carrier for BaseEngine's collective methods."""

    get_reduced = BaseEngine.get_reduced
    get_gathered = BaseEngine.get_gathered
    get_synchronized_metrics = BaseEngine.get_synchronized_metrics
    get_synchronized_outputs = BaseEngine.get_synchronized_outputs

    def __init__(self, rank, world_size):
        self.rank = rank
        self.n_gpus = world_size
        self.is_distributed = world_size > 1
        self.device = torch.device("cpu")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, case: str) -> None:
    """Runs in a spawned process. Assertions here surface in the parent as
    ProcessRaisedException with the traceback attached."""
    ddp_setup(rank, world_size=world_size, master_port=port, device=None, backend="gloo")
    try:
        engine = _Collectives(rank, world_size)

        if case == "reduce":
            # rank r contributes (r + 1); the mean over 2 ranks is 1.5
            metrics = {"loss": torch.tensor([float(rank + 1)])}
            out = engine.get_synchronized_metrics(dict(metrics))
            if rank == 0:
                assert set(out) == {"loss"}, out
                assert isinstance(out["loss"], float), type(out["loss"])
                assert abs(out["loss"] - 1.5) < 1e-6, out
            else:
                assert out == {}, f"only rank 0 may get a value, rank {rank} got {out}"

        elif case == "reduce_tensor_adapter":
            metrics = {"loss": torch.tensor([float(rank + 1)])}
            out = engine.get_reduced(dict(metrics))
            if rank == 0:
                assert torch.is_tensor(out["loss"]), type(out["loss"])
                assert abs(out["loss"].item() - 1.5) < 1e-6, out
            else:
                assert out == {}

        elif case == "gather":
            # rank-ordered concatenation: rank 0's rows first
            local = torch.tensor([[float(rank)], [float(rank)]])
            out = engine.get_synchronized_outputs({"preds": local})
            if rank == 0:
                got = out["preds"].ravel().tolist()
                assert got == [0.0, 0.0, 1.0, 1.0], f"expected rank-ordered concat, got {got}"

        elif case == "gather_tensor_adapter":
            local = torch.tensor([float(rank), float(rank)])
            out = engine.get_gathered(local)
            if rank == 0:
                assert out.tolist() == [0.0, 0.0, 1.0, 1.0], out.tolist()
            else:
                assert out.tolist() == [float(rank)] * 2, "non-zero ranks keep their local tensor"

        else:  # pragma: no cover
            raise AssertionError(f"unknown case {case}")
    finally:
        torch.distributed.destroy_process_group()


def _spawn(case: str, world_size: int = 2) -> None:
    mp.spawn(_worker, args=(world_size, _free_port(), case), nprocs=world_size, join=True)


@pytest.mark.skipif(not torch.distributed.is_available()
                    or not torch.distributed.is_gloo_available(),
                    reason="gloo backend unavailable")
@pytest.mark.parametrize("case", ["reduce", "gather"])
def test_collectives_two_ranks(case):
    """The two helpers the CNN loop uses: mean on rank 0, rank-ordered concatenation."""
    _spawn(case)


@pytest.mark.skipif(not torch.distributed.is_available()
                    or not torch.distributed.is_gloo_available(),
                    reason="gloo backend unavailable")
@pytest.mark.parametrize(
    "case",
    [c for c, present in (("reduce_tensor_adapter", "get_reduced" in REDUCE_ADAPTERS),
                          ("gather_tensor_adapter", "get_gathered" in GATHER_ADAPTERS))
     if present],
)
def test_tensor_adapters_agree_with_the_core(case):
    """The graph/multi-ring loops call the tensor-typed variants. Same algebra, different
    return type — and crossing the two is the mistake that stays silent."""
    _spawn(case)


def test_ddp_setup_accepts_a_cpu_backend():
    """Without this parameter the whole distributed layer is untestable off a GPU."""
    import inspect

    assert "backend" in inspect.signature(ddp_setup).parameters, (
        "ddp_setup lost its backend parameter; the DDP layer is now GPU-only to test"
    )
