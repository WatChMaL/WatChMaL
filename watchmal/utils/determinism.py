"""
Fully deterministic mode.

Setting a seed is not enough to make two runs of WatChMaL produce the same numbers. The
seed fixes *sampling* (which events, in which order, with which augmentations); it does
nothing about kernels that are free to accumulate in a different order each time. So a
run can be perfectly seeded and still drift, which makes "did this change alter the
result?" unanswerable - and that question is the whole point of a regression check.

`configure_determinism()` closes the remaining gaps:

  * **Nondeterministic kernels.** `torch.use_deterministic_algorithms(True)` makes torch
    pick a deterministic implementation where one exists and *raise* where none does,
    rather than silently returning run-dependent numbers. `warn_only=True` downgrades
    that to a warning, for the case where an op has no deterministic version and you
    would rather have the run than the guarantee.
  * **cuDNN autotuning.** `benchmark=False` stops cuDNN from picking an algorithm by
    timing it (the choice, and therefore the arithmetic, would depend on machine load);
    `deterministic=True` restricts it to reproducible kernels.
  * **cuBLAS workspaces.** CUDA >= 10.2 needs `CUBLAS_WORKSPACE_CONFIG` set *before the
    CUDA context is created* or reduction order in GEMMs is not reproducible. Call this
    before touching a device.
  * **The other two global RNGs.** `torch.manual_seed` does not touch python's `random`
    or numpy's legacy global RNG, and the image augmentations draw from `random`
    (`dataset/data_utils.py::apply_random_transformations`).

Deliberately NOT done here: a DataLoader `worker_init_fn`. Modern torch already seeds
every worker's `random`, `numpy` and `torch` from the base seed inside `_worker_loop`,
so adding one is redundant and gives numpy a cruder seed than torch's own.

What this does NOT give you: reproducibility *across* machines, torch versions, GPU
counts or `num_workers` values. Determinism means "the same run twice on the same
setup", which is what a regression check needs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from watchmal.utils.logging_utils import setup_logging

log = setup_logging(__name__)

# Value recommended by the CUDA docs; ":16:8" also works but is slower.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_determinism(enabled: bool = False, seed: int | None = None,
                          warn_only: bool = False, rank: int = 0) -> bool:
    """Put this process into fully deterministic mode.

    Args:
        enabled: when False this is a no-op and returns False, so call sites do not
            have to branch.
        seed: seeds python's `random` and numpy's global RNG. torch is seeded by the
            entrypoint (one seed for the whole run, set before the engine is built), so
            it is only re-applied here when a seed is passed.
        warn_only: warn instead of raising when an op has no deterministic
            implementation.
        rank: used to keep the log lines on rank 0.

    Returns:
        True if deterministic mode was engaged.
    """
    if not enabled:
        return False

    # Must precede CUDA context creation, hence before any .to(device)/set_device.
    if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))

    if rank == 0:
        log.info(
            f"Deterministic mode ON (warn_only={warn_only}). Two runs of this config on "
            f"this machine will produce identical numbers; results are not comparable "
            f"across machines, torch versions, GPU counts or num_workers."
        )
        if os.environ.get("PYTHONHASHSEED") is None:
            log.warning(
                "PYTHONHASHSEED is not set. It can only be fixed before the interpreter "
                "starts (`PYTHONHASHSEED=0 python main.py ...`). It affects iteration "
                "order of sets and of dicts keyed by str, so leave it unset only if no "
                "such order feeds the computation."
            )
    return True


def determinism_report() -> dict:
    """Current state of every knob this module sets - for logging and for tests."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
