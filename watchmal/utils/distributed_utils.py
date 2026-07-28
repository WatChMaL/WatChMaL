"""
Python files for distributed methods to be called in engines

To add :
- get_reduced
- get_gathered
"""

import inspect
import logging
import os

from torch.distributed import init_process_group

log = logging.getLogger(__name__)


def restrict_logging_to_rank0(rank, level=logging.WARNING):
    """Silence INFO/DEBUG on non-zero DDP ranks so start-up logs are not printed
    once per process.

    Each spawned worker re-runs hydra's ``configure_log()``, so without this every
    module's INFO line (dataset init, parameter counts, data-loader setup, ...) is
    emitted ``world_size`` times. Here rank 0 stays the only INFO/DEBUG emitter,
    while records at ``level`` and above (WARNING/ERROR) still get through from
    every rank so real problems are never hidden.

    Call this AFTER ``configure_log()`` has (re)installed the root handlers. The
    filter is attached to the root *handlers*, not the root logger: a record that
    propagates up from a module logger bypasses ancestor loggers' levels and
    filters but still passes through their handlers.

    Args:
        rank: this worker's rank; a no-op on rank 0.
        level: minimum level still allowed through on non-zero ranks.
    """
    if rank == 0:
        return

    class _MinLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno >= level

    rank_filter = _MinLevelFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(rank_filter)


def ddp_setup(rank, world_size, master_port, device=None, backend="nccl"):
    """
    Initialise the process group for a DDP worker.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        master_port: Port used for the rendez-vous, as a string
        device: Optional torch.device passed to init_process_group as `device_id`, so
            NCCL binds the right GPU eagerly (which avoids some init hangs). This
            argument only exists in newer torch releases: older cluster containers
            raise `TypeError: init_process_group() got an unexpected keyword argument
            'device_id'`. It is therefore passed only when the installed torch accepts
            it - the fallback is exactly what the caverns core always did, i.e. rely on
            the `torch.cuda.set_device(device)` the worker calls beforehand.
        backend: Process-group backend. Defaults to "nccl", which is what every real
            (multi-GPU) run uses. It is a parameter so the distributed layer can also
            be exercised on CPU with "gloo" - that is the only way the collectives
            (reduce / gather) can be tested anywhere without a GPU, e.g. in CI.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    kwargs = {}
    if device is not None:
        if "device_id" in inspect.signature(init_process_group).parameters:
            kwargs["device_id"] = device
        elif rank == 0:
            log.warning(
                "This torch build's init_process_group() has no `device_id` argument; "
                "falling back to torch.cuda.set_device() only. Harmless - it is an "
                "eager-binding optimisation - but upgrade the container to get it."
            )

    init_process_group(
        backend=backend, init_method='env://', rank=rank, world_size=world_size, **kwargs
    )
