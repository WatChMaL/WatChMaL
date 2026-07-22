"""
Python files for distributed methods to be called in engines

To add :
- get_reduced
- get_gathered
"""

import os

from torch.distributed import init_process_group


def ddp_setup(rank, world_size, master_port, device=None):
    """
    Initialise the process group for a DDP worker.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        master_port: Port used for the rendez-vous, as a string
        device: Optional torch.device passed to init_process_group. The watchmal
            core sets it so NCCL binds the right GPU up front; the caverns core
            leaves it None and relies on torch.cuda.set_device beforehand.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    kwargs = {} if device is None else {"device_id": device}
    init_process_group(
        backend="nccl", init_method='env://', rank=rank, world_size=world_size, **kwargs
    )
