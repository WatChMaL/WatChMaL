"""
Worker entry point for the WatChMaL core.

`main.py` selects a core and hands off to its `run`. This module holds what used to be
`main.main_worker_function`, unchanged; the split exists so that a second core can be
added alongside without either one's startup logic growing conditionals.

A core's `run` must accept `(rank, config, hydra_config=None)` and be importable at
module level, since it is handed to `torch.multiprocessing.spawn`.
"""

# hydra imports
from omegaconf import open_dict
from hydra.utils import instantiate
from hydra.core.utils import configure_log

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# generic imports
import logging

log = logging.getLogger(__name__)


def run(rank, config, hydra_config=None):
    """
    Instantiate model on a particular GPU, and perform train/evaluation tasks as specified

    Args:
        rank            ... rank of process among all spawned processes (in multiprocessing mode)
        config          ... hydra config specified in the @hydra.main annotation
        hydra_config    ... HydraConfig object for logging in multiprocessing
    """
    ngpus = len(config.gpu_list)
    is_distributed = ngpus > 1
    if is_distributed:
        # Spawned process needs to configure the job logging configuration
        configure_log(hydra_config.job_logging, hydra_config.verbose)
    if ngpus == 0:
        device = torch.device("cpu")
    else:
        # Infer rank from gpu and ngpus, rank is position in gpu list
        device = torch.device(f"cuda:{config.gpu_list[rank]}")
        torch.cuda.set_device(device)
        if is_distributed:
            # Set up pytorch distributed processing
            torch.distributed.init_process_group('nccl', init_method='env://', world_size=ngpus, rank=rank, device_id=device)

    log.info(f"Running main worker function rank {rank} on device: {device}")

    # Instantiate model and engine
    model = instantiate(config.model).to(device)

    # Configure the device to be used for model training and inference
    if is_distributed:
        # Convert model batch norms to synchbatchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device])

    # Instantiate the engine
    engine = instantiate(config.engine, model=model, rank=rank, device=device, dump_path=config.dump_path)

    # Configure automatic mixed precision
    engine.configure_amp(config.get("amp", False))
    for task, task_config in config.tasks.items():
        with open_dict(task_config):
            # Configure data loaders
            if 'data_loaders' in task_config:
                engine.configure_data_loaders(config.data, task_config.pop("data_loaders"), is_distributed, config.seed)
            # Configure optimizers
            if 'optimizers' in task_config:
                engine.configure_optimizers(task_config.pop("optimizers"))
            # Configure scheduler
            if 'scheduler' in task_config:
                engine.configure_scheduler(task_config.pop("scheduler"))
            # Configure loss
            if 'loss' in task_config:
                engine.configure_loss(task_config.pop("loss"))

    # Perform tasks
    for task, task_config in config.tasks.items():
        if is_distributed:
            # Before each task, ensure GPUs are in sync to avoid e.g. loading a state before a GPU finished training
            torch.distributed.barrier()
        getattr(engine, task)(**task_config)

    if is_distributed:
        torch.distributed.destroy_process_group()
