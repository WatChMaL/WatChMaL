"""
Worker entry point for the WatChMaL core (CNN / image / graph engines).

Engine contract served here:
  - engine ctor takes only the five common args
  - dump_path comes from the config, and the directory is created here
  - configure_amp is called before any task
  - configure_data_loaders takes (data_config, loaders_config, is_distributed, seed)
  - a barrier is raised between tasks so train -> restore_best_state -> evaluate is
    safe under DDP
  - MASTER_PORT is offset by the first GPU index, matching upstream

`dataset` and `wandb_run` are accepted so the spawn signature stays identical to the
caverns worker; this core builds its datasets inside the engine and has no wandb
integration.
"""

# hydra imports
from hydra.utils import instantiate
from hydra.core.utils import configure_log

from omegaconf import open_dict

# torch imports
import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# generic imports
import os
import logging

# watchmal imports
from watchmal.utils.distributed_utils import ddp_setup, restrict_logging_to_rank0

# Plain module logger, as upstream does it: this worker drives the watchmal core, so
# it stays off the caverns logging helper. Hydra's root handlers pick the records up
# either way.
log = logging.getLogger(__name__)


def run(rank, gpu_list, dataset, wandb_run, hydra_config, global_hydra_config):

    ngpus = len(gpu_list)
    is_distributed = ngpus > 1

    if ngpus == 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(gpu_list[rank])}")
        torch.cuda.set_device(device)

    if is_distributed:
        # Upstream offsets the port by the first GPU index to avoid collisions
        # between concurrent jobs sharing a node.
        master_port = int(hydra_config.MASTER_PORT) + int(gpu_list[0])
        ddp_setup(rank, world_size=ngpus, master_port=master_port, device=device)
        configure_log(global_hydra_config.job_logging, global_hydra_config.verbose)

    log.info(f"Running worker {rank} on device : {device}")

    # Each worker still announced itself above; from here on keep INFO/DEBUG on
    # rank 0 only so the shared start-up logs are not duplicated per process.
    restrict_logging_to_rank0(rank)

    torch.manual_seed(hydra_config.seed)

    dump_path = hydra_config.dump_path
    os.makedirs(dump_path, exist_ok=True)

    model = instantiate(hydra_config.model).to(device)
    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device])

    engine = instantiate(
        hydra_config.engine,
        model=model,
        rank=rank,
        device=device,
        dump_path=dump_path,
    )

    engine.configure_amp(hydra_config.get("amp", False))

    for task, task_config in hydra_config.tasks.items():

        with open_dict(task_config):

            if 'data_loaders' in task_config:
                engine.configure_data_loaders(
                    hydra_config.data,
                    task_config.pop("data_loaders"),
                    is_distributed,
                    hydra_config.seed,
                )

            if 'optimizers' in task_config:
                engine.configure_optimizers(task_config.pop("optimizers"))

            if 'scheduler' in task_config:
                engine.configure_scheduler(task_config.pop("scheduler"))

            if 'loss' in task_config:
                engine.configure_loss(task_config.pop("loss"))

    for task, task_config in hydra_config.tasks.items():
        if is_distributed:
            torch.distributed.barrier()
        getattr(engine, task)(**task_config)

    if is_distributed:
        log.info(f"Calling destroy_process_group()")
        destroy_process_group()
        log.info(f"Finished.")
