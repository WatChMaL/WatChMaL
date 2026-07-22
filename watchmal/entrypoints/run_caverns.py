"""
Worker entry point for the CAVERNS core (graph / multi-ring engines).

Engine contract served here:
  - engine ctor takes wandb_run= and dataset= on top of the five common args
  - dump_path is the hydra run directory
  - configure_dataset(data_config) then configure_data_loaders(loaders_config)
  - configure_early_stopping is available on some engines
  - MASTER_PORT is used verbatim, no offset

NOTE: this worker's signature and its reliance on parent-side setup (wandb init and
the in-memory dataset prebuild) differ from run_watchmal's. main.py is not yet wired to
dispatch here; that needs a core-aware parent (see the merge notes). The utils it
imports (build_utils, distributed_utils, logging_utils_caverns) arrive in a later step,
so this module is not importable until then - which is fine, main.py imports it lazily.
"""

# hydra imports
from hydra.utils import instantiate
from hydra.core.utils import configure_log

from omegaconf import open_dict

# torch imports
import torch
from torch.distributed import destroy_process_group

# generic imports
import os
import time

# watchmal imports
from watchmal.utils.logging_utils_caverns import setup_logging
from watchmal.utils.build_utils import build_model
from watchmal.utils.distributed_utils import ddp_setup, restrict_logging_to_rank0

log = setup_logging(__name__)
sleep_time = 5


def run(rank, gpu_list, dataset, wandb_run, hydra_config, global_hydra_config):

    if rank == 0:
        for k in list(os.environ.keys()):
            if k.startswith('WANDB_'):
                log.info(f"wandb env var {k}: {os.getenv(k)}")

    # Initialize the group and configure the log in case of distributed training
    if len(gpu_list) > 1:
        torch.cuda.set_device(int(gpu_list[rank])) ## Mathias :  to avoid illegal memory access error when using DDP with multiple gpus. Some librairies using CUDA allocate memory on the current device, not necessarily on the device of your input tensors.
        ddp_setup(rank, world_size=len(gpu_list), master_port=str(hydra_config.MASTER_PORT)) # Keep len(gpu_list here). After can call get_world_size()
        configure_log(global_hydra_config.job_logging, global_hydra_config.verbose)

    device = 'cpu' if len(gpu_list) == 0 else (f"cuda:{int(gpu_list[rank])}" if len(gpu_list) > 1 else f"cuda:{int(gpu_list[0])}")
    wandb_run = wandb_run if rank == 0 else None
    log.info(f"Running worker {rank} on device : {device} with wandb_run : {wandb_run}")

    # Each worker still announced itself above; from here on keep INFO/DEBUG on
    # rank 0 only so the shared start-up logs are not duplicated per process.
    restrict_logging_to_rank0(rank)

    # Instantiate the model (for each process if many)
    # Seed with the run-wide value un-offset: the engine captures it at construction and
    # uses it wherever the ranks must agree - sampler ordering, multi-ring train/val split.
    torch.manual_seed(hydra_config.seed)

    model, nb_params = build_model(
        model_config=hydra_config.model,
        device=device,
        use_ddp=(len(gpu_list) > 1)
    )
    if wandb_run is not None:
        wandb_run.log({'nb_params': nb_params})

    # A model constructor calling torch.manual_seed() would quietly take over the run's
    # randomness and make the configured seed a no-op from here on
    # Catch it and restore the intended stream.
    if torch.initial_seed() != hydra_config.seed:
        log.warning(
            f"Model construction reseeded the global RNG (expected {hydra_config.seed}, "
            f"found {torch.initial_seed()}). Restoring the configured seed - remove the "
            f"torch.manual_seed() call from the model."
        )
        torch.manual_seed(hydra_config.seed)

    # Instantiate the engine (for each process if many)
    # Pass pre-built dataset if it exists (in_memory case)
    hydra_output_dir = os.getcwd()
    engine = instantiate(
        config=hydra_config.engine,
        dump_path=hydra_output_dir + "/",
        model=model,
        rank=rank,
        device=device,
        wandb_run=wandb_run,
        dataset=dataset
    )

    # From here on, offset by rank. Anything that *should* differ between ranks - dropout,
    # augmentation, and the DataLoader workers, whose seeds torch derives from this stream
    # now are different.
    torch.manual_seed(hydra_config.seed + rank)

    for task, task_config in hydra_config.tasks.items():

        with open_dict(task_config):

            # Configure dataset and data loaders
            if 'data_loaders' in task_config:
                engine.configure_dataset(data_config=hydra_config.data)
                engine.configure_data_loaders(loaders_config=task_config.pop("data_loaders"))

            # Configure optimizers
            if 'optimizers' in task_config:
                engine.configure_optimizers(task_config.pop("optimizers"))

            # Configure scheduler
            if 'scheduler' in task_config:
                engine.configure_scheduler(task_config.pop("scheduler"))

            # Configure loss
            if 'loss' in task_config:
                engine.configure_loss(task_config.pop("loss"))

            if 'early_stopping' in task_config:
                engine.configure_early_stopping(task_config.pop('early_stopping'))

    # Perform tasks - only "train" or "evaluate" should remain in the DictConfig
    for task, task_config in hydra_config.tasks.items():
        getattr(engine, task)(**task_config)

    if ( rank == 0 ) and ( wandb_run is not None ): # 1. First close W&B

        run_id = wandb_run.id
        log.info(f"run id : {run_id}")

        log.info(f"Calling wandb.finish()")
        time.sleep(sleep_time)         # (s) Pause the execution of the current thread for this time
        wandb_run.finish()             # Force clean exit (compared to no args)
        time.sleep(sleep_time)
        log.info(f"Done")

    if len(gpu_list) > 1: # 2. Then tear down DDP
        log.info(f"Calling destroy_process_group()")
        destroy_process_group()
        log.info(f"Finished.")
        # torch.cuda.empty_cache()  # Clear GPU memory
