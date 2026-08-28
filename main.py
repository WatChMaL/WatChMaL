"""
Main file used for running the code.

Single entry point for the unified WatChMaL core. Everything here is model-family
agnostic: the parent process settles the seed, optionally initialises wandb, optionally
prebuilds an in-memory dataset, then spawns the single worker
(watchmal/entrypoints/run.py) once per GPU. There is no longer a `core:` switch - CNN,
graph and multi-ring runs all go through the same worker and the same engine base; a
model family is selected purely by the hydra `engine`/`model`/`data` configs.

The worker signature is `run(rank, gpu_list, dataset, wandb_run, wandb_run_id,
hydra_config, global_hydra_config)`. `dataset` is non-None only for in-memory (pyg)
datasets prebuilt here so each worker does not reload them. A live `wandb.Run` can't
survive being pickled into mp.spawn's fresh interpreters (it owns a socket to wandb's
background service process), so under DDP the parent finishes the run and passes only
`wandb_run_id`; rank 0 re-attaches to the same run inside run() instead of receiving the
object itself. In the single-process path there's no pickling boundary to cross, so
`wandb_run` is passed directly and `wandb_run_id` is unused.
"""

# hydra imports
import hydra
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig

# torch imports
import torch
import torch.multiprocessing as mp

# generic imports
import logging
import os

from watchmal.utils.logging_utils import get_git_version
from watchmal.entrypoints.run import run

log = logging.getLogger(__name__)


@hydra.main(config_path='tutorial/config/watchmal', config_name='resnet_train', version_base="1.1")
def main(config):
    """
    Run model using given config, spawn worker subprocesses as necessary

    Args:
        config  ... hydra config specified in the @hydra.main annotation
    """
    log.info(f"Using the following git version of WatChMaL repository: {get_git_version(os.path.dirname(to_absolute_path(__file__)))}")
    log.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    global_hydra_config = HydraConfig.get()

    # Seed policy is settled here, before the spawn, so every rank shares the same value.
    # If no seed is given one is generated (and logged); a config may still pin it.
    if config.get("seed", None) is None:
        config.seed = torch.seed()
        log.info(f"No seed provided; generated seed {config.seed}")

    if config.gpu_list is None:
        config.gpu_list = []
    gpu_list = config.gpu_list
    ngpus = len(gpu_list)

    # wandb is optional; imported lazily so a CSV-only run never needs the package.
    if config.get("launch_wandb", False) or ('WANDB_SWEEP_ID' in os.environ):
        import wandb
        wandb_conf = config.get("wandb", None)
        if wandb_conf is None:
            raise SystemExit(
                "\nlaunch_wandb is set (or WANDB_SWEEP_ID is in the environment) but the "
                "config has no `wandb` section to initialise from.\n"
            )
        wandb_run = wandb.init(**OmegaConf.to_container(wandb_conf, resolve=True))
        # A wandb sweep overrides config values; fold them back in.
        from watchmal.utils.build_utils import merge_config
        config = merge_config(config, wandb.config)
        wandb.config.update(OmegaConf.to_container(config))
    else:
        wandb_run = None

    # In-memory (pyg) datasets are built once here so each spawned worker does not reload
    # them; every other dataset (which carries no `kind`) is built inside the engine.
    data_config = config.get("data", None)
    dataset_config = data_config.get("dataset", None) if data_config is not None else None
    dataset_kind = dataset_config.get("kind", "") if dataset_config is not None else ""
    if 'in_memory' in dataset_kind:
        if 'pyg_in_memory' in dataset_kind:
            from watchmal.dataset.graph.data_utils import get_dataset
            dataset = get_dataset(data_config)
        else:
            raise ValueError(f"Unknown in_memory dataset kind: {dataset_kind}")
    else:
        dataset = None

    if ngpus > 1:
        wandb_run_id = None
        if wandb_run is not None:
            # A live wandb.Run holds a socket to wandb's background service process; it
            # can't survive being pickled into mp.spawn's fresh interpreters. Finish it
            # here and pass only the id — rank 0 re-attaches to the same run inside
            # run() instead of receiving the object itself.
            wandb_run_id = wandb_run.id
            wandb_run.finish()
            wandb_run = None
            for k in ("WANDB_SERVICE", "WANDB_RUN_ID", "WANDB_RUN_GROUP"):
                os.environ.pop(k, None)

        log.info(f"Using DistributedDataParallel on devices: {[f'cuda:{x}' for x in gpu_list]}")
        mp.spawn(run, nprocs=ngpus,
                 args=(gpu_list, dataset, wandb_run, wandb_run_id, config, global_hydra_config))
    else:
        log.info("Single device, not using multiprocessing")
        run(0, gpu_list, dataset, wandb_run, None, config, global_hydra_config)


if __name__ == '__main__':
    main()
