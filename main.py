"""
Main file used for running the code

This is the single entry point for both cores. Everything up to the worker spawn is
core-agnostic; the per-core startup lives in watchmal/entrypoints/run_<core>.py and is
selected by the optional top-level `core` key, which defaults to the watchmal core so
existing configs are unaffected.

The worker functions share one signature -
`run(rank, gpu_list, dataset, wandb_run, hydra_config, global_hydra_config)` - so the
spawn below is uniform; the watchmal worker ignores `dataset`/`wandb_run` (it builds
datasets in the engine and has no wandb integration), while the caverns worker uses
them (in-memory dataset prebuilt here, wandb run initialised here).
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

log = logging.getLogger(__name__)

CORES = ("watchmal", "caverns")


def select_run(core):
    """
    Return the worker function of the requested core.

    The import is done here rather than at module level so that a run only pulls in the
    modules of the core it actually uses.

    Args:
        core ... name of the core to run, one of CORES

    Returns:
        the core's `run(rank, gpu_list, dataset, wandb_run, hydra_config, global_hydra_config)`
    """
    if core == "watchmal":
        from watchmal.entrypoints.run_watchmal import run
    elif core == "caverns":
        from watchmal.entrypoints.run_caverns import run
    else:
        raise ValueError(f"Unknown core '{core}', expected one of {list(CORES)}")
    return run


@hydra.main(config_path='tutorial/config/watchmal', config_name='resnet_train', version_base="1.1")
def main(config):
    """
    Run model using given config, spawn worker subprocesses as necessary

    Args:
        config  ... hydra config specified in the @hydra.main annotation
    """
    log.info(f"Using the following git version of WatChMaL repository: {get_git_version(os.path.dirname(to_absolute_path(__file__)))}")
    log.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    core = config.get("core", "watchmal")
    run = select_run(core)
    log.info(f"Running with the '{core}' core")

    global_hydra_config = HydraConfig.get()

    # Seed policy is core-aware and settled here, before the spawn, so every rank shares
    # the same value. The watchmal core auto-generates one if none is given (upstream
    # behaviour; every shipped watchmal config uses seed: null); the caverns core
    # requires one.
    if config.get("seed", None) is None:
        if core == "caverns":
            log.error("No seed provided. The caverns core requires an explicit top-level `seed in the main config file")
            raise SystemExit(1)
        config.seed = torch.seed()
        log.info(f"No seed provided; generated seed {config.seed}")

    if config.gpu_list is None:
        config.gpu_list = []
    gpu_list = config.gpu_list
    ngpus = len(gpu_list)

    # wandb is a caverns feature for now; imported lazily so the watchmal core never needs it.
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

    # In-memory (pyg) datasets are built once here so each spawned worker does not
    # reload them; every other dataset (all watchmal ones, which carry no `kind`) is
    # built by the engine.
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
        log.info(f"Using DistributedDataParallel on devices: {[f'cuda:{x}' for x in gpu_list]}")
        mp.spawn(run, nprocs=ngpus,
                 args=(gpu_list, dataset, wandb_run, config, global_hydra_config))
    else:
        log.info("Single device, not using multiprocessing")
        run(0, gpu_list, dataset, wandb_run, config, global_hydra_config)


if __name__ == '__main__':
    main()
