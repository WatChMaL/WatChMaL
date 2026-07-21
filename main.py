"""
Main file used for running the code

Everything up to the worker spawn is core-agnostic. The per-core startup lives in
watchmal/entrypoints/run_<core>.py and is selected by the optional top-level `core`
key, which defaults to the watchmal core so existing configs are unaffected.
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

CORES = ("watchmal",)


def select_run(core):
    """
    Return the worker function of the requested core.

    The import is done here rather than at module level so that a run only pulls in the
    modules of the core it actually uses.

    Args:
        core ... name of the core to run, one of CORES

    Returns:
        the core's `run(rank, config, hydra_config=None)` function
    """
    if core == "watchmal":
        from watchmal.entrypoints.run_watchmal import run
    else:
        raise ValueError(f"Unknown core '{core}', expected one of {list(CORES)}")
    return run


@hydra.main(config_path='config/', config_name='resnet_train', version_base="1.1")
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

    if config.gpu_list is None:
        config.gpu_list = []
    ngpus = len(config.gpu_list)
    is_distributed = ngpus > 1

    # Initialize process group env variables
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'

        master_port = config.get("MASTER_PORT", 12355)

        # Automatically select port based on base gpu
        master_port += config.gpu_list[0]
        os.environ['MASTER_PORT'] = str(master_port)

    # create run directory
    os.makedirs(config.dump_path, exist_ok=True)
    log.info(f"Output directory: {config.dump_path}")

    # initialize seed
    if config.seed is None:
        config.seed = torch.seed()
    torch.manual_seed(config.seed)

    if is_distributed:
        log.info("Using multiprocessing...")
        devids = [f"cuda:{x}" for x in config.gpu_list]
        log.info(f"Using DistributedDataParallel on these devices: {devids}")
        mp.spawn(run, nprocs=ngpus, args=(config, HydraConfig.get()))
    else:
        log.info("Only one device found, not using multiprocessing...")
        run(0, config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
