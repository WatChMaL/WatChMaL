"""
Main file used for running the code
"""

# hydra imports
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp

# generic imports
import logging
import os

from watchmal.utils.logging_utils import get_git_version

log = logging.getLogger(__name__)



@hydra.main(config_path='config/', config_name='resnet_train', version_base="1.1")
def main(config):
    """
    Run model using given config, spawn worker subprocesses as necessary

    Args:
        config  ... hydra config specified in the @hydra.main annotation
    """
    log.info(f"Using the following git version of WatChMaL repository: {get_git_version(os.path.dirname(to_absolute_path(__file__)))}")
    log.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    ngpus = len(config.gpu_list)
    is_distributed = ngpus > 1
    
    # Initialize process group env variables
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'

        if 'MASTER_PORT' in config:
            master_port = config.MASTER_PORT
        else:
            #automatically find a free port
            import socket
            def find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    return s.getsockname()[1]
            master_port = find_free_port()
            
        # Automatically select port based on base gpu
        os.environ['MASTER_PORT'] = str(master_port)

    # create run directory
    if not os.path.exists(config.dump_path):
        log.info(f"Creating directory for run output at : {config.dump_path}")
        os.makedirs(config.dump_path)
    
    log.info(f"Output directory: {config.dump_path}")

    # initialize seed
    if config.seed is None:
        config.seed = torch.seed()
    
    if is_distributed:
        log.info("Using multiprocessing...")
        devids = [f"cuda:{x}" for x in config.gpu_list]
        log.info(f"Using DistributedDataParallel on these devices: {devids}")
        mp.spawn(main_worker_function, nprocs=ngpus, args=(ngpus, is_distributed, config, HydraConfig.get()))
    else:
        log.info("Only one gpu found, not using multiprocessing...")
        main_worker_function(0, ngpus, is_distributed, config)


def main_worker_function(rank, ngpus_per_node, is_distributed, config, hydra_config=None):
    """
    Instantiate model on a particular GPU, and perform train/evaluation tasks as specified

    Args:
        rank            ... rank of process among all spawned processes (in multiprocessing mode)
        ngpus_per_node  ... number of gpus being used (in multiprocessing mode)
        is_distributed  ... boolean indicating if running in multiprocessing mode
        config          ... hydra config specified in the @hydra.main annotation
        hydra_config    ... HydraConfig object for logging in multiprocessing
    """
    if is_distributed:
        # Spawned process needs to configure the job logging configuration
        configure_log(hydra_config.job_logging, hydra_config.verbose)
        # Set up pytorch distributed processing
        torch.distributed.init_process_group('nccl', init_method='env://', world_size=ngpus_per_node, rank=rank)
    if ngpus_per_node == 0:
        device = torch.device("cpu")
    else:
        # Infer rank from gpu and ngpus, rank is position in gpu list
        device = config.gpu_list[rank]
        torch.cuda.set_device(device)
    log.info(f"Running main worker function rank {rank} on device: {device}")

    log.info(f'I am about to instantiate the mode on {rank} on device: {device}')
    # Instantiate model and engine
    model = instantiate(config.model).to(device)
    log.info(f'I have successfully instantiated the mode on {rank} on device: {device}')

    # Configure the device to be used for model training and inference
    if is_distributed:
        log.info(f'I am about to convert model batch norms to syncbatchnorm on {rank} on device: {device}')
        # Convert model batch norms to synchbatchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        log.info(f'I have successfully converted model batch norms to syncbatchnorm on {rank} on device: {device}')
        log.info(f'I am about to do DPP on {rank} on device: {device}')
        model = DDP(model, device_ids=[device])
        log.info(f'I have successfully done DPP on {rank} on device: {device}')

        log.info(f"Device {device} has arrived at the first barrier")
        torch.distributed.barrier()

    log.info(f"I am about to instantiate engine on {rank} on device: {device}")
    # Instantiate the engine
    engine = instantiate(config.engine, model=model, rank=rank, device=device, dump_path=config.dump_path)
    log.info(f"I have successfully instantiated engine on {rank} on device: {device}")

    for task, task_config in config.tasks.items():
        if is_distributed:
            # Before each task, ensure GPUs are in sync to avoid e.g. loading a state before a GPU finished training
            log.info(f'I am about to do torch.distributed.barrier() on {rank} on device: {device}')
            torch.distributed.barrier()
            log.info(f'I have successfuly run torch.distributed.barrier() on {rank} on device: {device}')
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
        getattr(engine, task)(**task_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
