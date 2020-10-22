# hydra imports
import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

# torch imports
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# TODO: see if this can be removed
torch.multiprocessing.set_sharing_strategy('file_system')

# generic imports
import os

logger = logging.getLogger('train')

@hydra.main(config_path='config/', config_name='resnet_train')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    ngpus = len(config.gpu_list)
    is_distributed = ngpus > 1
    
    # Initialize process group env variables
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'

        if 'MASTER_PORT' in config:
            master_port = config.MASTER_PORT
        else:
            master_port = 12355
        # Automatically select port based on base gpu
        master_port += config.gpu_list[0]
        os.environ['MASTER_PORT'] = str(master_port)

    # create run directory
    try:
        os.stat(config.dump_path)
    except:
        print("Creating a directory for run dump at : {}".format(config.dump_path))
        os.makedirs(config.dump_path)
    
    print("Dump path: {}".format(config.dump_path))
    
    if is_distributed:
        print("Using multiprocessing...")
        devids = ["cuda:{0}".format(x) for x in config.gpu_list]
        print("Using DistributedDataParallel on these devices: {}".format(devids))
        mp.spawn(main_worker_function, nprocs=ngpus, args=(ngpus, is_distributed, config))
    else:
        print("Only one gpu found, not using multiprocessing...")
        main_worker_function(0, ngpus, is_distributed, config)

def main_worker_function(rank, ngpus_per_node, is_distributed, config):
    # infer rank from gpu and ngpus, rank is position in gpu list
    gpu = config.gpu_list[rank]

    print("Running main worker function on device: {}".format(gpu))
    torch.cuda.set_device(gpu)

    world_size = ngpus_per_node
    
    if is_distributed:
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )

    # Instantiate model and engine
    model = instantiate(config.model).to(gpu)

    # configure the device to be used for model training and inference
    if is_distributed:
        # Convert model batch norms to synchbatchnorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # Instantiate the engine
    engine = instantiate(config.engine, model=model, rank=rank, gpu=gpu, dump_path=config.dump_path)
    
    # Configure data loaders
    for task, task_config in config.tasks.items():
        if 'data_loaders' in task_config:
            engine.configure_data_loaders(config.data, task_config.data_loaders, is_distributed)
    
    # Configure optimizers
    for task, task_config in config.tasks.items():
        if 'optimizers' in task_config:
            engine.configure_optimizers(task_config.optimizers)

    # Reload previous state
    if 'load_state' in config:
        engine.restore_state(config.load_state)

    # Perform tasks
    for task, task_config in config.tasks.items():
        getattr(engine, task)(task_config)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()