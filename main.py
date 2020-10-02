import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('train')


@hydra.main(config_path='config/', config_name='resnet_train')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    #TODO: is this needed
    ngpus_per_node = torch.cuda.device_count()

    if (len(config.gpu_list)) > 1:
        print("Using multiprocessing")
        print("Requesting GPUs. GPU list : " + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in config.gpu_list]
        print("Using DistributedDataParallel on these devices: {}".format.devids)
    else:
        print("Only one gpu found")
        gpu = config.gpu_list[0]
        main_worker_function(gpu, ngpus_per_node, config)

def main_worker_function(gpu, ngpus_per_node, config):
    print("Main loop")

    # Instantiate model and engine
    model = instantiate(config.model)

    # configure the device to be used for model training and inference
    if len(config.gpu_list) > 1:
        # if more than one gpu given, then we must be using multiprocessing
        model = DDP(model, device_ids=[gpu], output_device=gpu)
    
    engine = instantiate(config.engine, model=model, gpu=gpu)

    # Configure optimizers and data loaders
    for task, task_config in config.tasks.items():
        print(task)
        if 'data_loaders' in task_config:
            engine.configure_data_loaders(config.data, task_config.data_loaders)
        """
        if 'optimizer' in task_config:
            engine.configure_optimizers(task_config.optimizer)
        """

    """
    # Reload previous state
    if 'load_state' in config:
        engine.reload(config.load_model)

    # Perform tasks
    for task, task_config in config.tasks.items():
        getattr(engine, task)(task_config)
    """

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()