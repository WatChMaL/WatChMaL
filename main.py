import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

logger = logging.getLogger('train')


@hydra.main(config_path='config/', config_name='resnet_train')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    # Instantiate model and engine
    model = instantiate(config.model)
    engine = instantiate(config.engine, model=model)

    # Configure optimizers and data loaders
    for task, task_config in config.tasks.items():
        if 'optimizer' in task_config:
            engine.configure_optimizers(task_config.optimizer)
        if 'data_loaders' in task_config:
            engine.configure_data_loaders(config.data, task_config.data_loaders)

    # Reload previous state
    if 'load_state' in config:
        engine.reload(config.load_model)

    # Perform tasks
    for task, task_config in config.tasks.items():
        getattr(engine, task)(task_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()