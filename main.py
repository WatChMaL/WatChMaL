import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

logger = logging.getLogger('train')


@hydra.main(config_path='config/', config_name='resnet_train')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    model = instantiate(config.model)

    engine = instantiate(config.engine, model=model)

    if 'load_state' in config:
        engine.reload(config.load_model)

    for task, task_config in config.tasks.items():
        engine.configure_optimizers(task_config.optimizer)
        engine.configure_data_loaders(config.data, task_config.data_loaders)
        getattr(engine, task)(task_config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()