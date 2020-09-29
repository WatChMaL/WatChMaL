import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

logger = logging.getLogger('train')


@hydra.main(config_path='config/', config_name='resnet_example_config')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    model = instantiate(config.model)

    engine = instantiate(config.engine, model=model, data_config=config.data, task_config=config.tasks)

    if 'load_state' in config:
        engine.reload(config.load_model)

    for task, config in config.tasks.items():
        getattr(engine, task)(config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()