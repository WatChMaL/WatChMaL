import logging
import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from hydra.utils import instantiate

logger = logging.getLogger('train')


@hydra.main(config_path='config/', config_name='train')
def main(config):
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(config)}")
    data = instantiate(config.dataset)
    trainer_logger = instantiate(config.logger) if "logger" in config else True

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    logger.info(model)
    trainer = Trainer(logger=trainer_logger)
    trainer.fit(model, data)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
