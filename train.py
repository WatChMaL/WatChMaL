import logging
import hydra
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from hydra.utils import instantiate

from watchmal.dataset.data_module import DataModule

logger = logging.getLogger('train')


@hydra.main(config_path='config/', config_name='train')
def main(config):
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(config)}")
    data = DataModule(**config.data)
    trainer_logger = instantiate(config.logger) if "logger" in config else True

    engine = instantiate(config.engine, model_config=config.model, train_config=config.train)
    logger.info(engine)
    trainer = Trainer(gpus=config.train.gpus, logger=trainer_logger)
    trainer.fit(engine, data)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
