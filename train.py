import logging
import hydra
from omegaconf import OmegaConf

#from pytorch_lightning import Trainer
from hydra.utils import instantiate

from watchmal.dataset.data_module import DataModule

logger = logging.getLogger('train')

# TODO: change config_name back to 'train' rather than 'train_pointnet'
@hydra.main(config_path='config/', config_name='train_pointnet')
def main(config):
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(config)}")

    data = DataModule(**config.data)

    engine = instantiate(config.engine, model_config=config.model, train_config=config.train, data=data)

    engine.train()

    engine.evaluate()

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
