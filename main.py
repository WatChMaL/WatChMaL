import logging
import hydra
from omegaconf import OmegaConf

#from pytorch_lightning import Trainer
from hydra.utils import instantiate

from watchmal.dataset.data_module import DataModule

logger = logging.getLogger('train')

@hydra.main(config_path='config/', config_name='resnet_example_config')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    data = DataModule(**config.data)

    model = instantiate(config.model)

    optim = -1

    engine = instantiate(config.engine, model=model, optimizer=optim, data=data)

    if (config.train is not None):
        engine.train(train_config=config.train)

    #engine.evaluate(test_config=config.test)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()