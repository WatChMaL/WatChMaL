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

    engine = instantiate(config.engine, model=model, data=data)

    engine.configure_optimizers(config.optimizer)

    if ("train" in config):
        engine.train(train_config=config.train)

    if ("test" in config):
        engine.evaluate(test_config=config.test)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()