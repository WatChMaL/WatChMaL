import logging
import hydra
from omegaconf import OmegaConf

#from pytorch_lightning import Trainer
from hydra.utils import instantiate

from watchmal.dataset.data_module import DataModule

logger = logging.getLogger('train')

@hydra.main(config_path='config/', config_name='test_example_config')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")

    data = DataModule(**config.data)

    model = instantiate(config.model)

    engine = instantiate(config.engine, model=model, data=data)

    engine.configure_optimizers(optimizer_config=config.optimizer)

    loc = '/home/jtindall/WatChMaL/outputs/2020-09-29/test_reload/outputs/ClassifierBEST.pth'
    engine.restore_state(loc)
    #engine.dirpath = loc

    if ('train' in config):
        engine.train(train_config=config.train)

    if ('test' in config):
        engine.evaluate(test_config=config.test)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()