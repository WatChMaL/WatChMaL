# Water Cherenkov Machine Learning (WatChMaL)

## Description

Unified framework for the building, training and testing of ML models for Water Cherenkov Detectors.

## Table of Contents

### 1. [Installation](#installation)
### 2. [Usage](#usage)

## Installation <a id="installation"></a>

Requirements can be found in requirements.txt.

To download the repository use :

`git clone https://github.com/WatChMaL/WatChMaL.git`

## Usage

# Run Examples:

```
# Train and evaluate a resnet model using the sample training config file on gpu 0

python main.py gpu_list=[0]
```

```
# Train and evaluate a resnet model using the sample training config file on gpus 0 and 1 (using DistributedDataParallel)

python main.py gpu_list=[0,1]
```

```
# Evaluate a pretrained resnet model using the sample evaluation config file on gpu 0

python main.py --config-name=resnet_test gpu_list=[0] tasks.restore_state.weight_file='filepath'
```

# Configuration

Hydra is used to create composable configs that allow modifications to be made without having to maintain separate configs.

Running main.py with the argument `--hydra-help` gives documentation on how to use Hydra's command line options for controlling the configuration.

Full Hydra documentation can be found here: https://hydra.cc/docs/tutorials/intro

Run parameters are collected into a main config that is passed to a main function when the code runs. The main config path is set with the command line arguments `--config-name` / `-cn` and `--config-path` / `-cp`, with the defaults defined in main.py:

```
@hydra.main(config_path='config/', config_name='resnet_train')
def main(config):
    logger.info(f"Running with the following config:\n{OmegaConf.to_yaml(config)}")
```

This main config specifies subconfigs which are loaded at runtime. An example main config is `resnet_train.yaml`:

```
gpu_list:
    - 0
    - 1
dump_path: './outputs/'
defaults:
    - data: iwcd_postveto_nomichel
    - data/dataset: iwcd_cnn
    - model: classifier
    - model/feature_extractor: resnet18
    - model/classification_network: resnet_fc
    - engine: classifier
    - tasks/train: train_resnet
    - optimizers@tasks.train.optimizers: adam
    - sampler@tasks.train.data_loaders.train.sampler: subset_random
    - sampler@tasks.train.data_loaders.validation.sampler: subset_sequential
    - tasks/evaluate: test
    - sampler@tasks.evaluate.data_loaders.test.sampler: subset_sequential

```

These subconfigs correspond to config groups (more on config groups can be found in the hydra documentation). Generally, all the parameters related to a certain task or object are grouped into separate config groups. The defaults list in the main config contains default config for relevant config groups.

You can list all config groups available for WatChMaL, and the full default config, using the `--help` command line argument.

```
$ python main.py --help
main is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)data: iwcd_postveto_nomichel

data/dataset: iwcd_cnn, iwcd_pointnet
engine: classifier
model: classifier
model/classification_network: pointnet_fc, resnet_fc
model/feature_extractor: pointnet, resnet18
optimizers: adam
sampler: subset_random, subset_sequential
tasks/evaluate: test
tasks/train: train_pointnet, train_resnet

== Config ==
Override anything in the config (foo.bar=value)

... [full default config listed]
```

Config parameters and groups can be overridden in the command line. For example, to run using default configuration:

```
python main.py
```

And to override the list of GPUs to use:

```
python main.py gpu_list=[0]
```

Default config groups (but not parameters) can also be overridden in the defaults list:

```
    - sampler@tasks.train.data_loaders.train.sampler: subset_random
```

Some of the config groups listed are used to instantiate objects using hydra. For example `subset_random.yaml` contains:

```
# @package _group_._name_
_target_: torch.utils.data.sampler.SubsetRandomSampler
```

And can be used to instantiate a SubsetRandomSampler, where `sampler_config` contains the config object obtained from loading `subset_random.yaml`:

```
from hydra.utils import instantiate

sampler = instantiate(sampler_config)
```

Other config packages are used for calling functions. The config object is passed as an instance of OmegaConf's DictConfig (used like a standard dictionary), and then its parameters are used as part of a function. In particular, this is how tasks like training and testing are organized. For example the training task config `train_resnet.yaml` contains parameters used to set up the training run:

```
# @package _group_
epochs: 20

report_interval: 10
val_interval: 10
num_val_batches: 32

checkpointing: False

data_loaders:
  train:
    split_key: train_idxs
    batch_size: 512
    num_workers: 4
    transforms:
      - horizontal_flip
      - vertical_flip
  validation:
    split_key: val_idxs
    batch_size: 512
    num_workers: 4
```
Which is then used in the call to the task:

```
def train(self, train_config):
    ...
    # initialize training params
    epochs          = train_config.epochs
    report_interval = train_config.report_interval
    val_interval    = train_config.val_interval
    num_val_batches = train_config.num_val_batches
    checkpointing   = train_config.checkpointing
```
In our set up, the `data_loaders` and `optimizers` subconfigs of each task (if there are any) are instantiated and configured separately, before performing the tasks.
