# Water Cherenkov Machine Learning (WatChMaL)

# Description

Unified framework for the training, testing and using Machine Learning models for Water Cherenkov Detectors.

# Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
    1. [Basic Usage](#basic-usage)
    2. [Configuration](#configuration)
        1. [Hydra configuration framework](#hydra-configuration-framework)
        2. [WatChMaL configuration](#watchmal-configuration)
        3. [Configuration example](#configuration-example)
3. [Development](#development)

# Installation

Requirements can be found in requirements.txt.
The main requirements are [pytorch](pytorch.org), [numpy](numpy.org) and [hydra](hydra.cc).

To download the repository use :

```
git clone https://github.com/WatChMaL/WatChMaL.git
```

# Usage
## Basic Usage

From within the WatChMaL/WatChMaL repository, run `main.py` in Python 3.
 
Using `-c job` to the options will tell the hydra config system to print the full config and then exit.
 
```
python main.py -c job [config options]
```

where the config options are will generally provide a top-level config file
as well as any additional config parameters overrides, as described in more detail below.

After checking the printed config, removing the `-c job` option and re-running the command will then print the config
and start the actual job.

```
python main.py [config options]
```

The minimal command will simply provide a single config file. For example:
```
python main.py --config-name=resnet_train
```
will load the configuration from `config/resnet_train.yaml`.

## Configuration

### Hydra configuration framework

Hydra is used to dynamically create a hierarchical configuration by composition and override it through config files
and the command line. The composed configuration effectively provides a dictionary of key-value pairs
(where the configuration hierarchy is built through values that are themselves configuration dictionaries),
with only minimal or even no code changes needed to handle new configuration options when they
are introduced.

Running main.py with the argument `--hydra-help` gives documentation on how to use Hydra's command line options
for controlling the configuration.

Full Hydra documentation can be found here: https://hydra.cc/docs/intro/

The default directory for config files is the `config` directory.
This can be changed using the `--config-path` option, but this is rarely necessary.
Related config options are collected together into config groups, using subdirectories that contain config files with
different sets of config options.
For example, the `config/data/` directory contains YAML files for different configurations of data to use in training.

Configuration groups can set a `_target_` key with a value that refers to a python class or method that is automatically
instantiated, allowing new features to be introduced simply by creating the relevant classes or methods, or using
existing library classes or methods, and simply referring to these at the appropriate place in the configuration.

The main config file, set using the `--config-name` option corresponds to the name of a yaml file
for the desired job configuration. This can contain the full hierarchy of config options self-contained in one file.
It can also contain a list of "defaults" which are not actually default configs,
but just specify which config file to use in a config group's subdirectory.
Hydra will then compose the full configuration from the specified files.

Command line options can be used to override parameters e.g. `tasks.train.epochs=50`,
add parameters using `+` (e.g. `+tasks.train.epochs=50`), or remove parameters using `~` (e.g. `~tasks.train.epochs`).
Similarly, config groups can be set by command line, e.g. `tasks/train=train_resnet` will get the train task's config
from the `config/tasks/train/train_resnet.yaml` file.

### WatChMaL configuration

The best way to understand the WatChMaL configuration options is to explore the example configuration files in the
`config` directory, where the config files are designed intended be readable and self-documenting.
Some configuration options refer to arbitrary classes through the special hydra config , for example the optimizer configuration
could use any optimizer class provided by pytorch, including configuring that optimizer's parameters, so it is not
possible to comprehensively document all possible configuration options.

Generally, the configuration options for WatChMaL are comprised of only a few top-level options with the rest
structured into config groups: `data`, `model`, `engine` and `tasks`.

The main WatChMaL process uses this configuration system to run jobs that may consist of training loops, loading saved
states, and/or testing or evaluating the model. The basic steps performed in each WatChMaL run are:

1. Set up the global configuration such as the GPUs being used, along with distributed multi-GPU and multiprocessing,
random number seeds, etc. After this step, for distributed runs the following steps are performed by each distributed
process.
2. Instantiate the model (that defines the neural network architecture)
3. Instantiate the engine (that defines the tasks to be performed, e.g. training loops)
4. Configure any data loaders used by each of the tasks to be performed.
5. Configure any optimizers used by each of the tasks to be performed.
6. Perform each of the tasks.

#### Top-level options
Top level options include the list of GPUs to use, the random seed, and the output path.
For example, a configuration may specify the following top-level options:
```
gpu_list: [0,1]
seed: 1234
dump_path: ./outputs/
```

#### `data`
The `data` config group defines the data used for the job. Generally it will contain a `dataset` subconfig, with a
`_target_` that refers to a class that implements the PyTorch `Dataset` class, and the remaining config defines that
class's options.
The `data` config group will also configure how the data is split between training, validation and testing, for example.

#### Example using `CNNDatasetDeadPMT`

Using `CNNDatasetDeadPMT` allows you to "turn off" some PMTs during training and evaluation, which can be helpful for simulating the effects of dead PMTs. "Turning off" dead PMTs involves setting charge and time to 0.

There are two ways to select the dead PMTs:

1. Provide a text file specifying the IDs of the dead PMTs. The path to the `.txt` file is specified in `dead_pmts_file`.
2. Randomly set the dead PMTs by providing a probability and seed value, referred to as `dead_pmt_rate` and `dead_pmt_seed`.

The first method takes priority. That is, if both `dead_pmts_file` and `dead_pmt_rate`/`dead_pmt_seed` are provided, the class reads the dead PMT IDs from the file and does not generate them randomly.

If `use_dead_pmt_mask` is `True`, the class adds another channel where a pixel is 1 if it corresponds to a dead PMT and 0 otherwise.

**Example:** under `dataset`,

```yaml
dead_pmt_rate: 0.03      # must be in [0, 1]
dead_pmt_seed: 5         # must be an integer
dead_pmts_file: /data/fcormier/Public/dead_pmts.sk4.txt
use_dead_pmt_mask: True
```


#### `model`
The `model` config group defines the network architecture being used. It will generally have a `_target_` that refers to
a class that implements a PyTorch `Module` and will set the options of that class.

#### `engine`
The `engine` config group defines the main engine class to be used, as well as its options.

#### `tasks`
The `tasks` config is a collection of tasks for the job to perform. Each element of the list must be a config group with
a `_target_` that refers to a method of the engine class and the method's parameters. Each task can also contain
`data_loaders` and `optimizers` sub-configs, which are a list of subconfigs defining the `DataLoader` and `Optimizer`
classes used in the task.

### Configuration example

A basic example of a configuration file for training and testing a ResNet CNN for classification could be
```yaml
gpu_list:
    - 0
    - 1
seed: 1234
dump_path: './outputs/'
defaults:
    - data: iwcd_short
    - data/dataset: iwcd_cnn_short
    - model: classifier
    - model/feature_extractor: resnet18
    - model/classification_network: resnet_fc
    - engine: classifier
    - tasks/train: train_resnet
    - optimizers@tasks.train.optimizers: adam
    - sampler@tasks.train.data_loaders.train.sampler: subset_random
    - sampler@tasks.train.data_loaders.validation.sampler: subset_random
    - tasks/restore_best_state: restore_best_state
    - tasks/evaluate: test
    - sampler@tasks.evaluate.data_loaders.test.sampler: subset_sequential
    - _self_
```
The top-level options are set directly, while all remaining configuration is defined through the "defaults" list, which
provides a list of config group files to compose the full configuration. E.g. the `data` configuration will load the
`config/data/iwcd_short.yaml` file, with sub-config group `dataset` from `config/data/dataset/iwcd_cnn_short.yaml`.

This example configuration performs three tasks:
* The `train` task that trains the model (configured through the `config/tasks/train/train_resnet.yaml` file).
* The `restore_best_state` task that loads the state that had the best validation loss during training.
* The `evaluate` task that evaluates the model.

Each task will define its own data loaders, optimizers, samplers, etc.

Lines in the configuration defaults list like
```
    - optimizers@tasks.train.optimizers: adam
```
indicate that the  `config/optimizers/adam.yaml` file should define the subconfig of `tasks.train.optimizers`.

The full composed configuration for the above example could then look something like

```yaml
data:
  split_path: /fast_scratch/WatChMaL/data/IWCD_mPMT_Short/index_lists/4class_e_mu_gamma_pi0/IWCD_mPMT_Short_4_class_3M_emgp0_idxs.npz
  dataset:
    h5file: /fast_scratch/WatChMaL/data/IWCD_mPMT_Short/IWCD_mPMT_Short_emgp0_E0to1000MeV_digihits.h5
    _target_: watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset.CNNmPMTDataset
    mpmt_positions_file: /data/WatChMaL/data/IWCDshort_mPMT_image_positions.npz
    collapse_arrays: false
    pad: true
model:
  _recursive_: false
  _target_: watchmal.model.classifier.Classifier
  num_classes: 4
  feature_extractor:
    _target_: watchmal.model.resnet.resnet18
    num_input_channels: 19
    num_output_channels: 128
  classification_network:
    _target_: watchmal.model.classifier.ResNetFullyConnected
engine:
  _target_: watchmal.engine.engine_classifier.ClassifierEngine
tasks:
  train:
    epochs: 20
    report_interval: 100
    val_interval: 100
    num_val_batches: 32
    checkpointing: false
    data_loaders:
      train:
        split_key: train_idxs
        batch_size: 512
        num_workers: 4
        transforms:
        - horizontal_flip
        - vertical_flip
        - front_back_reflection
        sampler:
          _target_: torch.utils.data.sampler.SubsetRandomSampler
      validation:
        split_key: val_idxs
        batch_size: 512
        num_workers: 4
        sampler:
          _target_: torch.utils.data.sampler.SubsetRandomSampler
    optimizers:
      _target_: torch.optim.Adam
      lr: 0.0001
      weight_decay: 0
  restore_best_state:
    placeholder: configs can't be empty
  evaluate:
    data_loaders:
      test:
        split_key: test_idxs
        batch_size: 4096
        num_workers: 4
        sampler:
          _target_: watchmal.dataset.samplers.SubsetSequentialSampler
gpu_list:
- 0
- 1
seed: 1234
dump_path: ./outputs/
```

## Development

The WatChMaL codebase follows a very similar structure to the configuration described above
(in fact the configuration is structured to match the code).

The `main.py` file contains the main code that performs
the steps of each WatChMaL run:

1. Set up the global configuration such as the GPUs being used, along with distributed multi-GPU and multiprocessing,
random number seeds, etc. After this step, for distributed runs the following steps are performed by each distributed
process.
2. Instantiate the model (that defines the neural network architecture)
3. Instantiate the engine (that defines the tasks to be performed, e.g. training loops)
4. Configure any data loaders used by each of the tasks to be performed.
5. Configure any optimizers used by each of the tasks to be performed.
6. Perform each of the tasks.

The remaining code is structured into `dataset`, `model` and `engine` code.

- The `dataset` code defines PyTorch `DataSet` classes used by `DataLoader` objects. These `DataSet` classes contain the
code for loading the data from disk and processing it into the format required by the network. This code also defines
any data transformations used for e.g. data augmentation.
- The `model` code defines PyTorch `nn.Module` classes that form the neural network architecture.
- The `engine` code defines a class that contains all the methods corresponding to the tasks being run. For example,
this will include the code to configure the data loaders, optimizers, etc., the training loop itself, as well code to
save and restore states, and so on.
