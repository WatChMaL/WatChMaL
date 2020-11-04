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

Examples:

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

