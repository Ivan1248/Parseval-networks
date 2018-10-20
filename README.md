# Parseval-networks
This repository is the result of my universtity project with the goal to implement some of the models and reproduce some of the results from the [Parseval networks](https://arxiv.org/abs/1704.08847) paper. The main idea of the paper is to control the Lipschitz norm of the model by enforcing weight matrices to be nearly orthognal and having a spectral norm of 1.

**NOTE**: There are some (unintended) differences between the original models and those implemented here. The Wide ResNet implementation has a bit worse performance than the [original implementation](https://github.com/szagoruyko/wide-residual-networks). The results of the experiments are in section 3 in this [report](https://github.com/Ivan1248/Parseval-networks/blob/master/report/izvjestaj.pdf) (it's in Croatian).

### Code structure

Due to time constraints (and abandonment of the code) the code (especially the outer parts of it) is not very well designed and is undocumented. Here is a directory tree describing some parts of the code:
```
code 
├── cleverhans  # code for adversarial examples, copied from https://github.com/tensorflow/cleverhans
├── data
│   ├── dataset_dir.py
│   ├── dataset.py
│   └── loaders.py
├── models
│   |── tf_utils
│   |   ├── __init__.py
│   |   ├── layers.py  # functions for parts of tensorflow models
│   |   ├── losses.py
│   |   ├── regularization.py
│   |   ├── update_ops.py  # here the orthogonality retraction step is defined
│   |   └── variables.py
|   ├── abstract_model.py
│   ├── parseval_resnet.py
│   └── resnet.py
├── processing
├── ioutils
├── visualization
├── data_utils.py  # abstractions for the `data` module
├── dirs.py  # filesystem paths
├── standard_resnets.py  # resnet factory
├── training.py  # `train` function
|
├── display_adversarial_examples_by_eps.py
├── display_adversarial_examples.py
├── load_wrn_test.py
├── plot_adversarial_robustness.py
├── plot_classification_error_curves.py
├── train_wrn_pars.py
├── train_wrn_pars_test.py
├── train_wrn.py
└── train_wrn_test.py
```

The parts of code specific to Parseval resnets are the `convex combination` function in `code/models/tf_utils/layers.py`, which is used in the `residual_block` function in the same file, and the `ortho_retraction_step` function in  `code/models/tf_utils/update_ops.py`, which is used in `code/models/parseval_resnet.py`. 

### Prerequisites

`code/dirs.py` requires that a directories matching the following regular expressions exist ('.' and '/' are not escaped for readability).
```
.(/..)+/data/datasets    # some ancestor directory contains `data/datasets`
.(/..)+/data/models    # some ancestor directory contains `data/models`
```

#### Dataset directory

The `<ancestor>/data/datasets` directory (stored in the `dirs.DATASETS` variable in `code/dirs.py`) needs to contain the required datasets. For example, for CIFAR-10 it needs to contain this directory subtree:
```
cifar-10-batches-py/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

#### Saved trained models directory

The `<ancestor>/data/models` directory (stored in the `dirs.SAVED_MODELS` variable in `code/dirs.py`) is used for saving trained model parameters and training and evaluation information.

### Usage

#### Training

To train a model, run one of the following scripts: 
```
train_wrn.py <depth> <width>
train_wrn_test.py <depth> <width>
train_wrn_pars.py <depth> <width>
train_wrn_pars_test.py <depth> <width>
```
The scripts containing "pars" are for training Parseval wide resnets. The scripts containing "test" are for training on the whole CIFAR-10 training test and testing on the test set, whereas those without "test" are for training on 80% of the training set and validation on the remaining 20% of the training set. Example: `python train_wrn_test.py 28 10` trains a WRN-28-10 on CIFAR-10-train.

#### Testing

To test a saved trained model, you need to *modify* (because saved model names are hardcoded) and run one of the following scripts: 
```
load_wrn_test.py
plot_adversarial_robustness.py
plot_classification_error_curves.py
```

You can also try these (which also need to be *modified*):
```
display_adversarial_examples_by_eps.py
display_adversarial_examples.py
```