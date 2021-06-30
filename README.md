## Trainer Class for Pytorch
Module **PyTorchTrainer** implements the training and evaluation functionalities for pytorch model. It has a modular design, which facilitates efficient extension through inheritance. This module implements two classes: `ClassifierTrainer`, which is used for classification problems (trained with cross entropy loss), and `RegressionTrainer`, which is used for regression problems (trained with mean squared error). 


Some of the features include training progress bar, total training time estimation, tensorboard logging, proper model validation (best weight selected via validation set if available), checkpoints and so on.


In addition, the module is designed in a modular manner that allows easy customization. Please see the examples how this can be done.  

## Installation
To install the module, clone this repository and change directory to PyTorchTrainer, then install using pip::

    pip install .  

### Documentation 
Detailed description of the interface can be found [here](https://github.com/viebboy/PyTorchTrainer/blob/master/interface.md)

### Examples
[train_cifar.py](https://github.com/viebboy/PyTorchTrainer/blob/master/examples/train_cifar.py) provides a basic example that uses `PyTorchTrainer.ClassifierTrainer` to train a CNN on CIFAR-10/CIFAR-100. 

[custom_metrics_example.py](https://github.com/viebboy/PyTorchTrainer/blob/master/examples/custom_metrics_example.py) provides an example that subclasses `PyTorchTrainer.ClassifierTrainer` to measure accuracy, precision, recall and f1 as metrics and using f1 as the validation metric when selecting the best model weight.  

[knowledge_distillation_example.py](https://github.com/viebboy/PyTorchTrainer/blob/master/examples/knowledge_distillation_example.py) provides an example that subclasses `PyTorchTrainer.ClassifierTrainer` to perform knowledge distillation between two networks.   

 
