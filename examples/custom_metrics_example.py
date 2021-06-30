"""
In this example, we will customize the trainer to use custom metrics and perform validation based on one of the custom metric
by subclassing the ClassifierTrainer 
"""

from PyTorchTrainer import ClassifierTrainer, get_cosine_lr_scheduler
import os, sys, getopt, pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


"""
Here we will subclass ClassifierTrainer to define a new trainer that
- measure accuracy, precision, recall, f1 as performance
- validate and select the best model weight based on f1 score

To do so, we need to redefine the init function and the eval function
A safe way to do so is to go to the source code and copy the corresponding 
function that we want to modify, then modify the needed part in the new class 
"""

class Trainer(ClassifierTrainer):
    def __init__(self,
                 n_epoch,
                 epoch_idx=0,
                 lr_scheduler=get_cosine_lr_scheduler(1e-3, 1e-5),
                 optimizer='adam',
                 weight_decay=1e-4,
                 temp_dir='',
                 checkpoint_freq=1,
                 print_freq=1,
                 use_progress_bar=True,
                 test_mode=False): 

        # We call the init of the ClassifierTrainer to perform necessary initialization 
        super(Trainer, self).__init__(n_epoch,
                                      epoch_idx,
                                      lr_scheduler,
                                      optimizer,
                                      weight_decay,
                                      temp_dir,
                                      checkpoint_freq,
                                      print_freq,
                                      use_progress_bar,
                                      test_mode)

        # then redefine the name of the metrics
        # note that the actual metric computation is done in the eval() function
        self.metrics = ['acc', 'precision', 'recall', 'f1']
        # we also specify the monitor_metric is f1
        # and the monitor_direction is higher, which means higher values are better
        self.monitor_metric = 'f1'
        self.monitor_direction = 'higher' 

    def eval(self, model, loader, device):
        """
        the eval function should return a dictionary that contain metric_name/metric_value as key/value 
        if loader is None, it should return an empty dictionary
        """
        # note that this step that returns an empty dictionary if loader is None
        # is NECESSARY
        if loader is None:
            return {}

        # also important to change the model to evaluation mode
        model.eval()

        # here we will create a list to collect all predictions and labels for later computation
        # because precision, recall and f1 cannot be computed in mini-batch manner
        labels = []
        predictions = []

        # this step is needed to use the test mode (perform only small number of steps for testing)  
        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        with torch.no_grad():
            for minibatch_idx, (inputs, targets) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                inputs = inputs.to(device)
                targets = targets.long().flatten().tolist()

                pred = model(inputs).argmax(dim=-1).cpu().long().tolist()
                labels.extend(targets)
                predictions.extend(pred)

        performance = {'acc': accuracy_score(labels, predictions),
                       'precision': precision_score(labels, predictions, average='macro'),
                       'recall': recall_score(labels, predictions, average='macro'),
                       'f1': f1_score(labels, predictions, average='macro')}

        return performance
                 

class AllCNN(nn.Module):
    def __init__(self, n_class):
        super(AllCNN, self).__init__()

        self.block1 = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.block2 = nn.ModuleList([
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.block3 = nn.ModuleList([
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        ])

        self.classifier = nn.ModuleList([
            nn.Conv2d(in_channels=192, out_channels=n_class, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_class),
            nn.ReLU(),
        ])

    def forward(self, x):
        for layer in self.block1:
            x = layer(x)
        for layer in self.block2:
            x = layer(x)
        for layer in self.block3:
            x = layer(x)
        for layer in self.classifier:
            x = layer(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return x

    def initialize(self,):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def get_parameters(self,):
        bn_params = list(self.block1[1].parameters()) +\
            list(self.block1[4].parameters()) +\
            list(self.block1[7].parameters()) +\
            list(self.block2[1].parameters()) +\
            list(self.block2[4].parameters()) +\
            list(self.block2[7].parameters()) +\
            list(self.block3[1].parameters()) +\
            list(self.block3[4].parameters()) +\
            list(self.classifier[1].parameters())

        other_params = list(self.block1[0].parameters()) +\
            list(self.block1[3].parameters()) +\
            list(self.block1[6].parameters()) +\
            list(self.block2[0].parameters()) +\
            list(self.block2[3].parameters()) +\
            list(self.block2[6].parameters()) +\
            list(self.block3[0].parameters()) +\
            list(self.block3[3].parameters()) +\
            list(self.classifier[0].parameters())

        return bn_params, other_params

def get_dataset(dataset):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='.', train=True,
                                                download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR10(root='.', train=False,
                                               download=True, transform=test_transform)

        n_class = 10
    else:
        trainset = torchvision.datasets.CIFAR100(root='.', train=True,
                                                download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR100(root='.', train=False,
                                               download=True, transform=test_transform)
        n_class = 100

    train_loader = DataLoader(trainset, batch_size=64, pin_memory=True, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, pin_memory=True, shuffle=False)

    return train_loader, test_loader, n_class 


def main(argv):

    dataset = 'cifar10'
    try:
      opts, args = getopt.getopt(argv,"h", ['dataset=', ])

    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--dataset':
            dataset = arg


    assert dataset in ['cifar10', 'cifar100']

    train_loader, test_loader, n_class = get_dataset(dataset)
    model = AllCNN(n_class)
    device = torch.device('cuda')
    trainer = Trainer(n_epoch=200)
    performance = trainer.fit(model, train_loader, None, test_loader, device)

if __name__ == "__main__":
    main(sys.argv[1:])

