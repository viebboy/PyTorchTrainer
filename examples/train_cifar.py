import PyTorchTrainer as Trainer
import os, sys, getopt, pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

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
    trainer = Trainer.ClassifierTrainer(n_epoch=200)
    performance = trainer.fit(model, train_loader, None, test_loader, device)

if __name__ == "__main__":
    main(sys.argv[1:])

