"""
In this example, we will customize the trainer to perform knowledge distillation between 2 models 
by subclassing the ClassifierTrainer 
"""

from PyTorchTrainer import ClassifierTrainer, get_cosine_lr_scheduler
import os, sys, getopt, pickle
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


"""
Here we will subclass ClassifierTrainer to define a new trainer that performs knowledge distillation
The idea is that we will
- modify the init function to receive also the teacher model and the KD loss hyperparameter like alpha and temperature
- modify the update_loop 
- we also modify the eval function to compute the KD loss

"""

# Here we define the KD loss
def KD_loss(outputs, labels, teacher_outputs, alpha, temperature):
    T = temperature
    loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / temperature, dim=1),
                                                     F.softmax(teacher_outputs / temperature, dim=1)) * (alpha * T * T) +\
        F.cross_entropy(outputs, labels) * (1. - alpha)

    return loss


class KDTrainer(ClassifierTrainer):
    def __init__(self,
                 teacher_network, # add teacher model as the first parameter
                 n_epoch,
                 epoch_idx=0,
                 lr_scheduler=get_cosine_lr_scheduler(1e-3, 1e-5),
                 optimizer='adam',
                 weight_decay=1e-4,
                 temp_dir='',
                 checkpoint_freq=1,
                 print_freq=1,
                 use_progress_bar=True,
                 test_mode=False,
                 alpha=0.5, # hyperparameter in KD loss
                 temperature=4, # hyperparameter in KD loss
                 ): 

        # We call the init of the ClassifierTrainer to perform necessary initialization 
        super(KDTrainer, self).__init__(n_epoch,
                                      epoch_idx,
                                      lr_scheduler,
                                      optimizer,
                                      weight_decay,
                                      temp_dir,
                                      checkpoint_freq,
                                      print_freq,
                                      use_progress_bar,
                                      test_mode)

        self.teacher_network = teacher_network
        self.alpha = alpha
        self.temperature = temperature

    def update_loop(self, model, loader, optimizer, device):
        """
        We redefine the update loop to backpropagate based on KD loss
        """

        # move teacher to device, also enable evaluation mode
        self.teacher_network.to(device)
        self.teacher_network.eval()

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        minibatch_idx = 0

        if self.use_progress_bar:
            loader = tqdm(loader, desc='#Epoch {}/{}: '.format(self.epoch_idx + 1, self.n_epoch), ncols=80, ascii=True)
        else:
            loader = loader

        for inputs, targets in loader:
            optimizer.zero_grad()
            self.update_lr(optimizer)

            inputs = inputs.to(device)
            targets = targets.to(device).long().flatten()

            student_predictions = model(inputs)
            teacher_predictions = self.teacher_network(inputs)
            loss = KD_loss(student_predictions,
                           targets,
                           teacher_predictions,
                           self.alpha,
                           self.temperature)
            loss.backward()
            optimizer.step()

            minibatch_idx += 1

            if minibatch_idx > total_minibatch:
                break
                 

    def eval(self, model, loader, device):
        """
        modify eval function to compute the KD loss
        """
        
        if loader is None:
            return {}

        # move teacher to device
        self.teacher_network.to(device)

        model.eval()

        L = torch.nn.CrossEntropyLoss()
        n_correct = 0
        n_sample = 0
        loss = 0
        kd_loss = 0

        if self.test_mode:
            total_minibatch = min(self.n_test_minibatch, len(loader))
        else:
            total_minibatch = len(loader)

        with torch.no_grad():
            for minibatch_idx, (inputs, targets) in enumerate(loader):
                if minibatch_idx == total_minibatch:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device).long().flatten()

                predictions = model(inputs)
                teacher_predictions = self.teacher_network(inputs)
                n_sample += inputs.size(0)
                loss += L(predictions, targets).item()
                n_correct += (predictions.argmax(dim=-1) == targets).sum().item()
                kd_loss += KD_loss(predictions,
                                   targets,
                                   teacher_predictions,
                                   self.alpha,
                                   self.temperature).item()

        performance = {'cross_entropy': loss / n_sample,
                   'acc': n_correct / n_sample,
                   'kd_loss': kd_loss / n_sample}

        return performance


"""
Teacher has more convolution layers than students
"""
class Teacher(nn.Module):
    def __init__(self, n_class):
        super(Teacher, self).__init__()

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

class Student(nn.Module):
    def __init__(self, n_class):
        super(Student, self).__init__()

        self.block1 = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.block2 = nn.ModuleList([
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.block3 = nn.ModuleList([
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
    # create teacher and student
    teacher_network = Teacher(n_class)
    student_network = Student(n_class)
    device = torch.device('cuda')

    # we use the original ClassifierTrainer to train the teacher
    print('********************************')
    print('\t Train Teacher')
    print('********************************')
    teacher_trainer = ClassifierTrainer(n_epoch=1)
    teacher_performance = teacher_trainer.fit(teacher_network, train_loader, None, test_loader, device)

    # then we create a KDTrainer to train student network
    print('********************************')
    print('\t Train Student')
    print('********************************')
    kd_trainer = KDTrainer(teacher_network, n_epoch=2)
    student_performance = kd_trainer.fit(student_network, train_loader, None, test_loader, device)

if __name__ == "__main__":
    main(sys.argv[1:])

