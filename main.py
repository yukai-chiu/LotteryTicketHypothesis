import argparse
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Custom libs
import models
import datasets

def main(args):
    #config
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    batch_size = 256 if cuda else 64
    num_workers = 4 if cuda else 0 

    #Create datasets and loaders
    training_dataset = datasets.MNIST(root='./data', transform=None, download=True)
    validation_dataset = datasets.MNIST(root='./data', train= False, transform=None, download=True)

    training_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)\
    if cuda else dict(shuffle=True, batch_size=batch_size)
    training_loader = DataLoader(training_dataset, **training_loader_args)

    validation_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)\
    if cuda else dict(shuffle=False, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, **validation_loader_args)

    #intialize modeland training parameters
    net = models.resnet18(pretrained=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=0)
    net.to(device)

    for train_cycle in range(num_prunes)
        for i in range(epochs):
            train_epoch(net, train_loader, criterion, optimizer)
            validatate_epoch(net, validation_loader, criterion, scheduler)
    prune(net)


def train_epoch(net, train_loader, criterion, optimizer):

def validatate_epoch(net, validation_loader, criterion, scheduler):

def prune(net):


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
