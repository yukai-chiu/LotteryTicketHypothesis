import argparse
import copy
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune


#Custom libs
import models
import datasets

def main(args):
    #config
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    batch_size = 256 if cuda else 64
    num_workers = 4 if cuda else 0 

    #Hyper-parameters
    #TODO: interate the args
    num_prunes = args.n_prune
    prune_amount = 0.1
    epochs = args.n_epoch
    learning_rate = 1e-3
    weightDecay = 1e-4

    #Transforms
    train_transforms = [
        transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ]
    validation_transforms = [
        transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ]

    #Create datasets and loaders
    #TODO: if statement to handle differnet datasets
    training_dataset = datasets.MNIST(root='./data', transform=torchvision.transforms.Compose(train_transforms), download=True)
    validation_dataset = datasets.MNIST(root='./data', train= False, transform=torchvision.transforms.Compose(validation_transforms), download=True)

    training_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)\
    if cuda else dict(shuffle=True, batch_size=batch_size)
    training_loader = DataLoader(training_dataset, **training_loader_args)

    validation_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)\
    if cuda else dict(shuffle=False, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, **validation_loader_args)

    #intialize model and training parameters
    #TODO: if statement to handle different models
    if args.model == 'resnet18':
        net = models.resnet18(pretrained=False)
  


    initial_state = copy.deepcopy(net.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    net.to(device)

    for prune_cycle in range(num_prunes):
        for i in range(epochs):
            print('Epoch: '+ str(i)+' ('+str(prune_cycle)+' prunings)')

            train_loss, train_acc = train_epoch(net, device, training_loader, criterion, optimizer)
            validation_loss, validation_acc = validatate_epoch(net, device, validation_loader, criterion, scheduler)

            print('Training Accuracy: ', train_acc, "%")
            print('Training Loss: ', train_loss)
            print('Validation Accuracy: ', validation_acc, "%")
            print('Validataion Loss: ', validation_loss)
            print('='*50)

        #TODO: implement a global pruning function
        #order of these two lines depends on how we implement prune
        #TODO: implement loading initial weights
        #net.load_state_dict(initial_state)
        my_prune(net, prune_amount)


def train_epoch(model, device, train_loader, criterion, optimizer):
    
    batch_losses_training = []
    
    #training phase
    print('Training Progress:')
    total_predictions_training = 0
    correct_predictions_training = 0
    model.train()
    
    for batch, labels in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        outputs = model(batch)
        
        _, predicted = torch.max(outputs.data, 1)
        total_predictions_training += labels.size(0)
        correct_predictions_training += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        batch_losses_training.append(loss.item()/np.size(batch, axis = 0))
        
        loss.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        del batch
        del labels
        del loss
        
    training_acc = (correct_predictions_training/total_predictions_training)*100.0
    return (np.mean(batch_losses_training), training_acc)

def validatate_epoch(model, device, validation_loader, criterion, scheduler):
    with torch.no_grad():
        batch_losses_validation = []

        #validation phase
        print('Validation Progress:')
        total_predictions_validation = 0
        correct_predictions_validation = 0
        model.eval()

        for batch, labels in tqdm(validation_loader):
            batch = batch.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            outputs = model(batch)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions_validation += labels.size(0)
            correct_predictions_validation += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            batch_losses_validation.append(loss.item()/np.size(batch, axis = 0))

            torch.cuda.empty_cache()
            del batch
            del labels
            del loss

        scheduler.step(np.mean(batch_losses_validation))

        validation_acc = (correct_predictions_validation/total_predictions_validation)*100.0
    return(np.mean(batch_losses_validation), validation_acc)

def my_prune(net, prune_amount):

    #TODO: separate prune into custom lib
    #TODO: function naming 
    #extract weights
    parameters_to_prune = []
    parameters_to_prune.append((net._modules['conv1'],'weight'))

    parameters_to_prune.append((net._modules['layer1'][0]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer1'][0]._modules['conv2'],'weight'))
    parameters_to_prune.append((net._modules['layer1'][1]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer1'][1]._modules['conv2'],'weight'))

    parameters_to_prune.append((net._modules['layer2'][0]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer2'][0]._modules['conv2'],'weight'))
    parameters_to_prune.append((net._modules['layer2'][0]._modules['downsample'][0],'weight'))
    parameters_to_prune.append((net._modules['layer2'][1]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer2'][1]._modules['conv2'],'weight'))

    parameters_to_prune.append((net._modules['layer3'][0]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer3'][0]._modules['conv2'],'weight'))
    parameters_to_prune.append((net._modules['layer3'][0]._modules['downsample'][0],'weight'))
    parameters_to_prune.append((net._modules['layer3'][1]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer3'][1]._modules['conv2'],'weight'))

    parameters_to_prune.append((net._modules['layer4'][0]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer4'][0]._modules['conv2'],'weight'))
    parameters_to_prune.append((net._modules['layer4'][0]._modules['downsample'][0],'weight'))
    parameters_to_prune.append((net._modules['layer4'][1]._modules['conv1'],'weight'))
    parameters_to_prune.append((net._modules['layer4'][1]._modules['conv2'],'weight'))

    parameters_to_prune.append((net._modules['fc'],'weight'))
    #print(parameters_to_prune)

    #global prune
    #TODO: change pruning method to L2-norm
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=prune_amount,
    )

    #print global sparsity after pruning
    print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(net.conv1.weight == 0)
            + torch.sum(net._modules['layer1'][0]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer1'][0]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer1'][1]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer1'][1]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer2'][0]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer2'][0]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer2'][0]._modules['downsample'][0].weight == 0)
            + torch.sum(net._modules['layer2'][1]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer2'][1]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer3'][0]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer3'][0]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer3'][0]._modules['downsample'][0].weight == 0)
            + torch.sum(net._modules['layer3'][1]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer3'][1]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer4'][0]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer4'][0]._modules['conv2'].weight == 0)
            + torch.sum(net._modules['layer4'][0]._modules['downsample'][0].weight == 0)
            + torch.sum(net._modules['layer4'][1]._modules['conv1'].weight == 0)
            + torch.sum(net._modules['layer4'][1]._modules['conv2'].weight == 0)
            + torch.sum(net.fc.weight == 0)
        )
        / float(
            net.conv1.weight.nelement()
            + net._modules['layer1'][0]._modules['conv1'].weight.nelement()
            + net._modules['layer1'][0]._modules['conv2'].weight.nelement()
            + net._modules['layer1'][1]._modules['conv1'].weight.nelement()
            + net._modules['layer1'][1]._modules['conv2'].weight.nelement()
            + net._modules['layer2'][0]._modules['conv1'].weight.nelement()
            + net._modules['layer2'][0]._modules['conv2'].weight.nelement()
            + net._modules['layer2'][0]._modules['downsample'][0].weight.nelement()
            + net._modules['layer2'][1]._modules['conv1'].weight.nelement()
            + net._modules['layer2'][1]._modules['conv2'].weight.nelement()
            + net._modules['layer3'][0]._modules['conv1'].weight.nelement()
            + net._modules['layer3'][0]._modules['conv2'].weight.nelement()
            + net._modules['layer3'][0]._modules['downsample'][0].weight.nelement()
            + net._modules['layer3'][1]._modules['conv1'].weight.nelement()
            + net._modules['layer3'][1]._modules['conv2'].weight.nelement()
            + net._modules['layer4'][0]._modules['conv1'].weight.nelement()
            + net._modules['layer4'][0]._modules['conv2'].weight.nelement()
            + net._modules['layer4'][0]._modules['downsample'][0].weight.nelement()
            + net._modules['layer4'][1]._modules['conv1'].weight.nelement()
            + net._modules['layer4'][1]._modules['conv2'].weight.nelement()
             + net.fc.weight.nelement()
            )
        )
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #added argument for parser
    parser.add_argument('--model', default='resnet18', help='')
    parser.add_argument('--n_epoch', type=int, default=1,help='')
    parser.add_argument('--n_prune', type=int, default=5,help='')
    

    args = parser.parse_args()
    main(args)
