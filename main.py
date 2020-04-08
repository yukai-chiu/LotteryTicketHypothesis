import argparse
import copy
import pickle as pkl
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
from tensorboardX import SummaryWriter 

# from torch.utils.tensorboard import SummaryWriter


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
    prune_amount = args.prune_amount
    epochs = args.n_epoch
    learning_rate = args.lr
    weightDecay = args.weight_decay
    warm_up_k = args.n_warm_up
    warm_up_iter = 0
    weight_init_type = args.weight_init_type

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

    training_loader_args = dict(shuffle=True, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                pin_memory=True) if cuda else dict(shuffle=True, 
                                                                    batch_size=batch_size)
    training_loader = DataLoader(training_dataset, **training_loader_args)

    validation_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)\
    if cuda else dict(shuffle=False, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, **validation_loader_args)

    #intialize model and training parameters
    #TODO: if statement to handle different models
    if args.model == 'resnet18':
        net = models.resnet18(pretrained=False)
  

    # if (weight_init_flag == "originial_first")
    initial_state = copy.deepcopy(net.state_dict())



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    net.to(device)

    writer = SummaryWriter()
    
    global_sparsity = 100
    results = {}
    results['train_accuracy'] = {}
    results['test_accuracy'] = {}
    results['train_loss'] = {}
    results['test_loss'] = {}

    for prune_cycle in range(num_prunes):
        results['train_accuracy']['prune_{0}'.format(global_sparsity)] = []
        results['test_accuracy']['prune_{0}'.format(global_sparsity)] = []
        results['train_loss']['prune_{0}'.format(global_sparsity)] = []
        results['test_loss']['prune_{0}'.format(global_sparsity)] = []
        for epoch in range(epochs):
            print('Epoch: '+ str(epoch)+' ('+str(prune_cycle)+' prunings)')

            train_loss, train_acc, warm_up_iter = train_epoch(net, device, training_loader, criterion, optimizer, warm_up_iter, warm_up_k, learning_rate)
            validation_loss, validation_acc = validate_epoch(net, device, validation_loader, criterion, scheduler)

            writer.add_scalar('train/accuracy/prune_{0}'.format(global_sparsity), train_acc, epoch)
            print('Training Accuracy: ', train_acc, "%")
            results['train_accuracy']['prune_{0}'.format(global_sparsity)].append(train_acc)

            writer.add_scalar('train/loss/prune_{0}'.format(global_sparsity), train_loss, epoch)
            print('Training Loss: ', train_loss)
            results['train_loss']['prune_{0}'.format(global_sparsity)].append(train_loss)
            
            writer.add_scalar('validation/accuracy/prune_{0}'.format(global_sparsity), validation_acc, epoch)
            print('Validation Accuracy: ', validation_acc, "%")
            results['test_accuracy']['prune_{0}'.format(global_sparsity)].append(validation_acc)
            
            writer.add_scalar('validation/loss/prune_{0}'.format(global_sparsity), validation_loss, epoch)
            print('Validataion Loss: ', validation_loss)
            results['test_loss']['prune_{0}'.format(global_sparsity)].append(validation_loss)

            print('='*50)

        pkl.dump(results, open("results_prune_{0}.p".format(global_sparsity), "wb" ))
            

        #TODO: implement a global pruning function
        #order of these two lines depends on how we implement prune
        #TODO: implement loading initial weights
        #net.load_state_dict(initial_state)
        if (reinit_type == "carry_weights"):
            initial_state = copy.deepcopy(net.state_dict())
        else:
            pass # this will keep the original weights
        

        net, global_sparsity = my_prune(net, prune_amount,initial_state, weight_init_type)
        

    pkl.dump(results, open("results_final.p", "wb"))
    writer.close()



def train_epoch(model, device, train_loader, criterion, optimizer, k, warm_up, lr):
    
    batch_losses_training = []
    
    #training phase
    print('Training Progress:')
    total_predictions_training = 0
    correct_predictions_training = 0
    model.train()
    
    for batch_idx, (batch, labels) in enumerate(tqdm(train_loader)):
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
        #warn up
        if k <= warm_up:
            k = learning_rate_scheduler(optimizer, k, warm_up, lr)

        torch.cuda.empty_cache()
        del batch
        del labels
        del loss

    training_acc = (correct_predictions_training/total_predictions_training)*100.0
    return (np.mean(batch_losses_training), training_acc, k)

def validate_epoch(model, device, validation_loader, criterion, scheduler):
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


def learning_rate_scheduler(my_optim, k, warm_up, lr):
    for param_group in my_optim.param_groups:
        param_group['lr'] = (k / warm_up) * lr
    return k + 1

def my_prune(net,prune_amount,initial_state, weight_init_type):

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


    if (weight_init_type == "carry_initial" or weight_init_type == "carry_previous"):
    #load the initial weight
        for name, param in initial_state.items():
            if name not in net.state_dict():          
                net.state_dict()[name+ '_orig'].copy_(param) 
            else:
                net.state_dict()[name].copy_(param)

    elif (weight_init_type == "xavier_init"):
        net.apply(xavier_init_weights)

    else:
        raise("You have not mentioned a weight initialization.")


    global_sparsity = 100. * float(
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
        ) / float(
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
    print( "Global sparsity after loading: {:.2f}%".format(global_sparsity))
    return net, global_sparsity 


def xavier_init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #added argument for parser
    parser.add_argument('--model', default='resnet18', help='')
    parser.add_argument('--n_epoch', type=int, default=10,help='')
    parser.add_argument('--n_prune', type=int, default=5,help='')
    parser.add_argument('--n_warm_up', type=int, default=20000,help='')
    parser.add_argument('--prune_amount', type=float, default=0.1, help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # xavier_init #carry_initial(carry over the first weights) #carry previous weights
    parser.add_argument('--weight_init_type', type = 'string', default = 'xavier_init')

    args = parser.parse_args()
    main(args)
