import argparse
import copy
import pickle as pkl
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# from tensorboardX import SummaryWriter

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from metrics.metrics import Metrics
import pdb
from models.fastdepth import weights_init
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args):

    global results
    # config
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 4 if cuda else 0

    # Hyper-parameters
    batch_size = args.batch_size
    num_prunes = args.n_prune
    prune_amount = args.prune_amount
    epochs = args.n_epoch
    learning_rate = args.lr
    weight_decay = args.weight_decay
    warm_up_k = args.n_warm_up
    warm_up_iter = 0
    weight_init_type = args.weight_init_type
    momentum = args.momentum

    # Create datasets and loaders
    if args.dataset == "MNIST":
        from datasets.mnist import MNIST

        training_dataset = MNIST(
            root="./data",
            transform=torchvision.transforms.Compose(
                [transforms.Grayscale(3), torchvision.transforms.ToTensor()]
            ),
            download=True,
        )
        validation_dataset = MNIST(
            root="./data",
            train=False,
            transform=torchvision.transforms.Compose(
                [transforms.Grayscale(3), torchvision.transforms.ToTensor()]
            ),
            download=True,
        )
    elif args.dataset == "nyudepthv2":
        from datasets.nyu import NYUDataset

        training_dataset = NYUDataset(root="./data/nyudepthv2/train", split="train")
        #training_dataset = NYUDataset(root="./data/nyudepthv2/val", split="train")
        validation_dataset = NYUDataset(root="./data/nyudepthv2/val", split="val")
    else:
        raise NotImplementedError("Invalid dataset input")

    training_loader_args = (
        dict(
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        if cuda
        else dict(shuffle=True, batch_size=batch_size)
    )
    training_loader = DataLoader(training_dataset, **training_loader_args)

    validation_loader_args = (
        dict(
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        if cuda
        else dict(shuffle=False, batch_size=batch_size)
    )
    validation_loader = DataLoader(validation_dataset, **validation_loader_args)

    # intialize model and training parameters
    if args.model == "resnet18":
        from models.resnet18 import resnet18

        net = resnet18(pretrained=False)
    elif args.model == "FastDepth":
        from models.fastdepth import MobileNetSkipAdd

        # Unsure of output size
        net = MobileNetSkipAdd(output_size=224, pretrained_encoder=True)
    else:
        raise NotImplementedError("Invalid model input")

    global net2
    net2 = copy.deepcopy(net)
    # if (weight_init_flag == "originial_first")
    initial_state = copy.deepcopy(net.state_dict())

    if args.model == "resnet18":
        criterion = nn.CrossEntropyLoss()
    elif args.model == "FastDepth":
        criterion = nn.L1Loss()
    else:
        raise NotImplementedError("No loss function defined for that model")

    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=2
    )
    net.to(device)



    global_sparsity = 0
    results = {}

    run_id = str(int(time.time()))

    for prune_cycle in range(num_prunes):
        writer = SummaryWriter(
            "./runs/" + run_id + "/Lottery_prune_{0}".format(global_sparsity)
        )
        for epoch in range(epochs):
            print("Epoch: " + str(epoch) + " (" + str(prune_cycle) + " prunings)")

            training_metrics, warm_up_iter = train_epoch(
                net,
                device,
                training_loader,
                criterion,
                optimizer,
                warm_up_iter,
                warm_up_k,
                learning_rate,
                writer,
                epoch,
            )
            validation_metrics = validate_epoch(
                net, device, validation_loader, criterion, scheduler, writer, epoch
            )
            if (args.vanilla_train):
                continue
            for metric, value in training_metrics.items():
                if prune_cycle == 0 and epoch == 0:
                    results["train_" + metric] = {}
                if epoch == 0:
                    results["train_" + metric]["prune_{0}".format(global_sparsity)] = []

                print("Training " + metric + ": ", value)
                results["train_" + metric]["prune_{0}".format(global_sparsity)].append(
                    value
                )

            for metric, value in validation_metrics.items():
                if prune_cycle == 0 and epoch == 0:
                    results["validation_" + metric] = {}
                if epoch == 0:
                    results["validation_" + metric][
                        "prune_{0}".format(global_sparsity)
                    ] = []

                print("Validation " + metric + ": ", value)
                results["validation_" + metric][
                    "prune_{0}".format(global_sparsity)
                ].append(value)

            print("=" * 50)
        if (args.vanilla_train):
            continue
        writer.close()
        pkl.dump(
            results, open("results/results_prune_{0}.p".format(global_sparsity), "wb")
        )

        if weight_init_type == "carry_previous":
            initial_state = copy.deepcopy(net.state_dict())
        else:
            # this will keep the original weights or use xavier intialization
            pass

        global_sparsity = my_prune(net, prune_amount, initial_state, weight_init_type, args.model)

    if (not(args.vanilla_train)):
        pkl.dump(results, open("results/results_final.pkl", "wb"))
    else:
        torch.save(net.state_dict(), 'model.pth')


def train_epoch(
    model, device, train_loader, criterion, optimizer, k, warm_up, lr, writer, epoch
):
    # training phase
    print("Training Progress:")
    metrics = Metrics(args.dataset, train=True)
    model.train()

    for batch_idx, (batch, labels) in enumerate(tqdm(train_loader)):
        iteration = epoch * len(train_loader) + batch_idx
        optimizer.zero_grad()
        batch = batch.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        outputs = model(batch)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # warm up
        if k <= warm_up:
            k = learning_rate_scheduler(optimizer, k, warm_up, lr)

        # Batch metrics
        metrics.update_metrics(outputs, labels, loss)
        if iteration % 10 == 0:
            metrics.write_to_tensorboard(writer, iteration)

    # Epoch metrics
    final_metrics = metrics.get_epoch_metrics()

    return (final_metrics, k)


def validate_epoch(
    model, device, validation_loader, criterion, scheduler, writer, epoch
):
    with torch.no_grad():
        # validation phase
        print("Validation Progress:")
        metrics = Metrics(args.dataset, train=False)
        model.eval()

        for batch_idx, (batch, labels) in enumerate(tqdm(validation_loader)):
            batch = batch.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            outputs = model(batch)
            loss = criterion(outputs, labels)

            # Batch metrics
            metrics.update_metrics(outputs, labels, loss)

        # Epoch metrics
        final_metrics = metrics.get_epoch_metrics()
        metrics.write_to_tensorboard(writer, epoch)
        scheduler.step(final_metrics["Loss"])

    return final_metrics


def learning_rate_scheduler(my_optim, k, warm_up, lr):
    for param_group in my_optim.param_groups:
        param_group["lr"] = (k / warm_up) * lr
    return k + 1


def my_prune(net, prune_amount, initial_state, weight_init_type, model_type):

    # TODO: separate prune into custom lib
    # TODO: function naming
    # extract weights
    parameters_to_prune = []
    if model_type == "resnet18":
            
        
        parameters_to_prune.append((net._modules["conv1"], "weight"))

        parameters_to_prune.append((net._modules["layer1"][0]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer1"][0]._modules["conv2"], "weight"))
        parameters_to_prune.append((net._modules["layer1"][1]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer1"][1]._modules["conv2"], "weight"))

        parameters_to_prune.append((net._modules["layer2"][0]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer2"][0]._modules["conv2"], "weight"))
        parameters_to_prune.append(
            (net._modules["layer2"][0]._modules["downsample"][0], "weight")
        )
        parameters_to_prune.append((net._modules["layer2"][1]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer2"][1]._modules["conv2"], "weight"))

        parameters_to_prune.append((net._modules["layer3"][0]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer3"][0]._modules["conv2"], "weight"))
        parameters_to_prune.append(
            (net._modules["layer3"][0]._modules["downsample"][0], "weight")
        )
        parameters_to_prune.append((net._modules["layer3"][1]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer3"][1]._modules["conv2"], "weight"))

        parameters_to_prune.append((net._modules["layer4"][0]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer4"][0]._modules["conv2"], "weight"))
        parameters_to_prune.append(
            (net._modules["layer4"][0]._modules["downsample"][0], "weight")
        )
        parameters_to_prune.append((net._modules["layer4"][1]._modules["conv1"], "weight"))
        parameters_to_prune.append((net._modules["layer4"][1]._modules["conv2"], "weight"))

        parameters_to_prune.append((net._modules["fc"], "weight"))
        # print(parameters_to_prune)

        # global prune
        # TODO: change pruning method to L2-norm
        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount,
        )

        # print global sparsity after pruning
        print(
            "Global sparsity: {:.2f}%".format(
                100.0
                * float(
                    torch.sum(net.conv1.weight == 0)
                    + torch.sum(net._modules["layer1"][0]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer1"][0]._modules["conv2"].weight == 0)
                    + torch.sum(net._modules["layer1"][1]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer1"][1]._modules["conv2"].weight == 0)
                    + torch.sum(net._modules["layer2"][0]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer2"][0]._modules["conv2"].weight == 0)
                    + torch.sum(
                        net._modules["layer2"][0]._modules["downsample"][0].weight == 0
                    )
                    + torch.sum(net._modules["layer2"][1]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer2"][1]._modules["conv2"].weight == 0)
                    + torch.sum(net._modules["layer3"][0]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer3"][0]._modules["conv2"].weight == 0)
                    + torch.sum(
                        net._modules["layer3"][0]._modules["downsample"][0].weight == 0
                    )
                    + torch.sum(net._modules["layer3"][1]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer3"][1]._modules["conv2"].weight == 0)
                    + torch.sum(net._modules["layer4"][0]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer4"][0]._modules["conv2"].weight == 0)
                    + torch.sum(
                        net._modules["layer4"][0]._modules["downsample"][0].weight == 0
                    )
                    + torch.sum(net._modules["layer4"][1]._modules["conv1"].weight == 0)
                    + torch.sum(net._modules["layer4"][1]._modules["conv2"].weight == 0)
                    + torch.sum(net.fc.weight == 0)
                )
                / float(
                    net.conv1.weight.nelement()
                    + net._modules["layer1"][0]._modules["conv1"].weight.nelement()
                    + net._modules["layer1"][0]._modules["conv2"].weight.nelement()
                    + net._modules["layer1"][1]._modules["conv1"].weight.nelement()
                    + net._modules["layer1"][1]._modules["conv2"].weight.nelement()
                    + net._modules["layer2"][0]._modules["conv1"].weight.nelement()
                    + net._modules["layer2"][0]._modules["conv2"].weight.nelement()
                    + net._modules["layer2"][0]._modules["downsample"][0].weight.nelement()
                    + net._modules["layer2"][1]._modules["conv1"].weight.nelement()
                    + net._modules["layer2"][1]._modules["conv2"].weight.nelement()
                    + net._modules["layer3"][0]._modules["conv1"].weight.nelement()
                    + net._modules["layer3"][0]._modules["conv2"].weight.nelement()
                    + net._modules["layer3"][0]._modules["downsample"][0].weight.nelement()
                    + net._modules["layer3"][1]._modules["conv1"].weight.nelement()
                    + net._modules["layer3"][1]._modules["conv2"].weight.nelement()
                    + net._modules["layer4"][0]._modules["conv1"].weight.nelement()
                    + net._modules["layer4"][0]._modules["conv2"].weight.nelement()
                    + net._modules["layer4"][0]._modules["downsample"][0].weight.nelement()
                    + net._modules["layer4"][1]._modules["conv1"].weight.nelement()
                    + net._modules["layer4"][1]._modules["conv2"].weight.nelement()
                    + net.fc.weight.nelement()
                )
            )
        )

        if weight_init_type == "carry_initial" or weight_init_type == "carry_previous":
            # load the initial weight
            pass
        elif weight_init_type == "xavier_init":
            net2.apply(xavier_init_weights)
            initial_state = copy.deepcopy(net2.state_dict())
        else:
            raise ("You have not mentioned a weight initialization.")

        for name, param in initial_state.items():
            if name not in net.state_dict():
                net.state_dict()[name + "_orig"].copy_(param)
            else:
                net.state_dict()[name].copy_(param)

        global_sparsity = (
            100.0
            * float(
                torch.sum(net.conv1.weight == 0)
                + torch.sum(net._modules["layer1"][0]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer1"][0]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer1"][1]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer1"][1]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer2"][0]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer2"][0]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer2"][0]._modules["downsample"][0].weight == 0)
                + torch.sum(net._modules["layer2"][1]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer2"][1]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer3"][0]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer3"][0]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer3"][0]._modules["downsample"][0].weight == 0)
                + torch.sum(net._modules["layer3"][1]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer3"][1]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer4"][0]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer4"][0]._modules["conv2"].weight == 0)
                + torch.sum(net._modules["layer4"][0]._modules["downsample"][0].weight == 0)
                + torch.sum(net._modules["layer4"][1]._modules["conv1"].weight == 0)
                + torch.sum(net._modules["layer4"][1]._modules["conv2"].weight == 0)
                + torch.sum(net.fc.weight == 0)
            )
            / float(
                net.conv1.weight.nelement()
                + net._modules["layer1"][0]._modules["conv1"].weight.nelement()
                + net._modules["layer1"][0]._modules["conv2"].weight.nelement()
                + net._modules["layer1"][1]._modules["conv1"].weight.nelement()
                + net._modules["layer1"][1]._modules["conv2"].weight.nelement()
                + net._modules["layer2"][0]._modules["conv1"].weight.nelement()
                + net._modules["layer2"][0]._modules["conv2"].weight.nelement()
                + net._modules["layer2"][0]._modules["downsample"][0].weight.nelement()
                + net._modules["layer2"][1]._modules["conv1"].weight.nelement()
                + net._modules["layer2"][1]._modules["conv2"].weight.nelement()
                + net._modules["layer3"][0]._modules["conv1"].weight.nelement()
                + net._modules["layer3"][0]._modules["conv2"].weight.nelement()
                + net._modules["layer3"][0]._modules["downsample"][0].weight.nelement()
                + net._modules["layer3"][1]._modules["conv1"].weight.nelement()
                + net._modules["layer3"][1]._modules["conv2"].weight.nelement()
                + net._modules["layer4"][0]._modules["conv1"].weight.nelement()
                + net._modules["layer4"][0]._modules["conv2"].weight.nelement()
                + net._modules["layer4"][0]._modules["downsample"][0].weight.nelement()
                + net._modules["layer4"][1]._modules["conv1"].weight.nelement()
                + net._modules["layer4"][1]._modules["conv2"].weight.nelement()
                + net.fc.weight.nelement()
            )
        )
        print("Global sparsity after loading: {:.2f}%".format(global_sparsity))

        return int(global_sparsity)

    elif model_type == "FastDepth":

        for i in range(14):
            layer = getattr(net, "conv{}".format(i))
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))

        for i in range(1, 7):
            layer = getattr(net, "decode_conv{}".format(i))
            for name, module in layer.named_modules(): 
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount,
        )

        pruned_weight = 0.0
        total_weight = 0.0

        for i in range(14):
            layer = getattr(net, "conv{}".format(i))
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    pruned_weight += torch.sum(module.weight == 0)
                    total_weight += module.weight.nelement()
        for i in range(1, 7):
            layer = getattr(net, "decode_conv{}".format(i))
            for name, module in layer.named_modules(): 
                if isinstance(module, torch.nn.Conv2d):
                    pruned_weight += torch.sum(module.weight == 0)
                    total_weight += module.weight.nelement()
        print("Global sparsity: {:.2f}%".format(100.0 * float(pruned_weight)/ float(total_weight)))
        global_sparsity = 100.0 * float(pruned_weight)/ float(total_weight)

        if weight_init_type == "carry_initial" or weight_init_type == "carry_previous":
            # load the initial weight
            pass
        elif weight_init_type == "xavier_init":
            net2.apply(weights_init)
            initial_state = copy.deepcopy(net2.state_dict())
        else:
            raise ("You have not mentioned a weight initialization.")

        for name, param in initial_state.items():
            if name not in net.state_dict():
                net.state_dict()[name + "_orig"].copy_(param)
            else:
                net.state_dict()[name].copy_(param)

        return int(global_sparsity)



def xavier_init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # added argument for parser
    parser.add_argument("--model", default="resnet18", help="")
    parser.add_argument("--dataset", default="MNIST", help="")
    parser.add_argument("--n_epoch", type=int, default=10, help="")
    parser.add_argument("--n_prune", type=int, default=10, help="")
    parser.add_argument("--n_warm_up", type=int, default=20000, help="")
    parser.add_argument("--prune_amount", type=float, default=0.1, help="")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--vanilla_train", type=bool, default=False)

    # xavier_init
    # carry_initial(carry over the first weights)
    # carry_previous weights
    parser.add_argument("--weight_init_type", type=str, default="carry_initial")

    args = parser.parse_args()
    main(args)
