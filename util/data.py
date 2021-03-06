import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, utils
from torchvision.transforms import (
    ToTensor, Lambda, Compose, Resize, RandomHorizontalFlip, RandomRotation)
from torchvision.transforms.transforms import RandomPerspective
from skills.key_point import SIFT, HarrisCorner
from skills.edge_detectoin import EDGE
from skills.texture import *
import os


def getDataSet(cfg_data):
    """
    Args: cfg_data

    output: dictionary that contains three datasets
            {"train": trainDataSet, "val": valDataSet, "test": testDataSet}
    """
    ################################################################################
    # TODO: preprocess 과정에서 컴비 기술 사용할 수 있도록 수정해야됨.
    #       아마 새로운 transform class를 선언해야 할 듯?
    ################################################################################

    aug_config = cfg_data["train"]["augmentation"]
    resize = cfg_data["resize"]
    train_transforms = [
            Resize((resize,resize)),
            RandomHorizontalFlip(),
            RandomRotation((-180,180)),
            ToTensor()
        ]
    test_transforms = [
            Resize((resize,resize)),
            ToTensor()
        ]
    val_transforms = [
            Resize((resize,resize)),
            ToTensor()
        ]
    
    if aug_config["perspective"]:
        train_transforms.append(RandomPerspective())

    if aug_config["sift"]:
        train_transforms.append(SIFT(mode=aug_config["sift"]))
        test_transforms.append(SIFT(mode=aug_config["sift"]))
        val_transforms.append(SIFT(mode=aug_config["sift"]))

    if aug_config["edge"]:
        train_transforms.append(EDGE(lthr=aug_config["edge"][0],hthr=aug_config["edge"][1]))
        test_transforms.append(EDGE(lthr=aug_config["edge"][0],hthr=aug_config["edge"][1]))
        val_transforms.append(EDGE(lthr=aug_config["edge"][0],hthr=aug_config["edge"][1]))

    if aug_config["gabor"]:
        train_transforms.append(GABOR())
        test_transforms.append(GABOR())
        val_transforms.append(GABOR())

    preprocess = {
        'train': Compose(train_transforms),
        'val': Compose(val_transforms),
        'test': Compose(test_transforms)
    }

    print("[ DATADIR ] ",cfg_data["dir"])
    
    # Print data augmentation
    print("[ AUGMENT ] [resize]",resize,end='')
    for aug in aug_config:
        print(" [{}] {}".format(aug, aug_config[aug]),end='')
    print()
    
    # Check if the dataset is CIFAR10/CIFAR100 or not
    if cfg_data["dir"] == "CIFAR10":
        imgsets = {'train': torchvision.datasets.CIFAR10(root="./data",train=True,transform=preprocess['train'],download=True),
                   'val': torchvision.datasets.CIFAR10(root="./data",train=False,transform=preprocess['val'],download=True),
                   'test': torchvision.datasets.CIFAR10(root="./data",train=False,transform=preprocess['test'],download=True)}
    elif cfg_data["dir"] == "CIFAR100":
        imgsets = {'train': torchvision.datasets.CIFAR100(root="./data",train=True,transform=preprocess['train'],download=True),
                   'val': torchvision.datasets.CIFAR100(root="./data",train=False,transform=preprocess['val'],download=True),
                   'test': torchvision.datasets.CIFAR100(root="./data",train=False,transform=preprocess['test'],download=True)}
    else:
        imgsets = {x: datasets.ImageFolder(os.path.join(cfg_data["dir"], x), preprocess[x])
                    for x in ['train', 'val', 'test']}

    n_class = len(imgsets['train'].classes)
    
    print("[ DATASET ]",end='')
    for x in ['train', 'val', 'test']:
        print(" [{}] n:{}, size:{}".format(x,len(imgsets[x].classes),len(imgsets[x])),end='')
        if len(imgsets[x].classes) != n_class:
            raise ("[WARNING] n_class are different! Reformulate your dataset!")
    print()

    return imgsets, n_class

def getDataLoader(imgsets, cfg_data):
    """
    Args: datasets from getDataset, cfg_data

    output: dictionary that contains three dataloaders
            {"train": trainDataLoader, "val": valDataLoader, "test": testDataLoader}
    """
    imgloaders = {x: DataLoader(imgsets[x], 
                                batch_size = cfg_data[x]["batch_size"],
                                shuffle = cfg_data[x]["shuffle"],
                                num_workers = cfg_data[x]["n_workers"])
                    for x in ['train','val','test']}
    return imgloaders

