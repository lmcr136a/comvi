import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torchvision.transforms import (
    ToTensor, Lambda, Compose, Resize, RandomHorizontalFlip, RandomRotation)

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

    preprocess = {
        'train': Compose([
            Resize((256,256)),
            RandomHorizontalFlip(),
            RandomRotation((-180,180)),
            ToTensor()
        ]),
        'val': Compose([
            Resize((256,256)),
            ToTensor()
        ]),
        'test': Compose([
            Resize((256,256)),
            ToTensor()
        ])
    }

    print("[ DATADIR ] ",cfg_data["dir"])

    imgsets = {x: datasets.ImageFolder(os.path.join(cfg_data["dir"], x), preprocess[x])
                for x in ['train', 'val', 'test']}

    n_class = len(imgsets['train'].classes)
    
    for x in ['train', 'val', 'test']:
        print("[ DATASET ] [",x,"] N_CLASS:",len(imgsets[x].classes),", SIZE:",len(imgsets[x]))
        if len(imgsets[x].classes) != n_class:
            raise ("[WARNING] n_class are different! Reformulate your dataset!")

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