import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torchvision.transforms import (
    ToTensor, Lambda, Compose, Resize, RandomHorizontalFlip, RandomRotation)

import os


def getData(cfg_data):
    """
    Args: cfg_data

    output: dictionary that contains three dataloaders
            {"train": trainDataLoader, "val": valDataLoader, "test": testDataLoader}
    """
    imgsets = _getDataset(cfg_data["dir"])
    imgloaders = {x: DataLoader(imgsets[x], 
                                batch_size = cfg_data[x]["batch_size"],
                                shuffle = cfg_data[x]["shuffle"],
                                num_workers = cfg_data[x]["n_workers"])
                    for x in ['train','val','test']}
    return imgloaders

def _getDataset(img_dir):
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

    imgsets = {x: datasets.ImageFolder(os.path.join(img_dir, x), preprocess[x])
                for x in ['train', 'val', 'test']}
    return imgsets