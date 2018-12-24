import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


# 1. transform
# 2. dataset_class
# 3. dataloader
# 4. to train.py
def get_transform(ini):
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(ini.getint('augmentation', 'resize_size')),
        transforms.RandomResizedCrop(ini.getint('augmentation', 'crop_size')),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ini.getint('augmentation', 'rotation')),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(ini.getint('augmentation', 'resize_size')),
        transforms.CenterCrop(ini.getint('augmentation', 'crop_size')),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}
