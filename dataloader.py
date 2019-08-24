import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image


class Rotate90(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        if len(self.angles) == 1:
            return img
        else:
            angle = np.random.choice(self.angles)
        return transforms.functional.rotate(img, angle)

    def __repr__(self):
        return self.__class__.__name__


def get_transform(ini):
    angles = [
        0, 90, -90] if ini.getboolean('augmentation', 'rotate90') else [0]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(
                p=ini.getfloat('augmentation', 'flip_h')),
            transforms.RandomVerticalFlip(
                p=ini.getfloat('augmentation', 'flip_v')),
            Rotate90(angles),
            transforms.RandomResizedCrop(
                ini.getint('augmentation', 'crop_size')),
            transforms.RandomRotation(ini.getint('augmentation', 'rotation')),
            transforms.Resize(ini.getint('augmentation', 'resize_size')),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(ini.getint('augmentation', 'crop_size')),
            transforms.Resize(ini.getint('augmentation', 'resize_size')),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]), }
    return data_transforms


class ChestXRayDataset(Dataset):
    """Dataset class for Chest X-Ray dataset """

    def __init__(self, img_path, ini, labels, transform=None, grayscale=False):
        """
        :param paths_file: str, path to image path list txt file
        :param config_file: str, path to config file
        :param transform: transform function
        :param grayscale: bool, whether to load image in grayscale
        """
        self.image_paths = img_path
        self.label_dict = labels
        self.transform = transform
        self.grayscale = grayscale
        self.ini = ini

    def __getitem__(self, idx):
        """
        :param idx: int
        :return X: torch.FloatTensor, input image
        :return Y: torch.FloatTensor, target label
        """
        image_path = self.image_paths[idx]
        with Image.open(self.ini.get('data', 'img_dir') + '/' + image_path) as img:
            if self.grayscale:
                X = img.convert('L')
            else:
                X = img.convert('RGB')
        Y = np.array(self.label_dict[image_path]['label'])

        if self.transform is not None:
            X = self.transform(X)

        return X, Y

    def __len__(self):
        return len(self.image_paths)
