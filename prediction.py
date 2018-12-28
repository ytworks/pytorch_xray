import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import inference
import dataloader
import os
from utils.optimizer import get_optimizer
from torch.autograd import Variable
import time
from tqdm import tqdm


class predictor(object):
    def __init__(self, ini):
        self.ini = ini

    def get_probs_and_imgs(self, filepath, dirname,
                           findings=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        return None, None

    def get_prob_and_img(self, filepath, dirname, finding=6):
        pass

    def get_json(self, filepath,
                 findings=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        pass

    def cam(self, feature_map):
        pass

    def wildcat_map(self):
        pass

def main():
    # Configの読み込み
    ini, debug_mode, filepath, dirname = utils.config.read_config_for_pred()
    print("Debug mode:", debug_mode)
    obj = predictor(ini)
    p, f = obj.get_probs_and_imgs(filepath, dirname)
    print(p, f)


if __name__ == '__main__':
    main()
