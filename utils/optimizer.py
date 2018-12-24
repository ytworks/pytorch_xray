import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms





def get_optimizer(params, opt, lr=1e-2, momentum=0., weight_decay=0., lr_decay_steps=20, lr_decay_rate=0.1):
    """
    Get optimizer instance.
    :param params: patametes of the main branch model
    :param opt: str, optimizer type
    :param lr: float, learning rate
    :param momentum: float
    :param weight_decay: float
    :return optimizer: optimizer instance
    :return scheduler: scheduler instance
    """
    scheduler = None
    if opt == 'adam':
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'ssgd':  # scheduled SGD
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay_rate)
    elif opt == 'psgd':  # Plateau SGD
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=2, verbose=True)
    else:
        raise NotImplementedError

    return optimizer, scheduler
