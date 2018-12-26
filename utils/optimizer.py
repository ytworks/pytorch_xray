import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms





def get_optimizer(params, opt, lr=1e-2, momentum=0., weight_decay=0.,
                  scheduler_type='stepLR',
                  lr_decay_steps=20, lr_decay_rate=0.1):
    scheduler = None
    if opt == 'adam':
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':  # scheduled SGD
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    if scheduler_type == 'stepLR':
        scheduler = StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay_rate)
    elif scheduler_type == 'ReduceOnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=2, verbose=True)

    return optimizer, scheduler
