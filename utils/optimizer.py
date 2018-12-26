import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineAnnealingLR_with_Restart(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. The original pytorch
    implementation only implements the cosine annealing part of SGDR,
    I added my own implementation of the restarts part.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations (batch) convert it to epoch.
        T_mult (float): Increase T_max by a factor of T_mult
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        model (pytorch model): The model to save.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mult, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.Te = self.T_max
        self.eta_min = eta_min
        self.current_epoch = last_epoch
        self.lr_history = []

        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.current_epoch / self.Te)) / 2

                for base_lr in self.base_lrs]

        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None):
        if epoch is None:

            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        ## restart
        if self.current_epoch == self.Te:
            print("restart at epoch {:03d}".format(self.last_epoch + 1))
            ## reset epochs since the last reset
            self.current_epoch = 0

            ## reset the next goal
            self.Te = int(self.Te * self.T_mult)
            self.T_max = self.T_max + self.Te


def get_optimizer(params, opt, lr=1e-2, momentum=0., weight_decay=0.,
                  scheduler_type='stepLR',
                  lr_decay_steps=20, lr_decay_rate=0.1,
                  patience=2,
                  te=1.0, tmult=2.0, lr_min=1.0e-8):
    scheduler = None
    if opt == 'adam':
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt == 'amsgrad':
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif opt == 'sgd':  # scheduled SGD
        optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    if scheduler_type == 'stepLR':
        scheduler = StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay_rate)
    elif scheduler_type == 'ReduceOnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate,
                                      patience=patience, verbose=True,
                                      min_lr=lr_min)
    elif scheduler_type == 'CosineAnnealing':
        scheduler = CosineAnnealingLR_with_Restart(optimizer, T_max=te, T_mult=tmult,
                                                   eta_min=lr_min)

    return optimizer, scheduler
