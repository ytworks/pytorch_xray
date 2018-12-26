import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


def get_loss(loss_type, alpha, gamma):
    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise NotImplementedError
