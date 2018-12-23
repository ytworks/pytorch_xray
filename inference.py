import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


def get_base_model(model_name, pretrained, pooling='max', num_classes=15):
    densenet = models.densenet121(pretrained=pretrained)
    conv = nn.Sequential(*list(densenet.features))
    pool_size = 7
    final_dense_dim = 1024
    pool = get_pooling_layer(pool_size, pooling)
    fc = nn.Linear(final_dense_dim, num_classes)
    return conv, pool, fc

    
def get_pooling_layer(kernel_size, pooling_method):
    if pooling_method == 'avg':
        pool = nn.AvgPool2d(kernel_size=kernel_size)
    elif pooling_method == 'max':
        pool = nn.MaxPool2d(kernel_size=kernel_size)
    else:
        raise NotImplementedError
    return pool


class Model(nn.Module):
    def __init__(self, model_name, pretrained, pooling='max', num_classes=15):
        super().__init__()
        self.conv, self.pool, self.fc = get_base_model(model_name, pretrained, pooling, num_classes)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)
        pool = pool.view(pool.size(0), -1)
        out = self.fc(pool)
        # out = nn.Sigmoid()(out)
        out = F.sigmoid(out)
        return conv, pool, out
