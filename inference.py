import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from utils.wildcat import *


def get_base_model(model_name, pretrained, pooling='max', num_classes=15):
    net = get_pretrained_model(model_name, pretrained)
    conv = nn.Sequential(*list(net.features))
    pool_size = 7
    final_dense_dim = 1024
    pool = get_pooling_layer(pool_size, pooling)
    fc = nn.Linear(final_dense_dim, num_classes)
    return conv, pool, fc


def get_pretrained_model(model_name, pretrained):
    densenet = models.densenet121(pretrained=pretrained)
    print(densenet.features)
    return densenet


def get_pooling_layer(kernel_size, pooling_method):
    if pooling_method == 'avg':
        pool = nn.AvgPool2d(kernel_size=kernel_size)
    elif pooling_method == 'max':
        pool = nn.MaxPool2d(kernel_size=kernel_size)
    else:
        raise NotImplementedError
    return pool

# paramsの設定で学習係数を層ごとに変えられるらしい
class Model_GlobalPool(nn.Module):
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

class Model_WildCat(nn.Module):
    def __init__(self, model_name, pretrained, kmax=1, kmin=None, alpha=1, num_maps=1, num_classes=15):
        super().__init__()
        self.net = get_pretrained_model(model_name, pretrained)
        self.features = nn.Sequential(*list(self.net.features))
        # conv
        num_features = self.net.features.norm5.num_features
        self.conv = nn.Conv2d(in_channels=num_features,
                              out_channels=num_classes*num_maps,
                              kernel_size=1,
                              stride=1, padding=0, dilation=1, groups=1,
                              bias=True)

        self.cwp = nn.Sequential()
        self.cwp.add_module('conv', self.conv)
        self.cwp.add_module('class_wise', ClassWisePool(num_maps))
        print(self.cwp)
        self.sp = nn.Sequential()
        self.sp.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
        print(self.sp)

    def forward(self, x):
        features = self.features(x)
        cmap = self.cwp(features)
        sp = self.sp(cmap)
        out = F.sigmoid(sp)
        return cmap, sp, out
