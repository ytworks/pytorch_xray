import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from utils.wildcat import *


def get_base_model(model_name, pretrained, pooling='max', num_classes=15, fine_tuning=False):
    conv, pool_size, final_dense_dim = get_pretrained_model(model_name, pretrained, fine_tuning)
    pool = get_pooling_layer(pool_size, pooling)
    fc = nn.Linear(final_dense_dim, num_classes)
    return conv, pool, fc


def get_pretrained_model(model_name, pretrained, fine_tuning):
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        net = nn.Sequential(*list(model.features))
        pool_size = 7
        final_dense_dim = 1024
    elif model_name == 'vgg':
        model = models.vgg19_bn(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        net = nn.Sequential(*list(model.features))
        pool_size = 7
        final_dense_dim = 512
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        net = nn.Sequential(*list(model.children())[:-2])
        pool_size = 8
        final_dense_dim = 512
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        net = nn.Sequential(*list(model.children())[:-2])
        pool_size = 7
        final_dense_dim = 2048
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        net = nn.Sequential(*list(model.children())[:-2])
        pool_size = 7
        final_dense_dim = 2048
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        net = nn.Sequential(*list(model.children())[:-2])
        pool_size = 7
        final_dense_dim = 2048
    else:
        raise NotImplementedError
    print(net)
    return net, pool_size, final_dense_dim


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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
    def __init__(self, model_name, pretrained, pooling='max', num_classes=15,
                 fine_tuning=False):
        super().__init__()
        self.features, self.pool, self.fc = get_base_model(model_name, pretrained, pooling, num_classes, fine_tuning)
        print(self.features)

    def forward(self, x):
        conv = self.features(x)
        pool = self.pool(conv)
        pool = pool.view(pool.size(0), -1)
        out = self.fc(pool)
        out = torch.sigmoid(out)
        return conv, pool, out

class Model_WildCat(nn.Module):
    def __init__(self, model_name, pretrained, kmax=1, kmin=None, alpha=1, num_maps=1, num_classes=15,
                 fine_tuning=False):
        super().__init__()
        self.features, _, num_features = get_pretrained_model(model_name, pretrained, fine_tuning)
        self.conv = nn.Conv2d(in_channels=num_features,
                              out_channels=num_classes*num_maps,
                              kernel_size=1,
                              stride=1, padding=0, dilation=1, groups=1,
                              bias=True)
        print(self.features)

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
        out = torch.sigmoid(sp)
        return cmap, sp, out
