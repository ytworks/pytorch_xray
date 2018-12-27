
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from .wildcat import *


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Model_CUSTOM(nn.Module):
    def __init__(self, model_name, pretrained, kmax=1, kmin=None, alpha=1, num_maps=1, num_classes=15,
                 fine_tuning=False):
        super().__init__()
        model = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        self.features = nn.Sequential()
        self.features.add_module('conv0', model.features.conv0)
        self.features.add_module('norm0', model.features.norm0)
        self.features.add_module('relu0', model.features.relu0)
        self.features.add_module('pool0', model.features.pool0)
        self.features.add_module('denseblock1', model.features.denseblock1)
        self.features.add_module('transition1', model.features.transition1)
        self.features.add_module('denseblock2', model.features.denseblock2)
        self.features.add_module('transition2', model.features.transition2)
        self.features.add_module('denseblock3', model.features.denseblock3)
        self.features.add_module('transition3', model.features.transition3)
        self.features.add_module('denseblock4', model.features.denseblock4)
        self.features.add_module('norm5', model.features.norm5)
        print(self.features)
        num_features = 1024
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
        out = torch.sigmoid(sp)
        return cmap, sp, out
