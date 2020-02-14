
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


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class SqEx(nn.Module):
    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()
        if n_features % reduction != 0:
            raise ValueError(
                'n_features must be divisible by reduction (default = 16)')
        self.linear1 = nn.Linear(
            n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(
            n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class Model_CUSTOM(nn.Module):
    def __init__(self, model_name, pretrained, kmax=1, kmin=None, alpha=1, num_maps=1, num_classes=15,
                 fine_tuning=False, dropout=0.5):
        super().__init__()
        model = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        self.se1 = CBAM(128)
        self.se2 = CBAM(256)
        self.se3 = CBAM(512)
        '''
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=3,
                               kernel_size=3,
                               stride=2, padding=0, dilation=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=3,
                               out_channels=3,
                               kernel_size=3,
                               stride=2, padding=0, dilation=1, groups=1, bias=False)
        print(self.conv1.weight)
        kernel = [[1. / 16., 2. / 16., 1. / 16.],
                  [2. / 16., 4. / 16., 2. / 16.],
                  [1. / 16., 2. / 16., 1. / 16.]]
        filter = [[kernel, np.zeros((3,3)), np.zeros((3,3))],
                  [np.zeros((3,3)), kernel, np.zeros((3,3))],
                  [np.zeros((3,3)), np.zeros((3,3)), kernel]
                  ]
        gf = torch.from_numpy(np.array(filter).astype(np.float32))

        self.conv1.weight = torch.nn.Parameter(gf)
        self.conv2.weight = torch.nn.Parameter(gf)
        '''
        self.features = nn.Sequential()
        #self.features.add_module('c1', self.conv1)
        #self.features.add_module('c2', self.conv2)
        self.features.add_module('conv0', model.features.conv0)
        self.features.add_module('norm0', model.features.norm0)
        self.features.add_module('relu0', model.features.relu0)
        self.features.add_module('pool0', model.features.pool0)
        self.features.add_module('denseblock1', model.features.denseblock1)
        self.features.add_module('transition1', model.features.transition1)
        self.features.add_module('se1', self.se1)
        self.features.add_module('denseblock2', model.features.denseblock2)
        self.features.add_module('transition2', model.features.transition2)
        self.features.add_module('se2', self.se2)
        self.features.add_module('denseblock3', model.features.denseblock3)
        self.features.add_module('transition3', model.features.transition3)
        self.features.add_module('se3', self.se3)
        self.features.add_module('denseblock4', model.features.denseblock4)
        self.features.add_module('norm5', model.features.norm5)
        print(self.features)

        num_features = 1024
        self.conv = nn.Conv2d(in_channels=num_features,
                              out_channels=num_classes * num_maps,
                              kernel_size=1,
                              stride=1, padding=0, dilation=1, groups=1,
                              bias=True)

        self.cwp = nn.Sequential()
        self.cwp.add_module('conv', self.conv)
        self.cwp.add_module('class_wise', ClassWisePool(num_maps))
        self.dropout = nn.Dropout2d(p=dropout)
        print(self.cwp)
        self.sp = nn.Sequential()
        self.sp.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
        print(self.sp)

    def forward(self, x):
        features = self.features(x)
        cmap = self.cwp(features)
        cmap = self.dropout(cmap)
        sp = self.sp(cmap)
        out = torch.sigmoid(sp)
        return cmap, sp, out
