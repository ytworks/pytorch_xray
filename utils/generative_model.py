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



class Generative_Model(nn.Module):
    def __init__(self, model_name, pretrained, kmax=1, kmin=None, alpha=1, num_maps=1, num_classes=15,
                 fine_tuning=False, dropout=0.5):
        super().__init__()
        # 生成モデルのための特徴マップの抽出
        model = models.resnet34(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        pool_size = 8
        final_dense_dim = 512

        print(self.encoder)
        vector_size = 1024
        self.fc_en1 = nn.Linear(pool_size * pool_size * final_dense_dim, vector_size)
        self.fc_en2 = nn.Linear(pool_size * pool_size * final_dense_dim, vector_size)

        # デコーダー
        self.fc_de1 = nn.Linear(vector_size, pool_size * pool_size * final_dense_dim)
        self.fc_de2 = nn.Linear(vector_size, pool_size * pool_size * final_dense_dim)


        # 差分エンコーダー
        dif_model = models.resnet34(pretrained=pretrained)
        set_parameter_requires_grad(dif_model, fine_tuning)
        self.dif_encoder = nn.Sequential(*list(dif_model.children())[:-2])

        # 最終出力層の定義
        num_features = 512
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
