import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from .wildcat import *


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



class Generative_Model(nn.Module):
    def __init__(self, model_name, pretrained, kmax=1, kmin=None, alpha=1, num_maps=1, num_classes=15,
                 fine_tuning=False, dropout=0.5):
        super().__init__()
        # VAE
        self.fc1 = nn.Linear(224*224*3, 512)
        self.fc21 = nn.Linear(512, 64)
        self.fc22 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 512)
        self.fc4 = nn.Linear(512, 224*224*3)


        model = models.resnet34(pretrained=pretrained)
        set_parameter_requires_grad(model, fine_tuning)
        self.features = nn.Sequential()
        #self.features = nn.Sequential(*list(model.children())[:-2])

        #L1
        self.l1 = nn.Sequential()
        self.l1.add_module('conv1', model.conv1)
        self.l1.add_module('norm1', model.bn1)
        self.l1.add_module('relu1', model.relu)
        self.l1.add_module('pool1', model.maxpool)
        #L2
        self.l2 = nn.Sequential()
        self.l2.add_module('l2', model.layer1)
        #L3
        self.l3 = nn.Sequential()
        self.l3.add_module('l3', model.layer2)
        #L4
        self.l4 = nn.Sequential()
        self.l4.add_module('l4', model.layer3)
        #L5
        self.l5 = nn.Sequential()
        self.l5.add_module('l5', model.layer4)

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
        self.transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # VAE
        h1 = F.relu(self.fc1(x.view(-1,224*224*3)))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        z = self.reparameterize(mu, logvar)
        h3 = F.relu(self.fc3(z))
        recon_x = F.sigmoid(self.fc4(h3))
        img = x - recon_x.view(-1, 3, 224, 224)
        img = self.transforms(img)
        #L1
        l1 = self.l1(img)
        #L2
        l2 = self.l2(l1)
        #L3
        l3 = self.l3(l2)
        #L4
        l4 = self.l4(l3)
        #L5
        l5 = self.l5(l4)
        #classifier
        features = l5
        cmap = self.cwp(features)
        cmap = self.dropout(cmap)
        sp = self.sp(cmap)
        out = torch.sigmoid(sp)
        return cmap, sp, recon_x, mu, logvar, out


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
