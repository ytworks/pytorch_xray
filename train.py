import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import inference
import dataloader
import os

# 0.2 データの分割
# 0.5. dataloader
# 1. Lossの設定
# 2. preprocess
# 3. Loss(pred, true)
# 4. no grad
# 5. backward
# 6. opt
# 7. eval
# save

def main():
    # Configの読み込み (utils)
    ini, debug_mode = utils.config.read_config()
    print("Debug mode:", debug_mode)
    # model
    model = inference.Model(model_name='densenet121', pretrained=True, pooling='max')
    # cuda
    num_gpu = ini.getint('env', 'num_gpu')
    if num_gpu > 1:
        model = nn.DataParallel(model)
    if num_gpu > 0:
        model.to('cuda')
        torch.backends.cudnn.benchmark = True
    # 教師データの読み込み
    labels, label_list = utils.label_maker.get_label(ini)
    # trainとvalidationとテストの分割
    train_list, val_list, test_list = utils.data_split.data_split(
        ini, labels, debug_mode)
    # データオーグメンターション
    trans = dataloader.get_transform(ini)
    # データセット
    dataset_class = dataloader.ChestXRayDataset
    train_dataset = dataset_class(
        img_path=train_list,
        labels=labels,
        ini=ini,
        transform=transforms.Compose(trans['train'])
        )
    valid_dataset = dataset_class(
        img_path=val_list,
        labels=labels,
        ini=ini,
        transform=transforms.Compose(trans['val'])
        )
    test_dataset = dataset_class(
        img_path=test_list,
        labels=labels,
        ini=ini,
        transform=transforms.Compose(trans['val'])
        )
    batch_size = ini.getint('params', 'batch_size')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
    }
    # Optimizer and Scheduler

    # Loss func
    criterion = nn.BCELoss()





if __name__=='__main__':
    main()
