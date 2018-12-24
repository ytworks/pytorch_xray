import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import inference
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
    # 教師データの読み込み
    labels, label_list = utils.label_maker.get_label(ini)
    # trainとvalidationとテストの分割
    train_list, val_list, test_list = utils.data_split.data_split(
        ini, labels, debug_mode)



if __name__=='__main__':
    main()
