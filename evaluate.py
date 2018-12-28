import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import inference
import dataloader
import os
from utils.optimizer import get_optimizer
from torch.autograd import Variable
import time
from tqdm import tqdm
from utils.metrics import calc_auc
import csv


def main():
    # Configの読み込み (utils)
    ini, debug_mode = utils.config.read_config()
    print("Debug mode:", debug_mode)
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    ckpt_path = ini.get('model', 'ckpt_path')
    if ini.get('network', 'pretrained_model') == 'custom':
        model = utils.custom.Model_CUSTOM(model_name=ini.get('network', 'pretrained_model'),
                                          pretrained=ini.getboolean(
                                              'network', 'pretrained'),
                                          kmax=ini.getfloat(
                                              'network', 'wc_kmax'),
                                          kmin=ini.getfloat(
                                              'network', 'wc_kmin'),
                                          alpha=ini.getfloat(
                                              'network', 'wc_alpha'),
                                          num_maps=ini.getint(
                                              'network', 'num_maps'),
                                          num_classes=ini.getint(
                                              'network', 'num_classes'),
                                          fine_tuning=ini.getboolean('network', 'fine_tuning'))
    else:
        if ini.get('network', 'pool_type') != 'wildcat':
            model = inference.Model_GlobalPool(model_name=ini.get('network', 'pretrained_model'),
                                               pretrained=ini.getboolean(
                                                   'network', 'pretrained'),
                                               pooling=ini.get(
                                                   'network', 'global_pool_type'),
                                               num_classes=ini.getint(
                                                   'network', 'num_classes'),
                                               fine_tuning=ini.getboolean('network', 'fine_tuning'))
        else:
            model = inference.Model_WildCat(model_name=ini.get('network', 'pretrained_model'),
                                            pretrained=ini.getboolean(
                                                'network', 'pretrained'),
                                            kmax=ini.getfloat(
                                                'network', 'wc_kmax'),
                                            kmin=ini.getfloat(
                                                'network', 'wc_kmin'),
                                            alpha=ini.getfloat(
                                                'network', 'wc_alpha'),
                                            num_maps=ini.getint(
                                                'network', 'num_maps'),
                                            num_classes=ini.getint(
                                                'network', 'num_classes'),
                                            fine_tuning=ini.getboolean('network', 'fine_tuning'))

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    # cuda
    num_gpu = ini.getint('env', 'num_gpu')
    if num_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device)
    if num_gpu > 0:
        torch.backends.cudnn.benchmark = True
    # 教師データの読み込み
    label_dict, label_list = utils.label_maker.get_label(ini)
    # trainとvalidationとテストの分割
    train_list, val_list, test_list = utils.data_split.data_split(
        ini, label_dict, debug_mode)
    # データオーグメンターション
    trans = dataloader.get_transform(ini)
    # データセット
    dataset_class = dataloader.ChestXRayDataset
    test_dataset = dataset_class(
        img_path=test_list,
        labels=label_dict,
        ini=ini,
        transform=trans['val']
    )
    batch_size = ini.getint('params', 'batch_size')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



    epoch_loss = []
    epoch_preds = []
    epoch_labels = []
    disable_tqdm = not ini.getboolean('env', 'verbose')
    for inputs, labels in tqdm(test_loader, disable=disable_tqdm):
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))

        with torch.no_grad():
            preds = model(inputs)
            preds = preds[-1]
            epoch_preds.append(preds.data.to('cpu').numpy())
            epoch_labels.append(labels.data.to('cpu').numpy())


    # calculate auc
    epoch_labels = np.concatenate(epoch_labels, axis=0)
    epoch_preds = np.concatenate(epoch_preds, axis=0)
    aucs, prob_map = calc_auc(epoch_labels, epoch_preds)
    mean_auc = np.mean(aucs)

    auc_msg = ''
    auc_abnormal = []
    for idx, finding in enumerate(label_list):
        auc_msg += finding + ':' + \
            '{score:.3f}'.format(score=aucs[idx]) + ', '
        if idx != 10:
            auc_abnormal.append(aucs[idx])
    abnormal_mean_auc = np.mean(auc_abnormal)
    print(auc_msg, 'average:',
          '{score:.3f}'.format(score=mean_auc),
          'abnormal average:',
          '{score:.3f}'.format(score=abnormal_mean_auc),
          )
    auc_msg += 'average:' + '{score:.3f}'.format(
        score=mean_auc) + 'abnormal average:' + '{score:.3f}'.format(score=abnormal_mean_auc)
    # checkpoint update for roc map
    ckpt['roc_map'] = prob_map
    torch.save(ckpt, ckpt_path)
    # csvの出力
    preds, labels = epoch_preds.T, epoch_labels.T
    for i, l in enumerate(ckpt['label_list']):
        with open(ini.get('model', 'csv_path').replace('.csv', '_'+l+'.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            data = zip(preds[i], labels[i])
            writer.writerows(data)


if __name__=='__main__':
    main()
