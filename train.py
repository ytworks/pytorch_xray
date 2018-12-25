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
    ckpt_save_path = ini.get('model', 'ckpt_path')
    if ini.get('network', 'pool_type') != 'wildcat':
        model = inference.Model_GlobalPool(model_name=ini.get('network', 'pretrained_model'),
                                           pretrained=ini.getboolean(
                                               'network', 'pretrained'),
                                           pooling=ini.get(
                                               'network', 'global_pool_type'),
                                           num_classes=ini.getint('network', 'num_classes'))
    else:
        model = inference.Model_WildCat(model_name=ini.get('network', 'pretrained_model'),
                                        pretrained=ini.getboolean(
                                            'network', 'pretrained'),
                                        pooling=ini.get(
                                            'network', 'global_pool_type'),
                                        num_classes=ini.getint('network', 'num_classes'))
    checkpoint = {'epoch': None,
                  'optimizer': None,
                  'scheduler': None,
                  'state_dict': None,
                  'best_auc': None,
                  'is_resume': False}
    # cuda
    num_gpu = ini.getint('env', 'num_gpu')
    if num_gpu > 1:
        model = nn.DataParallel(model)
    if num_gpu > 0:
        model.to('cuda')
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
    train_dataset = dataset_class(
        img_path=train_list,
        labels=label_dict,
        ini=ini,
        transform=trans['train']
    )
    valid_dataset = dataset_class(
        img_path=val_list,
        labels=label_dict,
        ini=ini,
        transform=trans['val']
    )
    test_dataset = dataset_class(
        img_path=test_list,
        labels=label_dict,
        ini=ini,
        transform=trans['val']
    )
    batch_size = ini.getint('params', 'batch_size')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
    }
    # Optimizer and Scheduler
    optimizer, scheduler = get_optimizer(model.parameters(),
                                         ini.get('optimizer', 'type'),
                                         ini.getfloat('optimizer', 'lr'),
                                         ini.getfloat('optimizer', 'momentum'),
                                         ini.getfloat(
                                             'optimizer', 'weight_decay'),
                                         ini.getint(
                                             'optimizer', 'lr_decay_steps'),
                                         ini.getfloat('optimizer', 'lr_decay_rate'))
    # Loss func
    criterion = nn.BCELoss()

    # Training Loop
    best_valid_auc = 0.
    init_epoch = 0
    num_epochs = ini.getint('params', 'epoch')
    patience = ini.getint('params', 'patience')

    # start training
    for epoch in range(init_epoch, num_epochs):
        start = time.time()
        print('-' * 50)
        epoch_result = {
            'train': {},
            'valid': {},
            'test': {},
        }

        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                model.train()
                volatile = False
            else:
                model.eval()
                volatile = True

            epoch_loss = []
            epoch_preds = []
            epoch_labels = []

            disable_tqdm = not ini.getboolean('env', 'verbose')
            for inputs, labels in tqdm(dataloaders[phase], disable=disable_tqdm):
                if num_gpu > 0:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                if volatile:
                    with torch.no_grad():
                        inputs = Variable(inputs)
                        labels = Variable(labels)
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()

                preds = model(inputs)
                preds = preds[-1]

                loss = criterion(preds, labels)

                epoch_preds.append(preds.data.to('cpu').numpy())
                epoch_labels.append(labels.data.to('cpu').numpy())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_loss.append(loss.item())

            # calculate auc
            epoch_labels = np.concatenate(epoch_labels, axis=0)
            epoch_preds = np.concatenate(epoch_preds, axis=0)
            aucs = calc_auc(epoch_labels, epoch_preds)
            mean_auc = np.mean(aucs[1:])

            if phase == 'valid':
                if mean_auc > best_valid_auc:
                    best_valid_auc = mean_auc
                    if num_gpu < 2:
                        best_weight = model.state_dict()
                    else:
                        best_weight = model.module.state_dict()
                    best_ckpt = {
                        'epoch': epoch,
                        'optimizer': optimizer,
                        'scheduler': scheduler,
                        'state_dict': best_weight,
                        'best_auc': best_valid_auc,
                    }
                    if isinstance(scheduler, ReduceLROnPlateau):
                        best_ckpt['scheduler'] = None  # cannot save as pkl
                    checkpoint.update(best_ckpt)
                    print('saving checkpoint ...')
                    torch.save(checkpoint, ckpt_save_path)
                    worse_count = 0
                else:
                    worse_count += 1

            # message
            epoch_loss = np.mean(epoch_loss)
            auc_msg = ''
            for idx, finding in enumerate(label_list):
                auc_msg += finding + ':' + \
                    '{score:.3f}'.format(score=aucs[idx]) + ', '
            print(auc_msg)
            epoch_result[phase]['loss'] = epoch_loss
            epoch_result[phase]['all_auc'] = aucs
            epoch_result[phase]['mean_auc'] = mean_auc
            epoch_result[phase]['auc_msg'] = auc_msg

        _time = (time.time() - start) / 60.  # min
        print(
            'EPOCH: {e:03d}, {_time:.1f}min, Train cost: {train_c:.4f}, Valid cost: {valid_c:.4f} Test cost: {test_c:.4f} '.format(
                e=epoch,
                _time=_time,
                train_c=epoch_result['train']['loss'],
                valid_c=epoch_result['valid']['loss'],
                test_c=epoch_result['test']['loss'])
        )
        print('Train AUC: ' + epoch_result['train']['auc_msg'])
        print('Vaiid AUC: ' + epoch_result['valid']['auc_msg'])
        print('Test  AUC: ' + epoch_result['test']['auc_msg'])

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # scheduler.step(epoch_result['valid']['loss'])
                scheduler.step(epoch_result['valid']['mean_auc'])
            else:
                scheduler.step()

        if worse_count >= patience:
            print('Early Stopping')
            return


if __name__ == '__main__':
    main()
