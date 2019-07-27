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


def main():
    # Configの読み込み (utils)
    ini, debug_mode = utils.config.read_config()
    print("Debug mode:", debug_mode)
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    ckpt_save_path = ini.get('model', 'ckpt_path')
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
                                          fine_tuning=ini.getboolean('network', 'fine_tuning'),
                                          dropout=ini.getfloat('network', 'dropout'))
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
    # 保存用checkpoint
    checkpoint = {'epoch': None,
                  'optimizer': None,
                  'scheduler': None,
                  'state_dict': None,
                  'best_auc': None,
                  'label_list': None,
                  'roc_map': None,
                  'is_resume': False}
    # 再学習用の読み込み
    if ini.getboolean('env', 'restore'):
        ckpt_path=ini.get('model', 'restore_path')
        ckpt = torch.load(ckpt_path, map_location=device)
        pretrained_dict = {}
        if ini.getboolean('env', 'transfer_mode'):
            for k, v in ckpt['state_dict'].items():
                if k.find('features') > -1:
                    pretrained_dict.setdefault(k, v)
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(ckpt['state_dict'], strict=True)

    # cuda対応
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
    if ini.getboolean('sampling', 'balance'):
        train_list = utils.balancer.sampling(ini, train_list, label_dict)

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
                                         ini.get('optimizer',
                                                 'scheduler_type'),
                                         ini.getint(
                                             'optimizer', 'lr_decay_steps'),
                                         ini.getfloat(
                                             'optimizer', 'lr_decay_rate'),
                                         ini.getint('optimizer', 'patience'),
                                         ini.getint('optimizer', 'te'),
                                         ini.getint('optimizer', 'tmult'),
                                         ini.getfloat('optimizer', 'min_lr')
                                         )

    # Loss func
    criterion = utils.loss.get_loss(loss_type=ini.get('loss', 'loss_type'),
                                    alpha=ini.getfloat('loss', 'focal_alpha'),
                                    gamma=ini.getfloat('loss', 'focal_gamma'))

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
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    zeros = torch.zeros(labels.size(), device=device)
                    input_labels = labels if phase == 'train' else zeros
                    preds = model(inputs, input_labels)
                    preds = preds[-1]

                    loss = criterion(preds, labels)

                    epoch_preds.append(preds.data.to('cpu').numpy())
                    epoch_labels.append(labels.data.to('cpu').numpy())

                    if phase == 'train':
                        loss.backward()
                        utils.gradient_clip.clipping(model,
                                                     ini.getboolean(
                                                         'gradient_clip', 'is_clip'),
                                                     ini.getboolean(
                                                         'gradient_clip', 'is_norm'),
                                                     ini.getfloat('gradient_clip', 'value'))
                        optimizer.step()

                epoch_loss.append(loss.item())

            # calculate auc
            epoch_labels = np.concatenate(epoch_labels, axis=0)
            epoch_preds = np.concatenate(epoch_preds, axis=0)
            aucs, _ = calc_auc(epoch_labels, epoch_preds)
            mean_auc = np.mean(aucs)

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
                        'label_list': label_list,
                        'roc_map': None
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
                score=mean_auc) + ', abnormal average:' + '{score:.3f}'.format(score=abnormal_mean_auc)
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

        # schedulerの実行
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
