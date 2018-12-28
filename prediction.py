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
from PIL import Image


class predictor(object):
    def __init__(self, ini):
        self.ini = ini
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model
        ckpt_path = self.ini.get('model', 'ckpt_path')
        if ini.get('network', 'pretrained_model') == 'custom':
            self.model = utils.custom.Model_CUSTOM(model_name=ini.get('network', 'pretrained_model'),
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
                self.model = inference.Model_GlobalPool(model_name=ini.get('network', 'pretrained_model'),
                                                        pretrained=ini.getboolean(
                                                            'network', 'pretrained'),
                                                        pooling=ini.get(
                                                            'network', 'global_pool_type'),
                                                        num_classes=ini.getint(
                                                            'network', 'num_classes'),
                                                        fine_tuning=ini.getboolean('network', 'fine_tuning'))
            else:
                self.model = inference.Model_WildCat(model_name=ini.get('network', 'pretrained_model'),
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
        self.ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(self.ckpt['state_dict'])
        self.model.eval()
        # cuda
        num_gpu = ini.getint('env', 'num_gpu')
        if num_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        if num_gpu > 0:
            torch.backends.cudnn.benchmark = True
        # 前処理
        self.trans = dataloader.get_transform(self.ini)

    def get_probs_and_imgs(self, filepath, dirname,
                           findings=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        # 画像を読み込む
        with Image.open(filepath) as img:
            image = img.convert('RGB')
        # 出力結果を得る
        image = self.trans['val'](image).unsqueeze(0)
        inputs = Variable(image.to(self.device))
        with torch.no_grad():
            features, _, probs = self.model(inputs)
            probs = probs.data.to('cpu').numpy()
            features = features.data.to('cpu').numpy()
        print(probs)
        # 確率を計算する
        # アノテーションを得る
        # アノテーションを保存する
        return None, None

    def get_prob_and_img(self, filepath, dirname, finding=6):
        pass

    def get_json(self, filepath,
                 findings=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        pass

    def cam(self, feature_map):
        pass

    def wildcat_map(self):
        pass


def main():
    # Configの読み込み
    ini, debug_mode, filepath, dirname = utils.config.read_config_for_pred()
    print("Debug mode:", debug_mode)
    obj = predictor(ini)
    p, f = obj.get_probs_and_imgs(filepath, dirname)
    print(p, f)


if __name__ == '__main__':
    main()
