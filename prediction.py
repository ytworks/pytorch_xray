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
import cv2


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
            h, w, _ = np.array(image).shape
            original_image = np.array(image)
        # 出力結果を得る
        image = self.trans['val'](image).unsqueeze(0)
        inputs = Variable(image.to(self.device))
        with torch.no_grad():
            feature_map, _, probs = self.model(inputs)
            probs = probs.data.to('cpu').numpy()
            features = feature_map.data.to('cpu').numpy()
        # 確率を計算する
        ps = [0 for x in range(self.ini.getint('network', 'num_classes'))]
        roc_maps = self.ckpt['roc_map']
        for x in range(self.ini.getint('network', 'num_classes')):
            for line in roc_maps[x]:
                if probs[0][x] <= float(line[2]):
                    ps[x] = float(line[0])
        # アノテーションを得る
        if self.ini.get('network', 'pool_type') == 'wildcat':
            annos = self.wildcat_map(features, h, w, original_image)
        else:
            annos = self.cam(feature_map, h, w, original_image)
        # アノテーションを保存する
        img_path = []
        fname, ext = os.path.splitext(os.path.basename(filepath))
        for i, finding in enumerate(self.ckpt['label_list']):
            a = annos[i]
            f = dirname + '/' + fname + '_' + finding + '.png'
            cv2.imwrite(f,a)
            img_path.append(f)
        return np.ones(self.ini.getint('network', 'num_classes')) - ps, img_path

    def get_prob_and_img(self, filepath, dirname, finding=6):
        pass

    def get_json(self, filepath,
                 findings=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        pass

    def cam(self, feature_map, h, w, original_image):
        # weightを取得する (MULTI GPU未対応)
        fc_weight = self.model.fc.weight
        # weightとfeature_mapからROI mapを作成する
        feature_map = torch.transpose(feature_map, 1, 3)  # (N, W, H, C)
        weight = fc_weight.transpose(0, 1)  # (C, num_classes)
        cam = torch.matmul(feature_map, weight)  # (N, W, H, num_classes)
        cam = torch.transpose(cam, 1, 3)  # (N, num_classes, H, W)
        # scaling (sample-wisely)
        cam = cam.contiguous()
        cam_flat = cam.view(cam.size(0), -1)  # (N, num_classes * H * W)
        _min = cam_flat.min(dim=1)[0].view(cam.size(0), 1, 1, 1)  # (N, 1, 1, 1)
        _max = cam_flat.max(dim=1)[0].view(cam.size(0), 1, 1, 1)  # (N, 1, 1, 1)
        cam = (cam - _min) / (_max - _min)  # (N, num_classes, H, W)
        cam = cam.data.to('cpu').numpy()
        # 重ね合わせイメージを作成する
        annos = []
        for m in cam[0]:
            anno = get_roi_map(m, h, w, original_image)
            annos.append(anno)
        return annos

    def wildcat_map(self, feature_map, h, w, original_image):
        annos = []
        for m in feature_map[0]:
            anno = get_roi_map(m, h, w, original_image)
            annos.append(anno)
        return annos

def get_roi_map(image, h, w, original_image):
    a = cv2.resize(image, (h, w))
    img = np.stack((a, a, a), axis=-1)
    img = 255.0 * (img- np.min(img)) / (np.max(img) - np.min(img))
    img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)
    roi_img = cv2.addWeighted(original_image, 0.8, img, 0.2, 0)
    return roi_img



def main():
    # Configの読み込み
    ini, debug_mode, filepath, dirname = utils.config.read_config_for_pred()
    print("Debug mode:", debug_mode)
    obj = predictor(ini)
    p, f = obj.get_probs_and_imgs(filepath, dirname)
    print(p, f)


if __name__ == '__main__':
    main()
