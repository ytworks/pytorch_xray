import utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import inference

# -1. cudaの設定
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
    model = inference.Model(model_name='densenet121', pretrained=True, pooling='max')



if __name__=='__main__':
    main()
