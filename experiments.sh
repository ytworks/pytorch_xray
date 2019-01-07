CUDA_VISIBLE_DEVICES=0,1,2 python train.py -config ./dropout.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./dropout.ini
