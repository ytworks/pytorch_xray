CUDA_VISIBLE_DEVICES=0 python train.py -config ./fine_tuning.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./fine_tuning.ini

CUDA_VISIBLE_DEVICES=0 python train.py -config ./transfer.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./transfer.ini
