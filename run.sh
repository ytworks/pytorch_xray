

CUDA_VISIBLE_DEVICES=0 python train.py -config ./experiments/base_cosine.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./experiments/base_cosine.ini

CUDA_VISIBLE_DEVICES=0 python train.py -config ./experiments/base_15.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./experiments/base_15.ini

CUDA_VISIBLE_DEVICES=0 python train.py -config ./experiments/base_balance.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./experiments/base_balance.ini

CUDA_VISIBLE_DEVICES=0 python train.py -config ./experiments/base_custom.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./experiments/base_custom.ini
