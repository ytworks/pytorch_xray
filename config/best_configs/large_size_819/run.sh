cd $(dirname $0)
cd ../../../

CUDA_VISIBLE_DEVICES=0,1,2 python train.py -config ./config/best_config/large_size_819/fine_tuning.ini
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -config ./config/best_config/large_size_819/fine_tuning.ini

CUDA_VISIBLE_DEVICES=0,1,2 python train.py -config ./config/best_config/large_size_819/transfer.ini
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -config ./config/best_config/large_size_819/transfer.ini
