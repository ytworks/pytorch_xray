CUDA_VISIBLE_DEVICES=0 python train.py -config ./experiments/test_cam.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./experiments/test_cam.ini

CUDA_VISIBLE_DEVICES=0 python train.py -config ./experiments/test_wc.ini
CUDA_VISIBLE_DEVICES=0 python evaluate.py -config ./experiments/test_wc.ini
