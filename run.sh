CUDA_VISIBLE_DEVICES=0,1,2 python train.py -init base.ini > base_t.out
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -init base.ini > base_e.out

CUDA_VISIBLE_DEVICES=0,1,2 python train.py -init base_cos.ini > base_cos_t.out
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -init base_cos.ini > base_cos_e.out

CUDA_VISIBLE_DEVICES=0,1,2 python train.py -init full_aug.ini > full_aug_t.out
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -init full_aug.ini > full_aug_e.out

CUDA_VISIBLE_DEVICES=0,1,2 python train.py -init custom_base.ini > custom_base_t.out
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -init custom_base.ini > custom_base_e.out

CUDA_VISIBLE_DEVICES=0,1,2 python train.py -init custom_full.ini > custom_full_t.out
CUDA_VISIBLE_DEVICES=0,1,2 python evaluate.py -init custom_full.ini > custom_full_e.out
