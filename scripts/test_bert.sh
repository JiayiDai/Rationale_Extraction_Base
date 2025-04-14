CUDA_VISIBLE_DEVICES=7 python run/main.py --cuda --model_form bert --init_lr 1e-5 --batch_size 64 --rand_seed 2024 --epochs 20 --get_rationales --datasize 5000 --select_lambda 8e-4 --train
