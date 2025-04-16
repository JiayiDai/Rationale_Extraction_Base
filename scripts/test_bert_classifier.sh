CUDA_VISIBLE_DEVICES=1 python3 run/main.py --cuda --model_form bert --init_lr 2e-5 --batch_size 32 --rand_seed 2024 --epochs 10 --train --test --task_name classifier
