main_path=/home/smliu/RLHF
launch_path=$main_path/train_dpo.py

cd $main_path

CUDA_VISIBLE_DEVICES=2,3 \
accelerate launch $launch_path \
    # --num_processes 2
    # --sample_batch_size 6