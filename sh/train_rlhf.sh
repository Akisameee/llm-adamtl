main_path=/home/smliu/RLHF
launch_path=$main_path/train_rlhf.py

cd $main_path

CUDA_VISIBLE_DEVICES=7 \
python $launch_path \
    --sample_batch_size 6