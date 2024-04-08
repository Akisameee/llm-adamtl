main_path=/home/smliu/RLHF
launch_path=$main_path/train_rlhf.py

cd $main_path

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch $launch_path \
    --sample_batch_size 6 \
    --model_cfg_model_pretrain_path /home/share/models/huggingface/bit-dny/MindLLM \
    --model_cfg_peft_cfg_r 4