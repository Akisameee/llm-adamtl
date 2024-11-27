main_path=/home/smliu/llm-adamtl
launch_path=$main_path/eval_multitask.py

model_path=/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0

cd $main_path

# TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
accelerate launch --num_processes=4 --main_process_port=29501 $launch_path \
# python $launch_path \
#     --sample_batch_size 1 \
#     --dateset_cfg_tokenizer_pretrain_path $model_path \
#     --model_cfg_model_pretrain_path $model_path \
#     --ref_cfg_model_pretrain_path $model_path \