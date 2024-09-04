import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat, reduce
from collections import deque
from functools import partial
from random import randrange
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead
from accelerate.utils import broadcast

from base_trainer import Base_Trainer
from datas.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import Panacea_PPO_Config, RM_Config, SVD_Lora_Config, SVD_Lora_Altered_Config
# from modules.lms import BaseLM, RewardLM
from modules.base import BaseLMWithValueHeads
from modules.ppo import PPO_Trainer, PPOMemory
from modules.moppo import MOPPO_Trainer
from modules.pefts import set_all_adapters
from modules.utils import shift, log_prob, merge_dict, get_model

TEST = 0

class MTL_Trainer(Base_Trainer):

    def __init__(
        self,
        config
    ):
        pass
        

def main():

    config = Panacea_PPO_Config()

    data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    # sub_data_path = ['helpful-base', 'harmless-base']
    sub_data_path = ['harmless-base']
    config.dateset_cfg.data_path = data_path
    config.dateset_cfg.sub_data_path = sub_data_path

    # Panacea
    # config.reward_scalariztion_type = 'ls'
    # config.manipulator_cfg.weighted_loss_type = None
    # config.model_cfg.peft_cfg = SVD_Lora_Config(pref_dim = 2)

    config.reward_scalariztion_type = None
    config.manipulator_cfg.weighted_loss_type = 'mols'
    config.model_cfg.peft_cfg = SVD_Lora_Altered_Config(pref_dim = 2)
    
    config.model_cfg.peft_cfg.r = 6
    config.model_cfg.peft_cfg.pref_r = 1
    config.model_cfg.peft_cfg.lora_alpha = 32
    
    model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

    # model_path = '/home/share/models/huggingface/meta-llama/Llama-2-7b-chat-hf'
    # config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    config.dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path
    config.ref_cfg.model_pretrain_path = model_path
    
    rm_path_1 = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
    rm_path_2 = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-harmless-reward_model')
    config.reward_cfg_0 = RM_Config(model_pretrain_path = rm_path_1)
    config.reward_cfg_1 = RM_Config(model_pretrain_path = rm_path_2)
    config.reward_name_0 = 'helpful'
    config.reward_name_1 = 'harmless'
    # config.reward_cfg_0.reward_weight = 0.1
    # config.reward_cfg_1.reward_weight = 10

    config.manipulator_cfg.svd_lora_type = 'adaptive'
    # config.manipulator_cfg.svd_lora_random_init = True
    config.manipulator_cfg.svd_lora_split_percentage = 0.125

    config.lr = 1e-4
    config.model_cfg.peft_cfg.init_strategy = 'diag_zero'
    # config.model_cfg.peft_cfg.init_strategy = 'b_zero'

    # whole dataset
    # max_sample = 0
    # config.n_update_timestep = 64
    # config.accelertor_cfg.gradient_accumulation_steps = 8
    # config.train_batch_size = 2
    # config.manipulator_cfg.n_adapt_step = 128

    # 1/2 dataset
    # max_sample = 20000
    max_sample = 15000
    config.n_update_timestep = 64
    config.accelertor_cfg.gradient_accumulation_steps = 8
    config.train_batch_size = 1
    config.manipulator_cfg.n_adapt_step = 128

    if TEST:
        config.accelertor_cfg.gradient_accumulation_steps = 2
        config.n_update_timestep = 8
        config.n_eval_sample = 4
        config.n_save_step = 1
        config.n_eval_epoch = 6
        config.manipulator_cfg.n_adapt_step = 2

    config.parse_args()

    trainer = MTL_Trainer(
        config = config
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    train_dataset = Instruct_Dataset(config.dateset_cfg)
    train_dataset.load(mode = 'train', max_sample = max_sample if not TEST else 60)
    eval_dataset = Instruct_Dataset(config.dateset_cfg)
    eval_dataset.load(mode = 'eval', max_sample = 500 if not TEST else 50)
    trainer.train_ppo(
        train_dataset = train_dataset.get_generator(),
        eval_dataset = eval_dataset.get_generator(),
        n_episode = config.n_episode,
        n_update_timestep = config.n_update_timestep,
        sample_batch_size = config.sample_batch_size
    )

if __name__ == '__main__':
    
    main()
