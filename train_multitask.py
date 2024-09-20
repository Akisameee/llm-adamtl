import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from datas import Instruct_MTL_Dataset, instruct_mtl_collator
from configs import Panacea_PPO_Config, RM_Config, SVD_Lora_Altered_Config, Instruct_MTL_Config
# from modules.lms import BaseLM, RewardLM
from modules.base import BaseLMWithValueHeads
from modules.manipulators import Base_MO_Manipulator_Altered
from modules.pefts import set_all_adapters, SVD_Lora_Linear, SVD_Lora_Linear_Altered
from modules.utils import shift, log_prob, merge_dict, get_model

TEST = 0

class MTL_Trainer(Base_Trainer):

    def __init__(
        self,
        config: Instruct_MTL_Config
    ):
        super().__init__(
            config = config,
            accelerator_cfg = config.accelertor_cfg,
            model_cfg = config.model_cfg
        )
        
        self.manipulator = Base_MO_Manipulator_Altered(
            model = self.model,
            accelerator = self.accelerator,
            optimizer = self.optimizer,
            logger = self.logger,
            pref_dim = len(config.dataset_data_paths),
            svd_lora_type = 'adaptive'
        )

    def set_pref_vec(
        self,
        pref_vec
    ):
        for module in self.model.modules():
            if isinstance(module, (SVD_Lora_Linear, SVD_Lora_Linear_Altered)):
                module.set_pref_vec(pref_vec)

        self.manipulator.set_pref_vec(pref_vec)

    def train(
        self,
        train_dataset: Dataset,
        # eval_dataset: Dataset,
        train_batch_size: int,
        n_episode: int = 1
    ):
        set_all_adapters(
            model = self.model,
            enable = True
        )

        dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = train_batch_size,
            shuffle = True,
            collate_fn = instruct_mtl_collator,
            drop_last = True
        )
        dataloader = self.accelerator.prepare(dataloader)
        max_timestep = len(dataloader) * train_batch_size * n_episode

        for episode in range(n_episode):
            for all_batch_inputs in dataloader:
                for task_idx, batch_inputs in enumerate(all_batch_inputs):
                    out = self.model(**batch_inputs)
                    print(out)
        

def main():

    config = Instruct_MTL_Config()
    config.dataset_data_paths = [
        '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/7M_domains/code',
        '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/7M_domains/commonsense',
        '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/7M_domains/math',
        '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/7M_domains/subjective'
    ]

    config.model_cfg.peft_cfg = SVD_Lora_Altered_Config(pref_dim = len(config.dataset_data_paths))
    
    config.model_cfg.peft_cfg.r = 4
    config.model_cfg.peft_cfg.pref_r = 1
    config.model_cfg.peft_cfg.lora_alpha = 32
    
    model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

    # model_path = '/home/share/models/huggingface/meta-llama/Llama-2-7b-chat-hf'
    # config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    config.base_dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path

    config.manipulator_cfg.svd_lora_type = 'adaptive'
    # config.manipulator_cfg.svd_lora_random_init = True
    config.manipulator_cfg.svd_lora_split_percentage = 0.125

    config.lr = 1e-4
    config.model_cfg.peft_cfg.init_strategy = 'diag_zero'
    # config.model_cfg.peft_cfg.init_strategy = 'b_zero'

    # whole dataset
    max_sample = 0
    config.accelertor_cfg.gradient_accumulation_steps = 8
    config.train_batch_size = 2
    config.manipulator_cfg.n_adapt_step = 128

    # 1/2 dataset
    # # max_sample = 20000
    # max_sample = 15000
    # config.accelertor_cfg.gradient_accumulation_steps = 8
    # config.train_batch_size = 1
    # config.manipulator_cfg.n_adapt_step = 128

    if TEST:
        config.accelertor_cfg.gradient_accumulation_steps = 2
        config.n_save_step = 1
        config.manipulator_cfg.n_adapt_step = 2

    config.parse_args()

    trainer = MTL_Trainer(
        config = config
    )
    
    config.base_dateset_cfg.tokenize_type = 'prompt_response'
    train_dataset = Instruct_MTL_Dataset(config.get_dataset_cfgs())
    train_dataset.load(mode = 'train', max_sample = max_sample if not TEST else 60)
    trainer.train(
        train_dataset = train_dataset.get_generator(),
        # eval_dataset = None,
        train_batch_size = config.train_batch_size,
        n_episode = config.n_episode
    )

if __name__ == '__main__':
    
    main()
