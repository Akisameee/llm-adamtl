import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange, repeat, reduce
from collections import deque
from functools import partial
from random import randrange
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead
from accelerate import Accelerator, dispatch_model
from accelerate.utils import broadcast
from transformers import PreTrainedModel
from accelerate import DistributedDataParallelKwargs
import numpy as np

from configs import RLHF_Config, LM_Config, RM_Config, Accelertor_Config, Trainer_Config
from modules.base import Base_Warpper
from modules.ppo import PPO_Trainer, PPOMemory
from modules.pefts import replace_peft_layers, set_all_adapters
from modules.utils import shift, log_prob, default, masked_mean, merge_dict, get_model
from logger import Logger

class Base_Evaluator(nn.Module):

    def __init__(
        self,
        config: Trainer_Config,
        accelerator_cfg: Accelertor_Config | Accelerator,
        model_cfg: LM_Config | Base_Warpper,
        res_dir: str = None,
        **model_kwargs,
    ):
        super().__init__()

        if isinstance(accelerator_cfg, Accelerator):
            self.accelerator = accelerator_cfg
        elif isinstance(accelerator_cfg, Accelertor_Config):
            # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
            self.accelerator = Accelerator(
                log_with = accelerator_cfg.log_with,
                gradient_accumulation_steps = accelerator_cfg.gradient_accumulation_steps,
                # kwargs_handlers=[ddp_kwargs]
            )
        else:
            raise TypeError(f'Invalid accelerator type \'{type(accelerator_cfg)}\'.')
        self.dispatch_models = self.accelerator.state.num_processes == 1 and torch.cuda.device_count() > 1

        self.logger = Logger(
            output_dir = config.output_dir if res_dir is None else res_dir,
            task_name = config.task_name,
            disable = not self.accelerator.is_main_process,
            make_dir = True if res_dir is None else False
        )
        self.logger.log_config(config, './eval_config.json')
        
        if isinstance(model_cfg, LM_Config):
            # models
            self.model, model_peft_info = get_model(
                config = model_cfg,
                dispatch = self.dispatch_models,
                **model_kwargs
            )
            if model_peft_info:
                self.logger.info(model_peft_info)
            
            # prepare with accelerator
            self.model = self.accelerator.prepare(self.model)
        elif isinstance(model_cfg, Base_Warpper):
            self.model = model_cfg
        else:
            raise TypeError(f'Invalid model type \'{type(model_cfg)}\'.')
        
        self.task_name = config.task_name
        self.model_name = model_cfg.model_name
        self.model_info = model_cfg.model_info
        self.generation_kwargs = self.model_info['generation_config'].to_dict()
        if 'max_new_tokens' in self.generation_kwargs.keys() and \
            'max_length' in self.generation_kwargs.keys():
            self.generation_kwargs.pop('max_length')
        self.clean_cache_every_iter = False

    def save(self, model, ckpt_dir = './output', wait_for_everyone = False, safetensor = False):

        if wait_for_everyone:
            self.accelerator.wait_for_everyone()
        if safetensor:
            self.accelerator.save_model(model, ckpt_dir)
        else:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            unwrapped_model = self.accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, 'checkpoint.pt'))

    def load(self, model, ckpt_path = './checkpoint.pt'):

        unwrapped_model = self.accelerator.unwrap_model(model)
        # print(torch.load(ckpt_path))
        unwrapped_model.load_state_dict(torch.load(ckpt_path, map_location = 'cpu'))

    @property
    def device(self):
        
        return self.accelerator.device