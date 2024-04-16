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
from transformers import PreTrainedModel

from datas.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import RLHF_Config, LM_Config, RM_Config, Accelertor_Config, Trainer_Config
# from modules.lms import BaseLM, RewardLM, get_model
from modules.ppo import PPO_Trainer, PPOMemory
from modules.pefts import replace_peft_layers, set_all_adapters
from modules.utils import shift, log_prob, default, masked_mean, merge_dict, get_model
from logger import Logger
from trl.core import LengthSampler

class Base_Trainer(nn.Module):

    def __init__(
        self,
        config: Trainer_Config,
        accelerator_cfg: Accelertor_Config | Accelerator,
        model_cfg: LM_Config,
        ref_cfg: LM_Config = None,
        reward_cfg: RM_Config | list = None,
        model_kwargs: dict = None,
    ):
        super().__init__()

        if isinstance(accelerator_cfg, Accelerator):
            self.accelerator = accelerator_cfg
        elif isinstance(accelerator_cfg, Accelertor_Config):
            self.accelerator = Accelerator(
                log_with = accelerator_cfg.log_with,
                gradient_accumulation_steps = accelerator_cfg.gradient_accumulation_steps
            )
        self.dispatch_models = self.accelerator.state.num_processes == 1 and torch.cuda.device_count() > 1

        self.logger = Logger(
            output_dir = config.output_dir,
            task_name = config.task_name,
            disable = not self.accelerator.is_main_process
        )
        self.logger.info(config.get_args_info())
        
        # models
        self.model, model_peft_info = get_model(
            config = model_cfg,
            dispatch = self.dispatch_models,
            **model_kwargs
        )
        if model_peft_info:
            self.logger.info(model_peft_info)
        
        if reward_cfg is not None:
            if isinstance(reward_cfg, RM_Config):
                reward_model, _ = get_model(
                    config = reward_cfg,
                    dispatch = self.dispatch_models
                )
                reward_model.set_freeze(freeze=True)
                reward_model.eval()
                reward_models = None
            elif isinstance(reward_cfg, list):
                reward_models = []
                for reward_cfg_single in reward_cfg:
                    reward_model, _ = get_model(
                        config = reward_cfg_single,
                        dispatch = self.dispatch_models
                    )
                    reward_model.set_freeze(freeze=True)
                    reward_model.eval()
                    reward_models.append(reward_model)
                reward_model = None
            else:
                raise NotImplementedError
        else:
            reward_model = None
            reward_models = None
        
        if ref_cfg is not None:
            if model_peft_info is not None:
                ref_model = None
                self.logger.info(f'Disabling target model peft layers as reference model.')
            else:
                ref_model, _ = get_model(
                    config = ref_cfg,
                    dispatch = self.dispatch_models
                )
                ref_model.set_freeze(freeze=True)
                ref_model.eval()
        else:
            ref_model = None
                
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = config.lr,
            weight_decay = config.weight_decay
        )

        # prepare with accelerator
        (
            self.model,
            self.optimizer
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer
        )
        if ref_model is not None:
            self.ref_model = self.accelerator.prepare(ref_model)
        else:
            self.ref_model = None
        if reward_model is not None:
            self.reward_model = self.accelerator.prepare(reward_model)
        else:
            self.reward_model = None
        if reward_models is not None:
            self.reward_models = []
            for reward_model in reward_models:
                reward_model = self.accelerator.prepare(reward_model)
                self.reward_models.append(reward_model)
        else:
            self.reward_models = None
        
        self.model_name = model_cfg.model_name
        self.model_info = model_cfg.model_info
        self.generation_config = self.model_info['generation_config']
        self.clean_cache_every_iter = False

    def save(self, model, ckpt_dir = './checkpoint.pt', wait_for_everyone = False, safetensor = False):

        if wait_for_everyone:
            self.accelerator.wait_for_everyone()
        if safetensor:
            self.accelerator.save_model(model, ckpt_dir)
        else:
            os.mkdir(ckpt_dir)
            unwrapped_model = self.accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, 'checkpoint.pt'))

    def load(self, model, ckpt_dir = './checkpoint.pt'):

        unwrapped_model = self.accelerator.unwrap_model(model)
        path_to_checkpoint = os.path.join(ckpt_dir, "pytorch_model.bin")
        unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))

    @property
    def device(self):
        
        return self.accelerator.device