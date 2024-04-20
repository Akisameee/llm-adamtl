import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange, repeat, reduce
from collections import deque
from functools import partial
from random import randrange
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead
from accelerate import Accelerator
from accelerate.utils import broadcast

from base_trainer import Base_Trainer
from datas.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import Panacea_PPO_Config, RM_Config, model_infos
# from modules.lms import BaseLM, RewardLM
from modules.base import BaseLMWithValueHeads
from modules.ppo import PPO_Trainer, PPOMemory
from modules.moppo import MOPPO_Trainer
from modules.pefts import set_all_adapters
from modules.utils import shift, log_prob, default, masked_mean, merge_dict, get_model
from logger import Logger
from trl.core import LengthSampler

class MORLHF_Trainer(Base_Trainer):

    def __init__(
        self,
        config: Panacea_PPO_Config
    ):
        config.model_cfg.model_class = BaseLMWithValueHeads

        self.reward_names = []
        self.reward_cfgs = []
        for i in range(3):
            reward_cfg = getattr(config, f'reward_cfg_{i}')
            reward_name = getattr(config, f'reward_name_{i}')
            if reward_cfg is not None and reward_name is not None:
                self.reward_cfgs.append(reward_cfg)
                self.reward_names.append(reward_name)

        self.pref_dim = len(self.reward_names)
        if self.pref_dim <= 1:
            raise ValueError(f'Expected pref_dim > 1, but got pref_dim = {self.pref_dim}')

        super().__init__(
            config = config,
            accelerator_cfg = config.accelertor_cfg,
            model_cfg = config.model_cfg,
            ref_cfg = config.ref_cfg,
            reward_cfg = self.reward_cfgs,
            model_kwargs = dict(n_v_head = self.pref_dim)
        )
        
        self.ppo_trainer = MOPPO_Trainer(
            config = config,
            pref_dim = self.pref_dim,
            model = self.model,
            optimizer = self.optimizer,
            accelerator = self.accelerator,
            logger = self.logger
        )
        
        self.retokenization = config.retokenization
        self.n_sample_reuse = config.n_sample_reuse
        self.scalariztion_type = config.scalariztion_type
        self.kl_ref_coef = config.kl_ref_coef
        self.n_save_time = config.n_save_time
        
    def sample_pref_vec(self):
        
        pref_vec = torch.rand(self.pref_dim).to(self.device)
        pref_vec = pref_vec / torch.sum(pref_vec)
        # print('unbroadcasted', pref_vec)
        pref_vec = broadcast(pref_vec)
        # print('broadcasted', pref_vec)

        return pref_vec
    
    def set_pref_vec(
        self,
        pref_vec: torch.FloatTensor
    ):
        self.ppo_trainer.set_pref_vec(
            pref_vec = pref_vec
        )

        # self.logger.info(f'pref_vec set to {pref_vec}')

    @torch.no_grad()
    def get_ref_logprobs(
        self,
        ref_model: nn.Module,
        sequences: torch.LongTensor,
        masks: torch.LongTensor
    ):
        
        if ref_model is None:
            # disable peft layers, switch to reference model
            set_all_adapters(
                model = self.ppo_trainer.model,
                enable = False
            )
            ref_logits, _ = self.ppo_trainer.batch_forward(
                sequences,
                mask = masks
            )
            set_all_adapters(
                model = self.ppo_trainer.model,
                enable = True
            )
        else:
            ref_logits, _, _ = ref_model(
                input_ids = sequences,
                attention_mask = masks
            )
        ref_logits = shift(ref_logits, shift = 1, dim = -2)
        ref_probs = ref_logits.softmax(dim = -1)
        ref_logprobs = log_prob(ref_probs, sequences)

        return ref_logprobs
        
    @torch.no_grad()
    def get_rm_rewards(
        self,
        reward_model: nn.Module,
        sequences: torch.FloatTensor,
        masks: torch.LongTensor,
        action_masks: torch.LongTensor,
        verbose: bool = False
    ):
        
        prompt_masks = (1 - action_masks) * masks
        prompt_texts, response_texts = self.ppo_trainer.decode_batch(
            inputs_ids = sequences,
            attention_masks = masks,
            prompt_masks = prompt_masks
        )
        rm_rewards = reward_model.get_rewards(
            prompts = prompt_texts,
            responses = response_texts
        )
        if verbose:
            self.logger.info(
                '-' * 50 + '\n' + \
                f'Prompt:{prompt_texts[0]}\n' + \
                '-' * 50 + '\n' + \
                f'Response:{response_texts[0]}\n' + \
                '-' * 50 + '\n' + \
                f'RM_Score:{rm_rewards[0]}'
            )

        return rm_rewards
    
    def reward_scalariztion(
        self,
        rms_rewards: list[torch.FloatTensor],
        pref_vec: torch.FloatTensor
    ):  
        
        rm_rewards_scalar = torch.zeros_like(rms_rewards[0])
        if self.scalariztion_type is None:
            rm_rewards_scalar = torch.stack(rms_rewards, dim = -1)
        elif self.scalariztion_type == 'ls':
            for rm_rewards, pref_weight in zip(rms_rewards, pref_vec):
                rm_rewards_scalar += rm_rewards * pref_weight
        else:
            raise NotImplementedError
        
        return rm_rewards_scalar

    def compute_action_log_prob(
        self,
        action_len,
        action_logits,
        sequence
    ):
        # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token
        action_logits = shift(action_logits, shift = 1, dim = -2)
        action_prob = action_logits.softmax(dim = -1)

        action_log_prob = log_prob(action_prob, sequence)
        action_log_prob = action_log_prob[:, -action_len:]

        return action_log_prob, action_prob

    def train_ppo(
        self,
        ds_generator,
        n_episode: int = 10,
        n_update_timestep: int = 8,
        sample_batch_size: int = 8,
    ):
        
        set_all_adapters(
            model = self.ppo_trainer.model,
            enable = True
        )

        dataloader = DataLoader(
            dataset = ds_generator,
            batch_size = sample_batch_size,
            shuffle = True,
            collate_fn = instruct_collator,
            drop_last = True
        )
        dataloader = self.accelerator.prepare(dataloader)
        max_timestep = len(dataloader) * sample_batch_size * n_episode

        memories = deque([])

        timestep = 0
        updatestep = 0
        sample_records = []
        pref_vec = self.sample_pref_vec()
        self.set_pref_vec(pref_vec)
        tqdm_bar = tqdm(
            total = max_timestep // n_update_timestep * n_update_timestep // sample_batch_size,
            disable = not self.accelerator.is_main_process
        )
        
        self.get_save_timesteps(self.n_save_time, tqdm_bar.total)
        for episode in range(n_episode):
            for prompts_ids, attention_masks, prompt_texts in dataloader:
                
                self.ppo_trainer.model.eval()
                batch_size = len(prompts_ids)
                timestep += batch_size
                sample_record = {}

                (
                    sequences,
                    masks,
                    action_masks
                ) = self.ppo_trainer.generate_batch(
                    prompts_ids = prompts_ids,
                    attention_masks = attention_masks,
                    # length_sampler = length_sampler,
                    return_padding = True,
                    **self.generation_config.to_dict()
                )

                with torch.no_grad():
                    logits, values = self.ppo_trainer.batch_forward(
                        sequences,
                        mask = masks
                    )
                    logits = shift(logits, shift = 1, dim = -2)
                    probs = logits.softmax(dim = -1)
                    logprobs = log_prob(probs, sequences)

                ref_logprobs = self.get_ref_logprobs(
                    ref_model = self.ref_model,
                    sequences = sequences,
                    masks = masks
                )

                rms_rewards = []
                
                for idx, reward_model in enumerate(self.reward_models):
                    rm_rewards = self.get_rm_rewards(
                        reward_model = reward_model,
                        sequences = sequences,
                        masks = masks,
                        action_masks = action_masks,
                        verbose = timestep % n_update_timestep == 0
                        # verbose = True
                    )
                    rms_rewards.append(rm_rewards)
                    sample_record[self.reward_names[idx]] = torch.sum(rm_rewards).item()
                sample_records.append(sample_record)
                rm_rewards_scalarized = self.reward_scalariztion(
                    rms_rewards = rms_rewards,
                    pref_vec = pref_vec
                )

                detach_to_cpu_ = lambda t: t.detach().cpu()

                for (
                    sequence,
                    mask,
                    action_mask,
                    prob,
                    logprob,
                    ref_logprob,
                    rm_reward_scalarized,
                    value
                ) in zip(
                    sequences,
                    masks,
                    action_masks,
                    probs,
                    logprobs,
                    ref_logprobs,
                    rm_rewards_scalarized,
                    values
                ):
                    seq_len = torch.sum(mask).item()
                    memories.append(PPOMemory(*map(detach_to_cpu_, (
                        sequence[: seq_len],
                        mask[: seq_len],
                        action_mask[: seq_len],
                        prob[: seq_len, :],
                        logprob[: seq_len],
                        ref_logprob[: seq_len],
                        rm_reward_scalarized,
                        value[: seq_len]
                    ))))

                if self.clean_cache_every_iter:
                    del (
                        sequences,
                        masks,
                        action_masks,
                        probs,
                        logprobs,
                        ref_logprobs,
                        rm_rewards,
                        values
                    )
                    torch.cuda.empty_cache()
                
                tqdm_bar.update(1)
                if timestep >= (updatestep + 1) * n_update_timestep:
                    # try:
                    updatestep += 1
                    self.accelerator.wait_for_everyone()
                    ppo_stats = [self.ppo_trainer.learn(memories)]
                    torch.cuda.empty_cache()

                    ppo_stats_gathered = self.accelerator.gather_for_metrics(ppo_stats)
                    sample_records_gathered = self.accelerator.gather_for_metrics(sample_records)
                    all_ppo_stats = merge_dict(unmerged_dicts = ppo_stats_gathered, reduce = 'mean')
                    all_sample_records = merge_dict(unmerged_dicts = sample_records_gathered, reduce = 'mean')
                    all_sample_records['pref_vec'] = pref_vec.cpu()
                    all_ppo_stats.update(all_sample_records)

                    if self.accelerator.is_main_process:
                        self.logger.step(
                            episode = episode + 1,
                            timestep = timestep,
                            stat_dict = all_ppo_stats
                        )

                    while len(memories) > (self.n_sample_reuse - 1) * n_update_timestep:
                        memories.popleft()
                    sample_records.clear()

                    if self.check_if_save(tqdm_bar.n):
                        if self.accelerator.is_main_process:
                            self.save(
                                self.ppo_trainer.model,
                                os.path.join(self.logger.dir, f'{self.model_name}_{episode}_{timestep}')
                            )

                    if max_timestep - timestep < n_update_timestep:
                        break
                    else:
                        pref_vec = self.sample_pref_vec()
                        self.set_pref_vec(pref_vec)  

        self.accelerator.wait_for_everyone()
        self.logger.info('Panacea Training Complete')
        self.logger.save_res()
        self.logger.save_pareto_front(
            tuple(self.reward_names),
            vecs_name = 'pref_vec'
        )

def main():

    config = Panacea_PPO_Config()

    config.scalariztion_type = None

    data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    # sub_data_path = ['helpful-base', 'harmless-base']
    sub_data_path = ['harmless-base']
    config.dateset_cfg.data_path = data_path
    config.dateset_cfg.sub_data_path = sub_data_path

    model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    config.dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path
    config.ref_cfg.model_pretrain_path = model_path
    
    rm_path_1 = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
    rm_path_2 = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-harmless-reward_model')
    config.reward_cfg_0 = RM_Config(model_pretrain_path = rm_path_1)
    config.reward_cfg_1 = RM_Config(model_pretrain_path = rm_path_2)
    config.reward_name_0 = 'helpful'
    config.reward_name_1 = 'harmless'
    config.parse_args()

    trainer = MORLHF_Trainer(
        config = config
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    dataset = Instruct_Dataset(config.dateset_cfg)
    dataset.load(mode = 'train')
    trainer.train_ppo(
        ds_generator = dataset.get_generator(),
        n_episode = config.n_episode,
        n_update_timestep = config.n_update_timestep,
        sample_batch_size = config.sample_batch_size
    )

if __name__ == '__main__':
    
    main()
