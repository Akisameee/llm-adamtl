import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import deque, namedtuple
from einops.layers.torch import Rearrange
from functools import partial
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig
from accelerate import Accelerator
import tqdm
import os

from configs import MOPPO_Config, SVD_Lora_Config, SVD_Lora_Altered_Config
from modules.ppo import PPO_Trainer, PPOMemory, pad_sequence_fixed
from modules.manipulators import (
    Base_MTL_Manipulator,
    Base_MO_Manipulator,
    MOE_Manipulator_Altered
)
from modules.pefts import SVD_Lora_Linear, SVD_Lora_Linear_Altered
from modules.base import BaseLMWithValueHeads
from modules.utils import masked_mean, ExperienceDataset, shift, log_prob, default, masked_whiten
from logger import Logger

manipulator_map = {
    None: Base_MTL_Manipulator,
    'ls': Base_MTL_Manipulator,
    'sils': Base_MTL_Manipulator,
    'mols': Base_MO_Manipulator,
    'mosils': Base_MO_Manipulator
}

class MOPPO_Trainer(PPO_Trainer):

    def __init__(
        self,
        config: MOPPO_Config,
        pref_dim: int,
        model: AutoModelForCausalLMWithValueHead | BaseLMWithValueHeads,
        tokenizer: AutoTokenizer = None,
        optimizer: torch.optim.Optimizer = None,
        accelerator: Accelerator = None,
        logger: Logger = None
    ):
        self.pref_dim = pref_dim
        super().__init__(
            config = config,
            model = model,
            tokenizer = tokenizer,
            optimizer = optimizer,
            accelerator = accelerator,
            logger = logger
        )

        manipulator_type = manipulator_map[config.manipulator_cfg.weighted_loss_type]
        self.is_alted = isinstance(config.model_cfg.peft_cfg, SVD_Lora_Altered_Config)
        manipulator_type = MOE_Manipulator_Altered \
            if self.is_alted and manipulator_type == Base_MO_Manipulator else manipulator_type
        
        self.manipulator = manipulator_type(
            model = self.model,
            accelerator = self.accelerator,
            optimizer = self.optimizer,
            logger = self.logger,
            pref_dim = self.pref_dim,
            max_norm = self.max_norm,
            **config.manipulator_cfg.to_dict()
        )

    def set_pref_vec(
        self,
        pref_vec
    ):
        for module in self.model.modules():
            if isinstance(module, (SVD_Lora_Linear, SVD_Lora_Linear_Altered)):
                module.set_pref_vec(pref_vec)

        self.manipulator.set_pref_vec(pref_vec)

    @torch.no_grad()
    def get_all_logprobs(
        self,
        sequences: torch.LongTensor,
        masks: torch.LongTensor
    ):
        seq_dataloader = DataLoader(
            ExperienceDataset(
                [
                    sequences,
                    masks,
                ],
                device = self.device
            ),
            batch_size = self.train_batch_size,
            shuffle = False,
            drop_last = False
        )

        all_old_logprobs = []
        all_old_values = []
        all_ref_logprobs = []
        for sequences, masks in seq_dataloader:
            old_logits, old_values = self.batch_forward(
                sequences,
                mask = masks
            )
            old_logits = shift(old_logits, shift = 1, dim = -2)
            old_probs = old_logits.softmax(dim = -1)
            old_logprobs = log_prob(old_probs, sequences)
            old_values = shift(old_values, shift = 1, dim = -2)
            all_old_logprobs.append(old_logprobs)
            all_old_values.append(old_values)

            ref_logprobs = self.get_ref_logprobs(
                ref_model = self.ref_model,
                sequences = sequences,
                masks = masks
            )
            all_ref_logprobs.append(ref_logprobs)

        return (
            torch.cat(all_old_logprobs, dim = 0),
            torch.cat(all_old_values, dim = 0),
            torch.cat(all_ref_logprobs, dim = 0)
        )

    def step(
        self,
        memories: deque[PPOMemory]
    ):
        # stack all data stored in the memories
        (
            sequences,
            masks,
            action_masks,
            rm_rewards_pref,
        ) = self.pad_memories(memories)

        self.model.train()
        (
            old_logprobs,
            old_values_pref,
            ref_logprobs
        ) = self.get_all_logprobs(
            sequences = sequences,
            masks = masks
        )

        # prepare dataloader
        dataloader = DataLoader(
            ExperienceDataset(
                [
                    sequences,
                    masks,
                    action_masks,
                    old_logprobs,
                    ref_logprobs,
                    rm_rewards_pref,
                    old_values_pref,
                ],
                device = self.device
            ),
            batch_size = self.train_batch_size,
            shuffle = True
        )

        # PPO training
        multi_losses = []
        weighted_losses = []
        self.manipulator.clear()
        for _ in range(self.n_update_epoch):
            for (
                sequences,
                masks,
                action_masks,
                old_logprobs,
                ref_logprobs,
                rm_rewards_pref,
                old_values_pref,
            ) in dataloader:
                assert old_values_pref.size(-1) == rm_rewards_pref.size(-1)
                n_reward_dim = rm_rewards_pref.size(-1)
                
                logits, values_pref = self.batch_forward(
                    sequences,
                    mask = masks
                )

                logits = shift(logits, shift = 1, dim = -2)
                probs = logits.softmax(dim = -1)
                logprobs = log_prob(probs, sequences)

                values_pref = shift(values_pref, shift = 1, dim = -2)

                losses = []
                skip_batch = False
                for i in range(n_reward_dim):
                    values = values_pref[..., i]
                    rm_rewards = rm_rewards_pref[..., i]
                    old_values = old_values_pref[..., i]

                    rewards, kl_refs = self.compute_rewards(
                        rm_rewards = rm_rewards,
                        logprobs = old_logprobs,
                        ref_logprobs = ref_logprobs,
                        masks = action_masks
                    )

                    old_values, advantages, returns = self.compute_advantages(
                        old_values,
                        rewards,
                        action_masks
                    )

                    policy_loss, value_loss = self.loss(
                        logprobs,
                        old_logprobs,
                        advantages,
                        returns,
                        action_masks,
                        values,
                        old_values
                    )
                    if policy_loss.item() == 0 or value_loss.item() == 0:
                        skip_batch = True
                    
                    # combine losses
                    loss = policy_loss + value_loss
                    losses.append(loss)
                losses = torch.stack(losses, dim = 0)
                if skip_batch:
                    losses = losses * 0.0
                
                # update
                weighted_loss, losses = self.manipulator.backward(
                    losses = losses
                )
                multi_losses.append(losses.detach().cpu().tolist())
                weighted_losses.append(weighted_loss.item())

                self.manipulator.step()

        multi_losses = {f'losses_{i}': [losses[i] for losses in multi_losses] for i in range(n_reward_dim)}
        ppo_stats = self.get_train_stats(
            masks = action_masks,
            weighted_losses = weighted_losses,
            kl_refs = kl_refs,
            **multi_losses
        )
        
        return ppo_stats
    
    def compute_ref_rewards(
        self,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
    ):
        kl_refs = []
        for logprob, ref_logprob in zip(logprobs, ref_logprobs):
            kl_ref = -self.kl_penalty(logprob, ref_logprob) * self.kl_ref_coef
            kl_refs.append(kl_ref)
        
        return torch.stack(kl_refs)
    
    def compute_rm_rewards(
        self,
        rm_rewards: torch.FloatTensor,
        masks: torch.LongTensor
    ):
        rewards = []
        for rm_reward, mask in zip(rm_rewards, masks):
            reward = torch.zeros_like(mask)
            reward[mask.nonzero()[-1]] += rm_reward
            rewards.append(reward)
        
        return torch.stack(rewards)
    
    def get_train_stats(self, masks, **train_records):

        loss_keys = [key for key, value in train_records.items() if key.startswith('losses_')]
        train_stats = {}
        for loss_key in loss_keys:
            losses = train_records.pop(loss_key)
            loss_mean = sum(losses) / len(losses)
            train_stats[loss_key] = loss_mean

        weighted_losses = train_records.pop('weighted_losses')
        weighted_loss_mean = sum(weighted_losses) / len(weighted_losses)
        train_stats['weighted_loss'] = weighted_loss_mean

        kl_refs = train_records.pop('kl_refs')
        kl_ref_mean = masked_mean(kl_refs, mask = masks, dim = None).item()
        train_stats['ref_kl'] = kl_ref_mean

        response_len_mean = torch.sum(masks).item() / masks.shape[0]
        train_stats['generate_length'] = response_len_mean

        return train_stats