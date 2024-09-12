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
import math
import tqdm
import os

from configs import SafePPO_Config, SVD_Lora_Config, SVD_Lora_Altered_Config
from modules.ppo import PPO_Trainer, pad_sequence_fixed
from modules.pefts import SVD_Lora_Linear, SVD_Lora_Linear_Altered
from modules.base import BaseLMWithValueHeads
from modules.utils import masked_mean, ExperienceDataset, shift, log_prob, default, masked_whiten
from logger import Logger

SafePPOMemory = namedtuple(
    'Memory',
    [
        'sequence',
        'mask',
        'action_mask',
        'reward_score',
        'cost_score'
    ]
)

class SafePPO_Trainer(PPO_Trainer):

    def __init__(
        self,
        config: SafePPO_Config,
        model: AutoModelForCausalLMWithValueHead | BaseLMWithValueHeads,
        tokenizer: AutoTokenizer = None,
        optimizer: torch.optim.Optimizer = None,
        accelerator: Accelerator = None,
        logger: Logger = None
    ):
        super().__init__(
            config = config,
            model = model,
            tokenizer = tokenizer,
            optimizer = optimizer,
            accelerator = accelerator,
            logger = logger
        )

        self.cost_coef_lr = config.cost_coef_lr
        self.cost_coef = 1
        self.cost_avg = 0
        self.update_epoch = 0

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
        memories: deque[SafePPOMemory]
    ):
        
        # stack all data stored in the memories
        (
            sequences,
            masks,
            action_masks,
            reward_scores,
            cost_scores
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
                    reward_scores,
                    cost_scores,
                    old_values_pref
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
                reward_scores,
                cost_scores,
                old_values_pref
            ) in dataloader:
                
                logits, values_pref = self.batch_forward(
                    sequences,
                    mask = masks
                )

                logits = shift(logits, shift = 1, dim = -2)
                probs = logits.softmax(dim = -1)
                logprobs = log_prob(probs, sequences)
                
                values_pref = shift(values_pref.squeeze(), shift = 1, dim = -1)
                reward_values = values_pref[..., 0]
                cost_values = values_pref[..., 1]
                
                rewards, costs, kl_refs = self.compute_rewards(
                    reward_scores = reward_scores,
                    cost_scores = cost_scores,
                    logprobs = old_logprobs,
                    ref_logprobs = ref_logprobs,
                    masks = action_masks
                )

                old_values_pref = old_values_pref.squeeze()
                old_reward_values = old_values_pref[..., 0]
                old_cost_values = old_values_pref[..., 1]

                # Reward Loss
                old_reward_values, reward_advantages, reward_returns = self.compute_advantages(
                    values = old_reward_values,
                    rewards = rewards,
                    mask = action_masks
                )

                policy_loss, value_loss = self.loss(
                    logprobs,
                    old_logprobs,
                    reward_advantages,
                    reward_returns,
                    action_masks,
                    reward_values,
                    old_reward_values
                )

                # combine losses
                reward_loss = policy_loss + value_loss

                # Cost Loss
                old_cost_values, cost_advantages, cost_returns = self.compute_advantages(
                    values = old_cost_values,
                    rewards = costs,
                    mask = action_masks
                )

                policy_loss, value_loss = self.loss(
                    logprobs,
                    old_logprobs,
                    cost_advantages,
                    cost_returns,
                    action_masks,
                    cost_values,
                    old_cost_values
                )

                # combine losses
                cost_loss = policy_loss + value_loss
            losses = torch.stack([reward_loss, cost_loss], dim = 0)
            
            weighted_loss = (reward_loss + self.cost_coef * cost_loss) / (1 + self.cost_coef)
            self.update_cost_lambda(cost_score = cost_scores)
            # update
            self.manipulator.backward(
                losses = weighted_loss
            )
            self.manipulator.step()
            multi_losses.append(losses.detach().cpu().tolist())
            weighted_losses.append(weighted_loss.item())

        multi_losses = {
            f'losses_reward': [losses[0] for losses in multi_losses],
            f'losses_cost': [losses[1] for losses in multi_losses]
        }

        ppo_stats = self.get_train_stats(
            masks = action_masks,
            weighted_losses = weighted_losses,
            kl_refs = kl_refs,
            **multi_losses
        )
        
        return ppo_stats
    
    def compute_rewards(
        self,
        reward_scores: torch.FloatTensor,
        cost_scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor
    ):
        
        kl_refs = []
        rewards = []
        costs = []
        for (
            reward_score,
            cost_score,
            logprob,
            ref_logprob,
            mask
        ) in zip(reward_scores, cost_scores, logprobs, ref_logprobs, masks):
            kl_ref = -self.kl_penalty(logprob, ref_logprob) * self.kl_ref_coef
            reward = kl_ref.clone() / 2
            reward[mask.nonzero()[-1]] += reward_score
            cost = -kl_ref.clone() / 2
            cost[mask.nonzero()[-1]] += cost_score
            kl_refs.append(kl_ref)
            rewards.append(reward)
            costs.append(cost)
        
        return torch.stack(rewards), torch.stack(costs), torch.stack(kl_refs)

    def update_cost_lambda(
        self,
        cost_score: torch.FloatTensor
    ):
        
        cost_score = self.accelerator.gather(cost_score).mean().detach().cpu().item()
        self.cost_avg = (self.cost_avg * self.update_epoch - cost_score) / (self.update_epoch + 1)
        self.update_epoch += 1

        self.cost_coef = math.exp(math.log(self.cost_coef) + self.cost_coef_lr * self.cost_coef * self.cost_avg)
        self.logger.info(f'cost_lambda:{self.cost_coef}')
    
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