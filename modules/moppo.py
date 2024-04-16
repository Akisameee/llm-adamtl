import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import deque, namedtuple
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from beartype.typing import List, Optional, Callable, Deque
from functools import partial
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig
from accelerate import Accelerator
import tqdm
import os

from configs import PPO_Config, model_infos
from modules.ppo import PPO_Trainer, PPOMemory, pad_sequence_fixed
from modules.pefts import Panacea_SVD_Linear
from modules.base import BaseLMWithValueHeads
from modules.manipulators import (
    Linear_Scalarization,
    ScaleInvariant_Linear_Scalarization
)
from modules.utils import masked_mean, ExperienceDataset, shift, log_prob, default, masked_whiten
from logger import Logger

class MOPPO_Trainer(PPO_Trainer):

    def __init__(
        self,
        config: PPO_Config,
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

        self.manipulator = Linear_Scalarization(
            model = self.model,
            pref_dim = self.pref_dim,
            accelerator = self.accelerator
        )

    def set_pref_vec(
        self,
        pref_vec
    ):
        for module in self.model.modules():
            if isinstance(module, Panacea_SVD_Linear):
                module.set_pref_vec(pref_vec)

        self.manipulator.set_pref_vec(pref_vec)

    def learn(
        self,
        memories: Deque[PPOMemory]
    ):
        # stack all data stored in the memories
        (
            sequences,
            masks,
            all_action_masks,
            old_probs,
            old_logprobs,
            ref_logprobs,
            rm_rewards_pref,
            old_values_pref
        ) = list(map(partial(pad_sequence_fixed, batch_first = True), zip(*memories)))

        old_values_pref_new = []
        advantages_pref = []
        returns_pref = []
        for i in range(self.pref_dim):
            
            rm_rewards = rm_rewards_pref[..., i]
            old_values = old_values_pref[..., i]
            rewards, kl_refs = self.compute_rewards(
                rm_rewards = rm_rewards,
                logprobs = old_logprobs,
                ref_logprobs = ref_logprobs,
                masks = all_action_masks
            )

            old_values = shift(old_values, shift = 1, dim = -2)
            old_values, advantages, returns = self.compute_advantages(
                old_values,
                rewards,
                all_action_masks
            )
            old_values_pref_new.append(old_values)
            advantages_pref.append(advantages)
            returns_pref.append(returns)
        old_values_pref = torch.stack(old_values_pref_new, dim = -1)
        advantages_pref = torch.stack(advantages_pref, dim = -1)
        returns_pref = torch.stack(returns_pref, dim = -1)

        # prepare dataloader
        dataloader = DataLoader(
            ExperienceDataset(
                [
                    sequences,
                    masks,
                    all_action_masks,
                    old_logprobs,
                    old_values_pref,
                    returns_pref,
                    advantages_pref,
                ],
                # device = self.device
            ),
            batch_size = self.train_batch_size,
            shuffle = True
        )
        dataloader = self.accelerator.prepare(dataloader)

        self.model.train()

        # PPO training
        weighted_losses = []
        for _ in range(self.n_update_epoch):
            for (
                sequences,
                masks,
                action_masks,
                old_logprobs,
                old_values_pref,
                returns_pref,
                advantages_pref,
            ) in dataloader:
                
                logits, values_pref = self.batch_forward(
                    sequences,
                    mask = masks
                )

                logits = shift(logits, shift = 1, dim = -2)
                probs = logits.softmax(dim = -1)
                logprobs = log_prob(probs, sequences)

                values_pref = shift(values_pref, shift = 1, dim = -2)

                losses = []
                for i in range(self.pref_dim):
                    advantages = advantages_pref[..., i]
                    returns = returns_pref[..., i]
                    values = values_pref[..., i]
                    old_values = old_values_pref[..., i]
                    policy_loss, value_loss = self.loss(
                        logprobs,
                        old_logprobs,
                        advantages,
                        returns,
                        action_masks,
                        values,
                        old_values
                    )
                    
                    # combine losses
                    loss = policy_loss + value_loss
                    losses.append(loss)
                losses = torch.stack(losses, dim = 0)
                # update
                # self.accelerator.backward(loss)
                weighted_loss = self.manipulator.backward(
                    losses = losses
                )
                weighted_losses.append(weighted_loss.item())

                if self.max_norm is not None and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model_params, self.max_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

        ppo_stats = self.get_train_stats(
            masks = all_action_masks,
            weighted_losses = weighted_losses,
            kl_refs = kl_refs
        )
        
        return ppo_stats
    
    def get_train_stats(self, masks, **train_records):

        weighted_losses = train_records.pop('weighted_losses')
        weighted_loss_mean = sum(weighted_losses) / len(weighted_losses)

        kl_refs = train_records.pop('kl_refs')
        kl_ref_mean = masked_mean(kl_refs, mask = masks, dim = None).item()

        response_len_mean = torch.sum(masks).item() / masks.shape[0]

        train_stats = {
            'weighted_loss': weighted_loss_mean,
            'ref_kl': kl_ref_mean,
            'generate_length': response_len_mean
        }

        return train_stats