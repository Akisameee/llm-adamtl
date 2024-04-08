import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import deque, namedtuple
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from typing import Dict
from functools import partial
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from accelerate import Accelerator
from tqdm import tqdm
import math

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import DPO_Config
from data import HumanFeedback_Dataset, hf_collator
from modules.lms import BaseLM, RewardLM
from modules.base import BaseLMWithValueHead
from modules.peft import replace_peft_layers
from modules.utils import masked_mean, ExperienceDataset, shift, log_prob, default, masked_whiten, pad_to_length
from logger import Logger

class DPO_Trainer(nn.Module):

    def __init__(
        self,
        config: DPO_Config,
        model: AutoModelForCausalLM = None,
        ref_model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        optimizer: torch.optim.Optimizer = None,
        accelerator: Accelerator = None,
        logger: Logger = None
    ):
        super().__init__()
        
        self.model = model
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(config.model_cfg.model_pretrain_path)
            if config.model_cfg.peft_cfg is not None:
                self.model, peft_info = replace_peft_layers(
                    model = model,
                    peft_config = config.model_cfg.peft_cfg,
                    return_info = True
                )
        else:
            peft_info = None

        self.ref_model = ref_model
        if self.ref_model is None:
            self.ref_model = AutoModelForCausalLM.from_pretrained(config.ref_cfg.model_pretrain_path)
        self.set_freeze(self.ref_model, freeze = True)
        self.ref_model = self.ref_model.eval()

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_cfg.model_pretrain_path)

        # optimizers
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr = config.lr,
                weight_decay = config.weight_decay
            )

        # accelerator
        self.accelerator = accelerator
        if self.accelerator is None:
            self.accelerator = Accelerator(
                log_with = config.accelertor_cfg.log_with,
                gradient_accumulation_steps = config.accelertor_cfg.gradient_accumulation_steps
            )
        
        (
            self.model,
            self.ref_model,
            self.optimizer
        ) = self.accelerator.prepare(
            self.model,
            self.ref_model,
            self.optimizer
        )

        self.logger = logger
        if self.logger is None:
            self.logger = Logger(
                config.output_dir,
                task_name = 'DPO_train',
                disable = not self.accelerator.is_local_main_process
            )           

        self.pad_token_id = config.dateset_cfg.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.pad_token_id
        self.label_pad_token_id = config.dateset_cfg.label_pad_token_id
        self.beta = config.beta
        self.label_smoothing = config.label_smoothing
        self.loss_type = config.loss_type

        self.ckpt_path = config.model_cfg.ckpt_path

        if peft_info:
            self.logger.info(peft_info)

    def save(self, save_path = None):

        if save_path is None:
            save_path = self.ckpt_path
        if self.accelerator is not None:
            if self.accelerator.is_local_main_process:
                self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)

    def set_freeze(self, model: nn.Module, freeze: bool):

        for p in model.parameters():
            p.requires_grad = not freeze
    
    @property
    def device(self):
        if not getattr(self.model, 'is_sequential_parallel', False):
            return self.accelerator.device
        else:
            return self.model.device
        
    def pad_inputs_batch(
        self,
        batch: Dict[str, torch.LongTensor],
        pad_token_id: int,
        label_pad_token_id: int
    ):
        input_ids = batch['chosen_input_ids'] + batch['rejected_input_ids']
        attention_mask = batch['chosen_attention_mask'] + batch['rejected_attention_mask']
        labels = batch['chosen_labels'] + batch['rejected_labels']

        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first = True, padding_value = 0)
        labels = pad_sequence(labels, batch_first = True, padding_value = label_pad_token_id)

        return input_ids, attention_mask, labels

    def logprobs_from_logits(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        masks: torch.LongTensor,
    ):
        
        logits = shift(logits, shift = 1, dim = -2)
        probs = torch.softmax(logits, dim = -1)
        labels = labels * masks
        logprobs = torch.gather(probs, dim = 2, index = labels.unsqueeze(-1)).squeeze(-1)

        return masked_mean(logprobs, mask = masks, dim = -1)
    
    def batch_forward_logprobs(
        self,
        model: nn.Module,
        input_ids: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
        label_masks: torch.LongTensor
    ):
        
        batch_size = int(input_ids.shape[0] / 2)

        logits = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).logits

        logprobs = self.logprobs_from_logits(
            logits = logits,
            labels = labels,
            masks = label_masks
        )
        chosen_logprobs = logprobs[: batch_size]
        rejected_logprobs = logprobs[batch_size:]

        return chosen_logprobs, rejected_logprobs

    @torch.no_grad()
    def eval_step(
        self,
        eval_dataset: HumanFeedback_Dataset,
        eval_batch_size: int
    ):

        self.model.eval()
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size = eval_batch_size,
            shuffle = False,
            collate_fn = hf_collator,
            drop_last = True
        )
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        eval_records = []
        for batch in eval_dataloader:
            eval_record = {}
            (
                input_ids,
                attention_mask,
                labels
            ) = self.pad_inputs_batch(
                batch = batch,
                pad_token_id = self.pad_token_id,
                label_pad_token_id = self.label_pad_token_id
            )
            label_masks = labels != self.label_pad_token_id

            # get model logprobs
            chosen_logprobs, rejected_logprobs = self.batch_forward_logprobs(
                model = self.model,
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
                label_masks = label_masks
            )

            # get reference model logprobs
            with torch.no_grad():
                ref_chosen_logprobs, ref_rejected_logprobs = self.batch_forward_logprobs(
                    model = self.ref_model,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels,
                    label_masks = label_masks
                )

            # compute loss
            loss, chosen_reward, rejected_reward = self.loss(
                chosen_logprobs = chosen_logprobs,
                rejected_logprobs = rejected_logprobs,
                ref_chosen_logprobs = ref_chosen_logprobs,
                ref_rejected_logprobs = ref_rejected_logprobs
            )
            eval_record['eval_loss'] = loss.item()
            eval_record['eval_chosen_reward'] = chosen_reward.item()
            eval_record['eval_rejected_reward'] = rejected_reward.item()
            eval_records.append(eval_record)

        return eval_records

    def train(
        self,
        train_dataset: HumanFeedback_Dataset,
        eval_dataset: HumanFeedback_Dataset,
        n_epoch: int,
        n_eval_step: int,
        train_batch_size: int,
        eval_batch_size: int
    ):
        
        self.model.train()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size = train_batch_size,
            shuffle = True,
            collate_fn = hf_collator,
            drop_last = True
        )
        
        train_dataloader = self.accelerator.prepare(train_dataloader)

        timestep = 0
        n_eval = 0
        train_records = []
        dpo_bar = tqdm(
            total = len(train_dataloader) * n_epoch,
            disable = not self.accelerator.is_local_main_process
        )
        for epoch in range(n_epoch):
            for batch in train_dataloader:
                train_record = {}

                (
                    input_ids,
                    attention_mask,
                    labels
                ) = self.pad_inputs_batch(
                    batch = batch,
                    pad_token_id = self.pad_token_id,
                    label_pad_token_id = self.label_pad_token_id
                )
                label_masks = labels != self.label_pad_token_id
                batch_size = int(input_ids.shape[0] / 2)
                timestep += batch_size

                # get model logprobs
                chosen_logprobs, rejected_logprobs = self.batch_forward_logprobs(
                    model = self.model,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels,
                    label_masks = label_masks
                )

                # get reference model logprobs
                with torch.no_grad():
                    ref_chosen_logprobs, ref_rejected_logprobs = self.batch_forward_logprobs(
                        model = self.ref_model,
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels,
                        label_masks = label_masks
                    )

                # compute loss
                loss, chosen_reward, rejected_reward = self.loss(
                    chosen_logprobs = chosen_logprobs,
                    rejected_logprobs = rejected_logprobs,
                    ref_chosen_logprobs = ref_chosen_logprobs,
                    ref_rejected_logprobs = ref_rejected_logprobs
                )
                train_record['train_loss'] = loss.item()
                train_record['train_chosen_reward'] = chosen_reward.item()
                train_record['train_rejected_reward'] = rejected_reward.item()
                train_records.append(train_record)

                self.accelerator.backward(loss)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                dpo_bar.update(1)

                if timestep >= (n_eval + 1) * n_eval_step:
                    n_eval += 1
                    self.accelerator.wait_for_everyone()
                    eval_records = self.eval_step(
                        eval_dataset = eval_dataset,
                        eval_batch_size = eval_batch_size
                    )
                    # print(f'Process {os.getpid()} eval result len = {len(eval_records)}.')
                    train_records_gathered = self.accelerator.gather_for_metrics(train_records)
                    eval_stats_gathered = self.accelerator.gather_for_metrics(eval_records)
                    if self.accelerator.is_local_main_process:
                        # print(f'Main Proc train_records_gathered len = {len(train_records_gathered)}')
                        train_stats = self.get_stats(
                            train_records = train_records_gathered
                        )
                        eval_stats = self.get_stats(
                            eval_records = eval_stats_gathered
                        )
                        train_stats.update(eval_stats)
                        self.logger.step(
                            episode = epoch,
                            timestep = timestep,
                            stat_dict = train_stats
                        )
                    train_records.clear()

        self.logger.info('DPO Training Complete')
        self.logger.save_res()

    def get_stats(
        self,
        **records
    ) -> dict:
        new_records = {}
        for key, value in records.items():
            if key.endswith('_records'):
                if isinstance(value, list):
                    rec_keys = value[0].keys()
                    new_records = {k: [v[k] for v in value] for k in rec_keys}
        records.update(new_records)
        stats = {}
        for rec_key, rec_value in records.items():
            if rec_key.endswith('_loss'):
                prefix = rec_key.removesuffix('_loss')
                loss_mean = sum(rec_value) / len(rec_value)
                stats[f'{prefix}_loss'] = loss_mean
            elif rec_key.endswith('_chosen_reward'):
                prefix = rec_key.removesuffix('_chosen_reward')
                chosen_reward = sum(rec_value) / len(rec_value)
                stats[f'{prefix}_chosen_reward'] = chosen_reward
            elif rec_key.endswith('_rejected_reward'):
                prefix = rec_key.removesuffix('_rejected_reward')
                rejected_reward = sum(rec_value) / len(rec_value)
                stats[f'{prefix}_rejected_reward'] = rejected_reward

        return stats
                
    def loss(
        self,
        chosen_logprobs: torch.FloatTensor,
        rejected_logprobs: torch.FloatTensor,
        ref_chosen_logprobs: torch.FloatTensor,
        ref_rejected_logprobs: torch.FloatTensor,
    ):
        
        logratios = chosen_logprobs - rejected_logprobs
        ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
        logits = logratios - ref_logratios

        if self.loss_type == 'sigmoid':
            loss = torch.mean(
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        chosen_reward = self.beta * (chosen_logprobs.to(self.device) - ref_chosen_logprobs.to(self.device)).detach().mean()
        rejected_reward = self.beta * (rejected_logprobs.to(self.device) - ref_rejected_logprobs.to(self.device)).detach().mean()

        return loss, chosen_reward, rejected_reward

if __name__ == '__main__':

    # torch.cuda.set_device(4)

    config = DPO_Config()
    dataset = HumanFeedback_Dataset(config.dateset_cfg, config.dateset_cfg.data_path)
    train_ds, eval_ds = dataset.get_generator()

    model = AutoModelForCausalLM.from_pretrained(config.model_cfg.model_pretrain_path)
    tokenizer = AutoTokenizer.from_pretrained(config.dateset_cfg.tokenizer_pretrain_path)

    dpo_trainer = DPO_Trainer(
        config = config,
        model = model
    )

    dpo_trainer.train(
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        n_epoch = config.n_epoch,
        n_eval_step = config.n_eval_step,
        train_batch_size = config.train_batch_size,
        eval_batch_size = config.eval_batch_size
    )