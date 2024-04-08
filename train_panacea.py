import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange, repeat, reduce
from collections import deque
from functools import partial
from random import randrange
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead
from accelerate import Accelerator
import os

from data.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import Panacea_PPO_Config, RM_Config
from modules.lms import BaseLM, RewardLM
from modules.ppo import PPO_Trainer, Memory
from modules.peft import replace_peft_layers, Panacea_SVD_Linear
from modules.utils import shift, log_prob, default, masked_mean
from logger import Logger
from trl.core import LengthSampler

class Panacea_Trainer(nn.Module):

    def __init__(
        self,
        config: Panacea_PPO_Config,
        model: AutoModelForCausalLMWithValueHead,
        reward_models: RewardLM,
        ref_model: BaseLM,
    ):
        super().__init__()

        self.logger = Logger(
            output_dir = config.output_dir,
            task_name = 'Panacea_train'
        )

        # models
        self.model = model
        if self.model is None:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_cfg.model_pretrain_path)
            self.model, peft_info = replace_peft_layers(
                model = model,
                peft_config = config.model_cfg.peft_config,
                return_info = True
            )
        else:
            peft_info = None

        reward_models.set_freeze(freeze=True)
        reward_models = reward_models.eval()
        self.pref_dim = len(reward_models)

        ref_model.set_freeze(freeze=True)
        ref_model = ref_model.eval()
                
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = config.lr,
            weight_decay = config.weight_decay
        )

        # prepare with accelerator
        self.accelerator = Accelerator(
            log_with = config.accelertor_cfg.log_with,
            gradient_accumulation_steps = config.accelertor_cfg.gradient_accumulation_steps,
            **config.accelertor_cfg.accelerator_kwargs
        )

        (
            model,
            self.ref_model,
            *self.reward_models,
            optimizer
        ) = self.accelerator.prepare(
            model,
            ref_model,
            *reward_models,
            optimizer
        )

        self.ppo_trainer = PPO_Trainer(
            config = config,
            model = model,
            optimizer = optimizer,
            accelerator = self.accelerator,
            logger = self.logger
        )
        
        self.generation_config = config.model_cfg.generation_config
        self.retokenization = config.retokenization
        self.kl_ref_coef = config.kl_ref_coef
        self.scalariztion_type = config.scalariztion_type

        if peft_info:
            self.logger.info(peft_info)

    def save(self, filepath = './checkpoint.pt'):
        torch.save(self.ppo_trainer.state_dict(), filepath)

    def load(self, filepath = './checkpoint.pt'):
        state_dict = torch.load(filepath)
        self.ppo_trainer.load_state_dict(state_dict)

    @property
    def device(self):
        if self.accelerator is not None:
            return self.accelerator.device
        else:
            return self.ppo_trainer.model.device
        
    def sample_pref_vec(self):
        
        pref_vec = torch.rand(self.pref_dim)
        pref_vec = pref_vec / torch.sum(pref_vec)

        return pref_vec
    
    def set_pref_vec(
        self,
        model: torch.nn.Module,
        pref_vec: torch.FloatTensor
    ):
        
        for module in model.modules():
            if isinstance(module, Panacea_SVD_Linear):
                module.set_pref_vec(pref_vec)
        
    def get_rm_rewards(
        self,
        reward_model: nn.Module,
        sequences: torch.FloatTensor,
        masks: torch.LongTensor,
        action_masks: torch.LongTensor,
        prompt_texts: list[str] = None
    ):

        if self.retokenization:
            prompt_masks = (1 - action_masks) * masks
            prompt_texts_decode, response_texts = self.ppo_trainer.decode_batch(
                inputs_ids = sequences,
                attention_masks = masks,
                prompt_masks = prompt_masks
            )
            # print(prompt_texts[0], response_texts[0])
            if prompt_texts is None:
                prompt_texts = prompt_texts_decode
            reward_sequences, reward_masks, reward_token_type_ids = reward_model.encode_batch(
                prompt_texts = prompt_texts,
                response_texts = response_texts,
                return_padding = True
            )
            # print(f'Prompt:{prompt_text}Response:{response_text}\n')
        else:
            reward_sequences = sequences
            reward_masks = masks
            reward_token_type_ids = action_masks

        rm_rewards = reward_model(
            reward_sequences,
            attention_mask = reward_masks,
            token_type_ids = reward_token_type_ids,
            sample = True
        )

        return rm_rewards
    
    def scalariztion(
        self,
        rms_rewards: list[torch.FloatTensor],
        pref_vec: torch.FloatTensor
    ):  
        
        rm_rewards_scalar = torch.zeros_like(rms_rewards[0])
        if self.scalariztion_type == 'ls':
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
        n_episode: int = 50000,
        n_timestep: int = 500,
        n_update_timestep: int = 5000,
        sample_batch_size: int = 8,
        max_seq_len: int = 2048,
    ):
        
        dataloader = DataLoader(
            dataset = ds_generator,
            batch_size = sample_batch_size,
            shuffle = True,
            collate_fn = instruct_collator,
            drop_last = True
        )
        if self.accelerator is not None:
            dataloader = self.accelerator.prepare(dataloader)

        memories = deque([])
        length_sampler = LengthSampler(32, 128)

        episode = 0
        total_timestep = 0
        updatestep = 0
        pref_vec = self.sample_pref_vec()
        self.set_pref_vec(pref_vec)

        rlhf_bar = tqdm(total = n_timestep * n_episode // n_update_timestep)
        while n_timestep * n_episode - total_timestep >= n_update_timestep:
            episode += 1
            for prompts_ids, attention_masks, prompt_texts in dataloader:

                batch_size = len(prompts_ids)
                total_timestep += batch_size

                (
                    sequences,
                    masks,
                    action_masks
                ) = self.ppo_trainer.generate_batch(
                    prompts_ids = prompts_ids,
                    attention_masks = attention_masks,
                    length_sampler = length_sampler,
                    return_padding = True,
                    **self.generation_config.to_dict()
                )

                logits, values = self.ppo_trainer.batch_forward(
                    sequences,
                    mask = masks
                )
                logits = shift(logits, shift = 1, dim = -2)
                probs = logits.softmax(dim = -1)
                logprobs = log_prob(probs, sequences)

                ref_logits, _, _ = self.ref_model(
                    input_ids = sequences,
                    attention_mask = masks
                )
                ref_logits = shift(ref_logits, shift = 1, dim = -2)
                ref_probs = ref_logits.softmax(dim = -1)
                ref_logprobs = log_prob(ref_probs, sequences)

                # prompt_masks = (1 - action_masks) * masks
                # if self.retokenization:
                #     _, response_texts = self.ppo_trainer.decode_batch(
                #         inputs_ids = sequences,
                #         attention_masks = masks,
                #         prompt_masks = prompt_masks
                #     )
                #     # print(prompt_texts[0], response_texts[0])
                #     reward_sequences, reward_masks, reward_token_type_ids = self.reward_model.encode_batch(
                #         prompt_texts = prompt_texts,
                #         response_texts = response_texts,
                #         return_padding = True
                #     )
                #     # print(f'Prompt:{prompt_text}Response:{response_text}\n')
                # else:
                #     reward_sequences = sequences
                #     reward_masks = masks
                #     reward_token_type_ids = action_masks

                # rm_rewards = self.reward_model(
                #     reward_sequences,
                #     attention_mask = reward_masks,
                #     token_type_ids = reward_token_type_ids,
                #     sample = True
                # )

                rms_rewards = []
                for reward_model in self.reward_models:
                    rm_rewards = self.get_rm_rewards(
                        reward_model = reward_model,
                        sequences = sequences,
                        masks = masks,
                        action_masks = action_masks,
                        prompt_texts = prompt_texts
                    )
                    rms_rewards.append(rm_rewards)
                rm_rewards_scalar = self.scalariztion(
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
                    rm_reward_scalar,
                    value
                ) in zip(
                    sequences,
                    masks,
                    action_masks,
                    probs,
                    logprobs,
                    ref_logprobs,
                    rm_rewards_scalar,
                    values
                ):
                    seq_len = torch.sum(mask).item()
                    memories.append(Memory(*map(detach_to_cpu_, (
                        sequence[: seq_len],
                        mask[: seq_len],
                        action_mask[: seq_len],
                        prob[: seq_len, :],
                        logprob[: seq_len],
                        ref_logprob[: seq_len],
                        rm_reward_scalar,
                        value[: seq_len]
                    ))))

                if total_timestep >= (updatestep + 1) * n_update_timestep:
                    updatestep += 1
                    ppo_stats = self.ppo_trainer.learn(memories)
                    self.logger.step(
                        episode = episode,
                        timestep = total_timestep,
                        stat_dict = ppo_stats
                    )
                    memories.clear()
                    rlhf_bar.update(1)
                    if n_timestep * n_episode - total_timestep < n_update_timestep:
                        break
                    else:
                        pref_vec = self.sample_pref_vec()
                        self.set_pref_vec(pref_vec)

        self.logger.info('RLHF Training Complete')
        self.logger.save_res()


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    torch.cuda.set_device(7)

    config = Panacea_PPO_Config(
        reward_cfgs = [
            RM_Config(
                model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'reward-model-deberta-v3-base')
            )
        ]
    )

    # model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_cfg.model_pretrain_path)
    reward_models = [RewardLM(reward_cfg) for reward_cfg in config.reward_cfgs]
    reference_model = BaseLM(config.model_cfg)

    trainer = Panacea_Trainer(
        config = config,
        # model = model,
        reward_models = reward_models,
        ref_model = reference_model
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    dataset = Instruct_Dataset(config.dateset_cfg, config.dateset_cfg.train_data_path)
    trainer.train_ppo(
        ds_generator = dataset.get_generator(),
        n_episode = config.n_episode,
        n_timestep = config.n_timestep,
        n_update_timestep = config.n_update_timestep,
        sample_batch_size = config.sample_batch_size,
        max_seq_len = 300
    )
