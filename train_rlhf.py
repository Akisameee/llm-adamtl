import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,7'
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
# from peft import get_peft_model

from data.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import RLHF_Config
from modules.lms import BaseLM, RewardLM, get_dispatched_model
from modules.ppo import PPO_Trainer, Memory
from modules.peft import replace_peft_layers, set_all_adapters
from modules.utils import shift, log_prob, default, masked_mean, merge_dict
from logger import Logger
from trl.core import LengthSampler

class RLHF_Trainer(nn.Module):

    def __init__(
        self,
        config: RLHF_Config,
        model: AutoModelForCausalLMWithValueHead = None,
        reward_model: RewardLM = None,
        ref_model: BaseLM = None,
        logger: Logger = None
    ):
        super().__init__()

        # models
        if model is None:
            model, model_peft_info = get_dispatched_model(
                config = config.model_cfg,
                model_class = AutoModelForCausalLMWithValueHead
            )
        else:
            model_peft_info = None

        self.reward_model = reward_model
        if self.reward_model is None:
            self.reward_model, _ = get_dispatched_model(
                config = config.reward_cfg,
                model_class = RewardLM
            )
        self.reward_model.set_freeze(freeze=True)
        self.reward_model.eval()

        self.ref_model = ref_model
        if self.ref_model is None:
            if model_peft_info is not None:
                self.ref_model = None
            else:
                self.ref_model, _ = get_dispatched_model(
                    config = config.ref_cfg,
                    model_class = BaseLM
                )
                self.ref_model.set_freeze(freeze=True)
                self.ref_model.eval()
                
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = config.lr,
            weight_decay = config.weight_decay
        )

        # prepare with accelerator
        self.accelerator = Accelerator(
            log_with = config.accelertor_cfg.log_with,
            gradient_accumulation_steps = config.accelertor_cfg.gradient_accumulation_steps
        )

        (
            model,
            self.ref_model,
            self.reward_model,
            optimizer
        ) = self.accelerator.prepare(
            model,
            self.ref_model,
            self.reward_model,
            optimizer
        )
        
        self.logger = logger
        if self.logger is None:
            self.logger = Logger(
                output_dir = config.output_dir,
                task_name = 'RLHF_train',
                disable = not self.accelerator.is_main_process
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

        self.logger.info(config.get_args_info())
        if model_peft_info:
            self.logger.info(model_peft_info)
        if self.ref_model is None:
            self.logger.info(f'Disabling target model peft layers as reference model.')

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
    
    def compute_reward_single(
        self,
        rm_reward: torch.FloatTensor,
        action_logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor
    ):
        kl_ref = self.kl_penalty(action_logprobs, ref_logprobs) * self.kl_ref_coef
        reward = -kl_ref
        reward[:, -1] += rm_reward
        kl_ref_mean = -torch.mean(kl_ref).unsqueeze(0)
        
        return reward, kl_ref_mean, kl_ref_mean + rm_reward
        
    def kl_penalty(
        self,
        logprob,
        ref_logprob
    ):
        return logprob - ref_logprob

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

    @torch.no_grad()
    def get_rm_rewards(
        self,
        reward_model: nn.Module,
        sequences: torch.FloatTensor,
        masks: torch.LongTensor,
        action_masks: torch.LongTensor,
        # prompt_texts: list[str] = None
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
        self.logger.info(
            '-' * 50 + '\n' + \
            f'Prompt:{prompt_texts[0]}\n' + \
            '-' * 50 + '\n' + \
            f'Response:{response_texts[0]}\n' + \
            '-' * 50 + '\n' + \
            f'RM_Score:{rm_rewards[0]}'
        )

        return rm_rewards
    
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

    def train(
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
        length_sampler = LengthSampler(32, 128)

        timestep = 0
        updatestep = 0
        rlhf_bar = tqdm(
            total = max_timestep // n_update_timestep * n_update_timestep // sample_batch_size,
            disable = not self.accelerator.is_main_process
        )
        for episode in range(n_episode):
            for prompts_ids, attention_masks, prompt_texts in dataloader:
                
                self.ppo_trainer.model.eval()
                batch_size = len(prompts_ids)
                timestep += batch_size

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

                ref_logprobs = self.get_ref_logprobs(
                    ref_model = self.ref_model,
                    sequences = sequences,
                    masks = masks
                )

                rm_rewards = self.get_rm_rewards(
                    reward_model = self.reward_model,
                    sequences = sequences,
                    masks = masks,
                    action_masks = action_masks,
                    # prompt_texts = prompt_texts
                )
                
                detach_to_cpu_ = lambda t: t.detach().cpu()

                for (
                    sequence,
                    mask,
                    action_mask,
                    prob,
                    logprob,
                    ref_logprob,
                    rm_reward,
                    value
                ) in zip(
                    sequences,
                    masks,
                    action_masks,
                    probs,
                    logprobs,
                    ref_logprobs,
                    rm_rewards,
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
                        rm_reward,
                        value[: seq_len]
                    ))))
                
                rlhf_bar.update(1)
                if timestep >= (updatestep + 1) * n_update_timestep:
                    updatestep += 1
                    ppo_stats = [self.ppo_trainer.learn(memories)]
                    # print(f'ppo_stats {ppo_stats}')
                    ppo_stats_gathered = self.accelerator.gather_for_metrics(ppo_stats)
                    if self.accelerator.is_main_process:
                        # print(f'ppo_stats_gathered {ppo_stats_gathered}')
                        self.logger.step(
                            episode = episode + 1,
                            timestep = timestep,
                            stat_dict = merge_dict(
                                unmerged_dicts = ppo_stats_gathered,
                                reduce = 'mean'
                            )
                        )
                    memories.clear()
                    if max_timestep - timestep < n_update_timestep:
                        break

        self.logger.info('RLHF Training Complete')
        self.logger.save_res()


def main():

    config = RLHF_Config()

    data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    sub_data_path = ['helpful-base']
    config.dateset_cfg.data_path = data_path
    config.dateset_cfg.sub_data_path = sub_data_path

    model_path = '/home/share/models/huggingface/bit-dny/MindLLM'
    config.dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path
    config.ref_cfg.model_pretrain_path = model_path
    
    # rm_path = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
    # config.reward_cfg.model_pretrain_path = rm_path
    config.parse_args()

    trainer = RLHF_Trainer(
        config = config
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    dataset = Instruct_Dataset(config.dateset_cfg)
    dataset.load(mode = 'train')
    trainer.train(
        ds_generator = dataset.get_generator(),
        n_episode = config.n_episode,
        n_update_timestep = config.n_update_timestep,
        sample_batch_size = config.sample_batch_size
    )

if __name__ == '__main__':
    
    main()