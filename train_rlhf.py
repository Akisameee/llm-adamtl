import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
# from peft import get_peft_model

from data.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import RLHF_Config, parse_args_into_dataclasses
from modules.lms import BaseLM, RewardLM
from modules.ppo import PPO_Trainer, Memory
from modules.peft import replace_peft_layers
from modules.utils import shift, log_prob, default, masked_mean, merge_dict
from logger import Logger
from trl.core import LengthSampler

class RLHFTrainer(nn.Module):

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
            model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_cfg.model_pretrain_path)
            model, peft_info = replace_peft_layers(
                model = model,
                peft_config = config.model_cfg.peft_config,
                return_info = True
            )
        else:
            peft_info = None

        self.reward_model = reward_model
        if self.reward_model is None:
            self.reward_model = RewardLM(config.reward_cfg)
        self.reward_model.set_freeze(freeze=True)
        self.reward_model = self.reward_model.eval()

        self.ref_model = ref_model
        if self.ref_model is None:
            self.ref_model = BaseLM(config.ref_cfg)
        self.ref_model.set_freeze(freeze=True)
        self.ref_model = self.ref_model.eval()
                
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

    @torch.no_grad()
    def generate(
        self,
        max_seq_len: int,
        *args,
        prompt: torch.FloatTensor,
        num_samples: int = 4,  # sample 4 per prompt and select the one with highest reward
        **kwargs
    ):
        
        (
            sequences,
            mask,
            prompt_mask
        ) = self.ppo_trainer.generate(
            max_seq_len = max_seq_len,
            *args,
            prompt = prompt,
            num_samples = num_samples,
            **kwargs
        )

        rewards = self.reward_model(
            sequences,
            prompt_mask = prompt_mask,
            mask = mask,
            sample = True
        )

        best_sequence_index = rewards.topk(1, dim = -1).indices

        best_sequence = sequences[best_sequence_index]
        best_sequence = rearrange(best_sequence, '1 ... -> ...')

        return best_sequence
    
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

    def train_single_sampling(
        self,
        prompts,
        n_episode: int = 50000,
        n_timestep: int = 500,
        n_update_timestep: int = 5000,
        max_seq_len: int = 2048,
    ):
        device = self.device
        total_timestep = 0
        memories = deque([])
        length_sampler = LengthSampler(32, 128)

        # prompts_ids = [prompt['input_ids'].squeeze().to(device) for prompt in prompts]
        # prompts_text = [prompt['prompt_text'] for prompt in prompts]
        prompts_ids = [prompt.squeeze().to(device) for prompt in prompts[0]]
        prompts_text = prompts[2]

        for episode in tqdm(range(n_episode)):
            for timestep in range(n_timestep):
                total_timestep += 1

                # select a bunch of random states (prompts)
                # and get the action (sampled sequence from palm as well as the action probs)
                # also calculate the reward using reward model and store
                rand_prompt_index = randrange(0, len(prompts_ids))

                prompt_ids = prompts_ids[rand_prompt_index]
                prompt_text = prompts_text[rand_prompt_index]

                # get predicted sequence
                self.generation_config.max_new_tokens = length_sampler()
                (
                    actions,
                    sequence,
                    mask,
                    prompt_mask,
                    logits,
                    value
                ) = self.ppo_trainer.generate(
                    prompt_ids = rearrange(prompt_ids, 'n -> 1 n'),
                    max_seq_len = max_seq_len,
                    **self.generation_config.to_dict()
                )

                # need to shift along sequence dimension by 1, since actions start from the last prompt (state) token
                logits = shift(logits, shift = 1, dim = -2)
                prob = logits.softmax(dim = -1)

                action_len = actions.shape[-1]
                logprob = log_prob(prob, sequence)
                # action_log_prob = action_log_prob[:, -action_len:]

                # reference model logits
                action_mask = ~prompt_mask.bool()
                ref_logits, _, _ = self.ref_model(
                    input_ids = sequence,
                    attention_mask = mask
                )

                ref_logits = shift(ref_logits, shift = 1, dim = -2)
                ref_prob = ref_logits.softmax(dim = -1)
                ref_logprob = log_prob(ref_prob, sequence)
                # ref_log_prob = ref_log_prob[:, -action_len:]
                
                actions = rearrange(actions, '1 ... -> ...')

                # get reward as given by supervised trained reward model
                sequence = torch.cat((prompt_ids, actions), dim = 0)

                prompt_length = len(prompt_ids)
                prompt_mask = torch.arange(sequence.shape[-1], device = device) < prompt_length

                sequence = rearrange(sequence, 'n -> 1 n')
                prompt_mask = rearrange(prompt_mask, 'n -> 1 n').long()
                mask = default(mask, lambda: torch.ones(sequence.shape, dtype = torch.bool, device = device))

                if self.retokenization:
                    _, response_text = self.ppo_trainer.decode_single(
                        input_ids = sequence,
                        attention_mask = mask,
                        prompt_mask = prompt_mask
                    )
                    reward_sequence, reward_mask, reward_prompt_mask = self.reward_model.encode_single(
                        prompt_text = prompt_text,
                        response_text = response_text
                    )
                    # print(f'Prompt:{prompt_text}Response:{response_text}\n')
                else:
                    reward_sequence = sequence
                    reward_mask = mask
                    reward_prompt_mask = prompt_mask

                rm_reward = self.reward_model(
                    reward_sequence,
                    attention_mask = reward_mask,
                    token_type_ids = reward_prompt_mask,
                    sample = True
                )
                
                # reward, kl_ref, reward_mean = self.compute_reward_single(
                #     rm_reward = rm_reward,
                #     action_logprobs = logprob,
                #     ref_logprobs = ref_logprob
                # )

                detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')

                # store memory for learning
                memories.append(Memory(*map(detach_to_cpu_, (
                    sequence,
                    mask,
                    action_mask,
                    prob,
                    logprob,
                    ref_logprob,
                    rm_reward,
                    value
                ))))

                # learn from the stored memories
                if total_timestep % n_update_timestep == 0:
                    # self.logger.info('Updating...')
                    ppo_stats = self.ppo_trainer.learn(memories)
                    ppo_stats_gathered = self.accelerator.gather_for_metrics(ppo_stats)
                    if self.accelerator.is_main_process:
                        self.logger.step(
                            episode = episode,
                            timestep = total_timestep,
                            stat_dict = merge_dict(
                                unmerged_dicts = ppo_stats_gathered,
                                reduce = 'mean'
                            ) if isinstance(ppo_stats_gathered, list) else ppo_stats_gathered
                        )
                    memories.clear()

        self.logger.info('RLHF Training Complete')
        self.logger.save_res()

    def train(
        self,
        ds_generator,
        n_episode: int = 10,
        n_update_timestep: int = 8,
        sample_batch_size: int = 8,
    ):
        
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

                ref_logits, _, _ = self.ref_model(
                    input_ids = sequences,
                    attention_mask = masks
                )
                ref_logits = shift(ref_logits, shift = 1, dim = -2)
                ref_probs = ref_logits.softmax(dim = -1)
                ref_logprobs = log_prob(ref_probs, sequences)

                rm_rewards = self.get_rm_rewards(
                    reward_model = self.reward_model,
                    sequences = sequences,
                    masks = masks,
                    action_masks = action_masks,
                    prompt_texts = prompt_texts
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
                    ppo_stats = self.ppo_trainer.learn(memories)
                    ppo_stats_gathered = self.accelerator.gather_for_metrics(ppo_stats)
                    if self.accelerator.is_main_process:
                        self.logger.step(
                            episode = episode + 1,
                            timestep = timestep,
                            stat_dict = merge_dict(
                                unmerged_dicts = ppo_stats_gathered,
                                reduce = 'mean'
                            ) if isinstance(ppo_stats_gathered, list) else ppo_stats_gathered
                        )
                    memories.clear()
                    if max_timestep - timestep < n_update_timestep:
                        break

        self.logger.info('RLHF Training Complete')
        self.logger.save_res()


def main():

    config = RLHF_Config()
    config = parse_args_into_dataclasses(
        dataclass = RLHF_Config
    )

    # model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #     config.model_cfg.model_pretrain_path,
    # )
    # model = replace_peft_layers(
    #     model = model,
    #     peft_config = config.model_cfg.peft_config
    # )
    # reward_model = RewardLM(config.reward_cfg)
    # reference_model = BaseLM(config.ref_cfg)

    trainer = RLHFTrainer(
        config = config
        # reward_model = reward_model,
        # ref_model = reference_model
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    dataset = Instruct_Dataset(config.dateset_cfg, config.dateset_cfg.train_data_path)
    trainer.train(
        ds_generator = dataset.get_generator(),
        n_episode = config.n_episode,
        n_update_timestep = config.n_update_timestep,
        sample_batch_size = config.sample_batch_size
    )

if __name__ == '__main__':
    
    main()