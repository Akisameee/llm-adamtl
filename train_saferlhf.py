import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import deque
from tqdm import tqdm
from accelerate.utils import broadcast

from base_trainer import Base_Trainer
from datas.instruct_dataset import Instruct_Dataset, instruct_collator
from configs import Safe_RLHF_Config, RM_Config, SVD_Lora_Config, SVD_Lora_Altered_Config
# from modules.lms import BaseLM, RewardLM
from modules.base import BaseLMWithValueHeads
from modules.safeppo import SafePPO_Trainer, SafePPOMemory
from modules.moppo import MOPPO_Trainer
from modules.pefts import set_all_adapters
from modules.utils import shift, log_prob, merge_dict, get_model

TEST = 0

class SafeRLHF_Trainer(Base_Trainer):

    def __init__(
        self,
        config: Safe_RLHF_Config
    ):
        config.model_cfg.model_class = BaseLMWithValueHeads

        self.reward_names = [
            'Helpful',
            'Harmless'
        ]
        self.reward_cfgs = [
            config.reward_cfg,
            config.cost_cfg
        ]

        super().__init__(
            config = config,
            accelerator_cfg = config.accelertor_cfg,
            model_cfg = config.model_cfg,
            ref_cfg = config.ref_cfg,
            reward_cfg = self.reward_cfgs,
            n_v_head = 2
        )
        
        self.ppo_trainer = SafePPO_Trainer(
            config = config,
            model = self.model,
            optimizer = self.optimizer,
            accelerator = self.accelerator,
            logger = self.logger
        )

        # self.n_sample_reuse = config.n_sample_reuse
        self.n_sample_reuse = 1
        if config.n_sample_reuse != 1:
            self.logger.info(
                f'Unable to reuse samples in safe rlhf, setting n_sample_reuse to 1.'
            )
        self.n_save_step = config.n_save_step
        # self.clean_cache_every_iter = True
        self.n_eval_sample = config.n_eval_sample

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

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        n_episode: int = 1,
        n_update_timestep: int = 8,
        sample_batch_size: int = 1,
    ):
        set_all_adapters(
            model = self.ppo_trainer.model,
            enable = True
        )

        dataloader = DataLoader(
            dataset = train_dataset,
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
        tqdm_bar = tqdm(
            total = max_timestep // n_update_timestep * n_update_timestep // sample_batch_size,
            disable = not self.accelerator.is_main_process
        )
        
        self.get_save_timesteps(self.n_save_step, tqdm_bar.total)
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

                reward_scores = []
                cost_scores = []
                for idx, reward_model in enumerate(self.reward_models):
                    rm_rewards = self.get_rm_rewards(
                        reward_model = reward_model,
                        sequences = sequences,
                        masks = masks,
                        action_masks = action_masks,
                        verbose = timestep % n_update_timestep == 0
                        # verbose = True
                    )
                    if idx == 0:
                        reward_scores.append(rm_rewards)
                    elif idx == 1:
                        cost_scores.append(rm_rewards)
                    sample_record[self.reward_names[idx]] = torch.sum(rm_rewards).item()
                sample_records.append(sample_record)

                detach_to_cpu_ = lambda t: t.detach().cpu()

                for (
                    sequence,
                    mask,
                    action_mask,
                    reward_score,
                    cost_score
                ) in zip(
                    sequences,
                    masks,
                    action_masks,
                    reward_scores,
                    cost_scores
                ):
                    seq_len = torch.sum(mask).item()
                    memories.append(SafePPOMemory(*map(detach_to_cpu_, (
                        sequence[: seq_len],
                        mask[: seq_len],
                        action_mask[: seq_len],
                        reward_score,
                        cost_score
                    ))))

                if self.clean_cache_every_iter:
                    del (
                        sequences,
                        masks,
                        action_masks,
                        rm_rewards,
                    )
                    torch.cuda.empty_cache()
                
                tqdm_bar.update(1)
                if timestep >= (updatestep + 1) * n_update_timestep:
                    updatestep += 1
                    self.accelerator.wait_for_everyone()
                    ppo_stats = [self.ppo_trainer.step(memories)]
                    torch.cuda.empty_cache()

                    ppo_stats_gathered = self.accelerator.gather_for_metrics(ppo_stats)
                    sample_records_gathered = self.accelerator.gather_for_metrics(sample_records)
                    all_ppo_stats = merge_dict(unmerged_dicts = ppo_stats_gathered, reduce = 'mean')
                    all_sample_records = merge_dict(unmerged_dicts = sample_records_gathered, reduce = 'mean')
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
                        # save model
                        if self.accelerator.is_main_process:
                            self.save(
                                self.ppo_trainer.model,
                                os.path.join(self.logger.dir, f'{self.model_name}_{episode}_{timestep}')
                            )
                        # eval
                        self.accelerator.wait_for_everyone()
                        self.eval_step(
                            eval_dataset = eval_dataset,
                            eval_step = self.save_step - 1,
                            n_eval_sample = self.n_eval_sample
                        )
                        # self.logger.save_pareto_front_test(
                        #     tuple(self.reward_names),
                        #     vecs_name = 'pref_vec',
                        #     save_dir = os.path.join(self.logger.dir, f'{self.model_name}_{episode}_{timestep}'),
                        #     eval_step = self.save_step - 1
                        # )
                        torch.cuda.empty_cache()

                    if max_timestep - timestep < n_update_timestep:
                        break

        self.accelerator.wait_for_everyone()
        self.logger.info(f'{self.task_name} complete.')
        self.logger.save_res()

    @torch.no_grad()
    def eval_step(
        self,
        eval_dataset: Dataset,
        eval_step: int,
        n_eval_sample: int = 100,
        eval_batch_size: int = 1,
    ):
        self.model.eval()
        set_all_adapters(
            model = self.ppo_trainer.model,
            enable = True
        )

        # print(ds_generator.datas[:n_test_sample])
        eval_dataset.datas = eval_dataset[:n_eval_sample]
        dataloader = DataLoader(
            dataset = eval_dataset,
            batch_size = eval_batch_size,
            shuffle = False,
            collate_fn = instruct_collator,
            drop_last = True
        )
        dataloader = self.accelerator.prepare(dataloader)

        timestep = 0
        sample_records = []
        tqdm_bar = tqdm(
            total = (len(dataloader) * eval_batch_size) // eval_batch_size,
            disable = not self.accelerator.is_main_process
        )
        self.logger.info(
            f'Evaluation step {eval_step}:\n'
        )

        dl_len = len(dataloader)
        for prompts_ids, attention_masks, prompt_texts in dataloader:
            
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
                return_padding = True,
                **self.generation_config.to_dict()
            )

            reward_scores = []
            cost_scores = []
            for idx, reward_model in enumerate(self.reward_models):
                rm_rewards = self.get_rm_rewards(
                    reward_model = reward_model,
                    sequences = sequences,
                    masks = masks,
                    action_masks = action_masks,
                    verbose = tqdm_bar.n % dl_len == 0
                    # verbose = True
                )
                if idx == 0:
                    reward_scores.append(rm_rewards)
                elif idx == 1:
                    cost_scores.append(rm_rewards)
                sample_record[self.reward_names[idx]] = torch.sum(rm_rewards).item()
            sample_records.append(sample_record)
            tqdm_bar.update(1)
        
        sample_records_gathered = self.accelerator.gather_for_metrics(sample_records)
        all_sample_records = merge_dict(unmerged_dicts = sample_records_gathered, reduce = 'mean')
        self.logger.step(
            episode = 0,
            timestep = timestep,
            stat_dict = all_sample_records,
            eval_step = eval_step
        )
        sample_records.clear()
        
        set_all_adapters(
            model = self.ppo_trainer.model,
            enable = True
        )
        self.accelerator.wait_for_everyone()
        self.logger.info(f'Evaluation step {eval_step} complete.')
        

def main():

    config = Safe_RLHF_Config()

    data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    # sub_data_path = ['helpful-base', 'harmless-base']
    sub_data_path = ['harmless-base']
    config.dateset_cfg.data_path = data_path
    config.dateset_cfg.sub_data_path = sub_data_path
    
    model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']

    # model_path = '/home/share/models/huggingface/meta-llama/Llama-2-7b-chat-hf'
    # config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    config.dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path
    config.ref_cfg.model_pretrain_path = model_path
    
    reward_path = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
    cost_path = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-harmless-reward_model')
    config.reward_cfg = RM_Config(model_pretrain_path = reward_path)
    config.cost_cfg = RM_Config(model_pretrain_path = cost_path)

    config.lr = 1e-4

    # whole dataset
    max_sample = 0
    config.n_update_timestep = 64
    config.accelertor_cfg.gradient_accumulation_steps = 16
    config.train_batch_size = 1

    # 1/2 dataset
    # max_sample = 20000
    # # max_sample = 15000
    # config.n_update_timestep = 64
    # config.accelertor_cfg.gradient_accumulation_steps = 8
    # config.train_batch_size = 1

    if TEST:
        config.accelertor_cfg.gradient_accumulation_steps = 2
        config.n_update_timestep = 8
        config.n_eval_sample = 4
        config.n_save_step = 1

    config.parse_args()

    trainer = SafeRLHF_Trainer(
        config = config
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    train_dataset = Instruct_Dataset(config.dateset_cfg)
    train_dataset.load(mode = 'train', max_sample = max_sample if not TEST else 60)
    eval_dataset = Instruct_Dataset(config.dateset_cfg)
    eval_dataset.load(mode = 'eval', max_sample = 500 if not TEST else 50)
    trainer.train(
        train_dataset = train_dataset.get_generator(),
        eval_dataset = eval_dataset.get_generator(),
        n_episode = config.n_episode,
        n_update_timestep = config.n_update_timestep,
        sample_batch_size = config.sample_batch_size
    )

if __name__ == '__main__':
    
    main()
