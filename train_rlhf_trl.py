import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from random import randrange
import os
from peft import get_peft_model

from configs import RLHF_Config
from data.instruct_dataset import Instruct_Dataset
from modules.lms import RewardLM
from modules.peft import replace_peft_layers
from logger import Logger

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    torch.cuda.set_device(7)
    
    config = RLHF_Config()
    logger = Logger(config.output_dir, 'RLHF_train_trl')

    ppo_config = PPOConfig(
        model_name=config.model_cfg.model_pretrain_path,
        batch_size=config.n_update_timestep,
        mini_batch_size=config.train_batch_size,
        learning_rate=config.lr,
        # log_with="wandb",
    )

    dataset = Instruct_Dataset(config.dateset_cfg, config.dateset_cfg.train_data_path)
    instruct_dataset = dataset.get_generator()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_cfg.model_pretrain_path,
        peft_config = config.model_cfg.peft_config
    )
    # model = replace_peft_layers(
    #     model = model,
    #     peft_config = config.model_cfg.peft_config
    # )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    total_non_trainable_params = total_params - total_trainable_params
    print(f'{total_non_trainable_params:,} non-trainable parameters.')
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.ref_cfg.model_pretrain_path)
    reward_model = RewardLM(config.reward_cfg)
    tokenizer = AutoTokenizer.from_pretrained(config.model_cfg.model_pretrain_path)

    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model,
        tokenizer,
        dataset=instruct_dataset,
        data_collator=collator
    )

    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": config.model_cfg.generation_config.eos_token_id,
    }

    output_min_length = 32
    output_max_length = 128
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    device = config.model_cfg.device
    prompts = dataset.datas
    total_timestep = 0
    
    # prompts_ids = [prompt['input_ids'].squeeze().to(device) for prompt in prompts]
    # prompts_text = [prompt['prompt_text'] for prompt in prompts]
    prompt_idss = [prompt.squeeze().to(device) for prompt in prompts[0]]
    prompt_texts = prompts[2]

    query_tensors = []
    response_tensors = []
    rewards = []
    for episode in tqdm(range(config.n_episode)):
        for timestep in range(config.n_timestep):
            total_timestep += 1
            rand_prompt_index = randrange(0, len(prompt_idss))

            prompt_ids = prompt_idss[rand_prompt_index]
            query_tensors.append(prompt_ids)
            prompt_text = prompt_texts[rand_prompt_index]
            
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(prompt_ids, **generation_kwargs)
            response_id = response.squeeze()[-gen_len:]
            response_tensors.append(response_id)
            response_text = tokenizer.decode(response_id.squeeze())

            # print(f'Prompt:{prompt_text}Response:{response_text}\n')

            reward_sequence, reward_mask, reward_prompt_mask = reward_model.encode_single(
                prompt_text = prompt_text,
                response_text = response_text
            )
            rm_reward = reward_model(
                reward_sequence,
                attention_mask = reward_mask,
                token_type_ids = reward_prompt_mask,
                sample = True
            )
            rewards.append(rm_reward)

            if total_timestep % config.n_update_timestep == 0:
                # logger.info('Updating...')
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                avg_rm_reward = sum(rewards).item() / len(rewards)
                logger.step(
                    episode = episode,
                    timestep = total_timestep,
                    stat_dict = {
                        'Policy_Loss':stats['ppo/loss/policy'],
                        'Critic_Loss':stats['ppo/loss/value'],
                        'Reward':stats['ppo/mean_scores'],
                        'RM_Score':avg_rm_reward,
                        'Ref_KL':stats['ppo/mean_non_score_reward']
                    }
                )
                query_tensors.clear()
                response_tensors.clear()
                rewards.clear()
                torch.cuda.empty_cache()

    logger.save_res()


    # for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    #     query_tensors = [query.squeeze() for query in batch["input_ids"]]

    #     # Get response
    #     response_tensors = []
    #     for query in query_tensors:
    #         gen_len = output_length_sampler()
    #         generation_kwargs["max_new_tokens"] = gen_len
    #         response = ppo_trainer.generate(query, **generation_kwargs)
    #         response_tensors.append(response.squeeze()[-gen_len:])
    #     batch["response_text"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #     # Get reward
    #     rewards = []
    #     for prompt_text, response_text in zip(batch["prompt_text"], batch["response_text"]):
    #         print(f'Prompt:{prompt_text}Response:{response_text}\n')
    #         reward_sequence, reward_mask, reward_prompt_mask = reward_model.encode_single(
    #             prompt_text = prompt_text,
    #             response_text = response_text
    #         )
    #         reward = reward_model(
    #             reward_sequence,
    #             attention_mask = reward_mask,
    #             token_type_ids = reward_prompt_mask,
    #             sample = True
    #         )
    #         rewards.append(reward)

    #     # Run PPO step
    #     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    #     ppo_trainer.log_stats(
    #         stats,
    #         {
    #             'query':query_tensors,
    #             'response':response_tensors
    #         },
    #         rewards
    #     )