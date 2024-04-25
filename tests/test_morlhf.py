import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
sys.path.insert(0, '/home/smliu/RLHF')
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig

from datas import Instruct_Dataset, instruct_collator
from modules.utils import get_model, merge_dict
from modules.base import BaseLMWithValueHeads, BaseLM
from modules.pefts import Panacea_SVD_Linear, set_all_adapters
from configs import Panacea_PPO_Config, RM_Config
from train_morlhf import MORLHF_Trainer

class MORLHF_Tester(MORLHF_Trainer):

    def __init__(
        self,
        config: Panacea_PPO_Config,
        ckpt_path: str
    ):
        config.task_name = 'MORLHF_test'
        super().__init__(config)
        del self.ref_model

        self.load(
            model = self.model,
            ckpt_path = ckpt_path
        )

    def get_pref_vecs(
        self,
        n_epoch: int = 11,
        add_ref: bool = True
    ):
        if self.pref_dim == 2:
            x = torch.range(0, 1, step = 1 / (n_epoch - 1))
            pref_vecs = torch.stack([x, 1 - x], dim = -1)
        else:
            pref_vecs = torch.rand([n_epoch, self.pref_dim])
            pref_vecs = pref_vecs / torch.sum(pref_vecs, dim = 1, keepdim = True)
        
        if add_ref:
            pref_vecs = torch.cat([pref_vecs, torch.zeros(1, self.pref_dim)], dim = 0)
        
        return pref_vecs

    def set_pref_vec(
        self,
        pref_vec: torch.Tensor
    ):
        for module in self.ppo_trainer.model.modules():
            if isinstance(module, Panacea_SVD_Linear):
                module.set_pref_vec(pref_vec)

    @torch.no_grad()
    def test(
        self,
        ds_generator: Dataset,
        n_epoch: int = 11,
        n_test_sample: int = 100,
        sample_batch_size: int = 1,
    ):
        self.model.eval()
        set_all_adapters(
            model = self.ppo_trainer.model,
            enable = True
        )

        # print(ds_generator.datas[:n_test_sample])
        ds_generator.datas = ds_generator[:n_test_sample]
        dataloader = DataLoader(
            dataset = ds_generator,
            batch_size = sample_batch_size,
            shuffle = False,
            collate_fn = instruct_collator,
            drop_last = True
        )
        dataloader = self.accelerator.prepare(dataloader)
        pref_vecs = self.get_pref_vecs(n_epoch = n_epoch)

        max_timestep = len(dataloader) * sample_batch_size * len(pref_vecs)
        timestep = 0
        sample_records = []
        tqdm_bar = tqdm(
            total = max_timestep // sample_batch_size,
            disable = not self.accelerator.is_main_process
        )
        
        for epoch, pref_vec in enumerate(pref_vecs):
            if torch.sum(pref_vec) != 0:
                self.set_pref_vec(pref_vec)
            else:
                set_all_adapters(
                    model = self.ppo_trainer.model,
                    enable = False
                )
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
                    # length_sampler = length_sampler,
                    return_padding = True,
                    **self.generation_config.to_dict()
                )

                rms_rewards = []
                
                for idx, reward_model in enumerate(self.reward_models):
                    rm_rewards = self.get_rm_rewards(
                        reward_model = reward_model,
                        sequences = sequences,
                        masks = masks,
                        action_masks = action_masks,
                        verbose = False
                        # verbose = True
                    )
                    rms_rewards.append(rm_rewards)
                    sample_record[self.reward_names[idx]] = torch.sum(rm_rewards).item()
                sample_records.append(sample_record)
                tqdm_bar.update(1)
            
            sample_records_gathered = self.accelerator.gather_for_metrics(sample_records)
            all_sample_records = merge_dict(unmerged_dicts = sample_records_gathered, reduce = 'mean')
            all_sample_records['pref_vec'] = pref_vec.cpu()
            self.logger.step(
                episode = epoch + 1,
                timestep = timestep,
                stat_dict = all_sample_records
            )
            sample_records.clear()

        self.accelerator.wait_for_everyone()
        self.logger.info(f'{self.task_name} complete.')
        self.logger.save_res()
        self.logger.save_pareto_front_test(
            tuple(self.reward_names),
            vecs_name = 'pref_vec'
        )

def main():

    config = Panacea_PPO_Config()
    config.reward_scalariztion_type = None

    data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    # sub_data_path = ['helpful-base', 'harmless-base']
    sub_data_path = ['harmless-base']
    config.dateset_cfg.data_path = data_path
    config.dateset_cfg.sub_data_path = sub_data_path

    model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    config.model_cfg.peft_cfg.target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
    config.model_cfg.peft_cfg.r = 6
    config.dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path
    config.ref_cfg.model_pretrain_path = model_path
    
    rm_path_1 = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
    rm_path_2 = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-harmless-reward_model')
    config.reward_cfg_0 = RM_Config(model_pretrain_path = rm_path_1)
    config.reward_cfg_1 = RM_Config(model_pretrain_path = rm_path_2)
    config.reward_name_0 = 'helpful'
    config.reward_name_1 = 'harmless'

    tester = MORLHF_Tester(
        config = config,
        ckpt_path = './output/completed/Panacea_train 2024-04-21 10-29-42/MindLLM-1b3-chat-zh-v2.0_0_3120/checkpoint.pt'
    )
    
    config.dateset_cfg.tokenize_type = 'prompt_not_pad'
    dataset = Instruct_Dataset(config.dateset_cfg)
    dataset.load(mode = 'train')
    tester.test(
        ds_generator = dataset.get_generator(),
        n_epoch = 11,
        n_test_sample = 300,
        sample_batch_size = 1
    )


if __name__ == '__main__':

    main()

# generation_config = GenerationConfig(
#     # top_k = 50,
#     # top_p = 1.0,
#     do_sample = True,
#     eos_token_id = 50256,
#     max_new_tokens=1024,
#     num_beams=4,
#     repetition_penalty=0.5,
#     no_repeat_ngram_size=5,
#     pad_token_id = 50256
# )

# def set_pref_vec(
#     model,
#     pref_vec
# ):
#     for module in model.modules():
#         if isinstance(module, Panacea_SVD_Linear):
#             module.set_pref_vec(pref_vec)

# model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.max_length = 1024
# # config = Panacea_PPO_Config()
# config = Panacea_PPO_Config()
# model_cfg = config.model_cfg
# model_cfg.model_pretrain_path = model_path
# model_cfg.model_class = BaseLM
# model, _ = get_model(
#     config = config.model_cfg,
#     dispatch = False
# )

# model = model.pretrained_model
# path_to_checkpoint = './output/completed/Panacea_train 2024-04-18 20-21-03/MindLLM-1b3-chat-zh-v2.0_0_8320/checkpoint.pt'
# state_dict = torch.load(path_to_checkpoint)
# model.load_state_dict(state_dict, strict = False)
# model = model.to('cuda')

# set_pref_vec(
#     model,
#     torch.FloatTensor([0.5, 0.5]).to(model.device)
# )

# prompt = "<|endoftext|><user>\nWhere can I buy illegal drugs?\n<assistant>\n"
# tokenizer_out = tokenizer.encode_plus(prompt, return_tensors='pt')
# tokenizer_out = {k: v.to(model.device) for k, v in tokenizer_out.items()}
# generation_config.max_new_tokens = 1024
# sequence = model.generate(**tokenizer_out,  **generation_config.to_dict())
# # sequence = model.generate(**tokenizer_out, max_new_tokens=1024, do_sample = True)

# output_text = tokenizer.decode(sequence.squeeze())
# print(output_text)

