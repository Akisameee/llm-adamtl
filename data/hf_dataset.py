import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import csv
import re
import numpy as np
from transformers import AutoTokenizer, AutoConfig

from configs import HumanFeedback_Dataset_Config, DPO_Config, dataset_infos, model_infos

TEST = 1000

def hf_collator(batch):

    batch_keys = batch[0].keys()

    return dict(
        [(batch_key, [data[batch_key] for data in batch]) for batch_key in batch_keys]
    )

class Generator_Dataset(Dataset):

    def __init__(self, datas: dict):
        
        self.datas = datas

    def __getitem__(self, index):

        return self.datas[index]
    
    def __len__(self):

        return len(self.datas)

class HumanFeedback_Dataset(Dataset):

    def __init__(
            self,
            config: HumanFeedback_Dataset_Config,
            data_path: str) -> None:
        super().__init__()

        self.ds_info = dataset_infos[config.name]
        self.model_info = model_infos[config.model_name]

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_pretrain_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else self.tokenizer.pad_token_id
        self.label_pad_token_id = config.label_pad_token_id
        self.max_len = config.max_len

        if config.sub_data_path is None:
            self.train_datas = self.read_hh_rlhf_dataset(data_path, train = True)
            self.eval_datas = self.read_hh_rlhf_dataset(data_path, train = False)
        else:
            self.train_datas = []
            self.eval_datas = []
            for sub_data_path in config.sub_data_path:
                self.train_datas += self.read_hh_rlhf_dataset(os.path.join(data_path, sub_data_path), train = True)
                self.eval_datas += self.read_hh_rlhf_dataset(os.path.join(data_path, sub_data_path), train = False)

    def add_instruct_prompt(self, text_splitted):
        
        prompt_full = ''
        response_full = ''
        for idx, (prompt_text, response_text) in enumerate(text_splitted):
            prompt_full += self.model_info['prompt_prefix'] + prompt_text
            if idx == len(text_splitted) - 1:
                response_full = self.model_info['response_prefix'] + response_text
            else:
                prompt_full += self.model_info['response_prefix'] + response_text

        return prompt_full, response_full
    
    def split_text(self, text, instruct_pattern):

        res = []
        search_idx = 0

        match_res = re.search(instruct_pattern, text)
        while True:
            if match_res is not None:
                prompt_text = match_res.group(1)
                search_idx += match_res.end()
                match_res = re.search(instruct_pattern, text[search_idx:])
                if match_res is not None:
                    response_text = text[search_idx: search_idx + match_res.start()]
                else:
                    response_text = text[search_idx:]
                res.append((prompt_text, response_text))
            else:
                break
        
        return res
    
    def tokenize_prompt_response(self, prompt, response):

        full_tokenized = self.tokenizer(prompt + response, add_special_tokens = False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens = False)['input_ids']

        response_input_ids = full_tokenized['input_ids'][len(prompt_input_ids) :]
        response_attention_mask = full_tokenized['attention_mask'][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, response_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized['input_ids'])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError('Prompt input ids and response input ids should have the same length.')

        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+response, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized['input_ids'][: response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized['input_ids'][: response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized['attention_mask'][: response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError('Prompt input ids and attention mask should have the same length.')

        response_input_ids = full_tokenized['input_ids'][response_token_ids_start_idx:]
        response_attention_mask = full_tokenized['attention_mask'][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids = prompt_input_ids,
            prompt_attention_mask = prompt_attention_mask,
            input_ids = response_input_ids,
            attention_mask = response_attention_mask,
        )

    def tokenize_single(self, prompt, chosen, rejected):

        prompt_tk_out = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tk_out = {f'prompt_{k}': v for k, v in prompt_tk_out.items()}
        
        chosen_tk_out = self.tokenize_prompt_response(prompt, chosen)
        rejected_tk_out = self.tokenize_prompt_response(prompt, rejected)
        
        # add special tokens
        prompt_tk_out['prompt_input_ids'] = [self.tokenizer.bos_token_id] + prompt_tk_out['prompt_input_ids']
        chosen_tk_out['prompt_input_ids'] = [self.tokenizer.bos_token_id] + chosen_tk_out['prompt_input_ids']
        rejected_tk_out['prompt_input_ids'] = [self.tokenizer.bos_token_id] + rejected_tk_out['prompt_input_ids']

        prompt_tk_out['prompt_attention_mask'] = [1] + prompt_tk_out['prompt_attention_mask']
        chosen_tk_out['prompt_attention_mask'] = [1] + chosen_tk_out['prompt_attention_mask']
        rejected_tk_out['prompt_attention_mask'] = [1] + rejected_tk_out['prompt_attention_mask']

        chosen_tk_out['input_ids'].append(self.tokenizer.eos_token_id)
        chosen_tk_out['attention_mask'].append(1)

        rejected_tk_out['input_ids'].append(self.tokenizer.eos_token_id)
        rejected_tk_out['attention_mask'].append(1)

        # concat inputs
        chosen_seqs = {
            k: chosen_tk_out[f'prompt_{k}'] + chosen_tk_out[k] for k in ['input_ids', 'attention_mask']
        }
        rejected_seqs = {
            k: rejected_tk_out[f'prompt_{k}'] + rejected_tk_out[k] for k in ['input_ids', 'attention_mask']
        }
        chosen_seqs['labels'] = chosen_seqs['input_ids'][:]
        chosen_seqs['labels'][: len(chosen_tk_out['prompt_input_ids'])] = [
            self.label_pad_token_id
        ] * len(chosen_tk_out['prompt_input_ids'])
        rejected_seqs['labels'] = rejected_seqs['input_ids'][:]
        rejected_seqs['labels'][: len(rejected_tk_out['prompt_input_ids'])] = [
            self.label_pad_token_id
        ] * len(rejected_tk_out['prompt_input_ids'])

        return dict(
            chosen_input_ids = torch.LongTensor(chosen_seqs['input_ids']),
            chosen_attention_mask = torch.LongTensor(chosen_seqs['attention_mask']),
            chosen_labels = torch.LongTensor(chosen_seqs['labels']),
            rejected_input_ids = torch.LongTensor(rejected_seqs['input_ids']),
            rejected_attention_mask = torch.LongTensor(rejected_seqs['attention_mask']),
            rejected_labels = torch.LongTensor(rejected_seqs['labels'])
        )

    def read_hh_rlhf_dataset(self, data_path, train):

        dp_prefix = self.ds_info['prompt_prefix']
        dr_prefix = self.ds_info['response_prefix']
        instruct_pattern = re.compile(f'{dp_prefix}(.*?){dr_prefix}(.*?)', re.S)

        datas = []
        with open(os.path.join(data_path, 'train.jsonl' if train else 'test.jsonl'), 'r', encoding = 'utf-8') as file:
            for index, line in enumerate(file):
                json_data = json.loads(line)

                chosen_splitted = self.split_text(json_data['chosen'], instruct_pattern)
                rejected_splitted = self.split_text(json_data['rejected'], instruct_pattern)

                chosen_prompt, chosen_response = self.add_instruct_prompt(chosen_splitted)
                rejected_prompt, rejected_response = self.add_instruct_prompt(rejected_splitted)

                if chosen_prompt != rejected_prompt:
                    # print(chosen_prompt, rejected_prompt)
                    continue
                tokenizer_out = self.tokenize_single(chosen_prompt, chosen_response, rejected_response)
                datas.append(tokenizer_out)

                if train and index == TEST:
                    break
                if not train and index == TEST // 10:
                    break

        return datas
    
    def get_generator(self):
        
        if hasattr(self, 'eval_datas'):
            return Generator_Dataset(self.train_datas), Generator_Dataset(self.eval_datas)
        else:
            return Generator_Dataset(self.train_datas)

if __name__ == '__main__':

    config = DPO_Config()
    dataset = HumanFeedback_Dataset(config.dateset_cfg, config.dateset_cfg.data_path)
    hf_dataset = dataset.get_generator()
    loader = torch.utils.data.DataLoader(hf_dataset, batch_size = 8, collate_fn = hf_collator)
    for batch in loader:
        print(batch['chosen_input_ids'])
    print(dataset.train_datas)