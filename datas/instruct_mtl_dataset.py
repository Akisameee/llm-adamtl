import torch
from torch.utils.data import Dataset
import os
import csv
import re
from transformers import AutoTokenizer, AutoConfig, GPT2Tokenizer
from datasets import load_dataset
from copy import deepcopy

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datas.dataset_parser import Dataset_Parser
from configs import Instruct_Dataset_Config

def instruct_mtl_collator(batch):
    
    all_model_inputs = []
    n_task = len(batch[0])
    for task_idx in range(n_task):
        model_inputs = {
            'input_ids': torch.stack([data[task_idx]['input_ids'] for data in batch], dim = 0),
            'attention_mask': torch.stack([data[task_idx]['attention_mask'] for data in batch], dim = 0),
            'token_type_ids': torch.stack([data[task_idx]['token_type_ids'] for data in batch], dim = 0),
            'labels': torch.stack([data[task_idx]['labels'] for data in batch], dim = 0)
        }
        all_model_inputs.append(model_inputs)

    return all_model_inputs

class Instruct_MTL_Generator(Dataset):

    def __init__(self, all_datas, tokenize_func = None) -> None:
        super().__init__()

        self.all_datas = all_datas
        self.tokenize_func = tokenize_func

    def __getitem__(self, index):

        if self.tokenize_func is not None:
            return [self.tokenize_func(datas[index]) for datas in self.all_datas]
        else:
            return [datas[index] for datas in self.all_datas]
    
    def __len__(self):

        return min([len(datas) for datas in self.all_datas])

class Instruct_MTL_Dataset():

    def __init__(
        self,
        configs: list[Instruct_Dataset_Config],
        # group_key_labels: tuple
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(configs[0].tokenizer_pretrain_path, padding_side = configs[0].padding_side)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.add_bos_token = self.tokenizer.bos_token_id is not None
        self.add_sep_token = self.tokenizer.sep_token_id is not None
        self.add_eos_token = self.tokenizer.eos_token_id is not None
        self.max_len = configs[0].max_len

        self.dataset_parsers = []
        for config in configs:
            dataset_parser = Dataset_Parser(config = config)
            self.dataset_parsers.append(dataset_parser)
        self.model_info = self.dataset_parsers[0].model_info

        self.tokenize = {
            'prompt_response': self.tokenize_prompt_response
        }[config.tokenize_type]
        self.pre_tokenize = False

    def load(
        self,
        mode = 'train',
        max_sample = None,
        pre_tokenize = False
    ):
        self.pre_tokenize = pre_tokenize
        self.all_datas = []
        all_texts = []
        if len(self.dataset_parsers) > 1:
            for dataset_parser in self.dataset_parsers:
                texts = dataset_parser.parse_dataset(mode = mode, max_sample = max_sample)
                all_texts.append(texts)
        else:
            all_texts = self.dataset_parsers[0].parse_dataset(mode = mode, max_sample = max_sample)
        for texts in all_texts:
            if pre_tokenize:
                self.all_datas.append(self.tokenize_parsed_texts(texts))
            else:
                self.all_datas.append(texts)

    def tokenize_parsed_texts(self, texts):

        datas = []
        for text_pairs in texts:
            datas.append(self.tokenize(text_pairs))

        return datas

    def tokenize_prompt_response(self, text_pairs, add_special_tokens = True):

        input_ids = []
        attention_mask = []
        token_type_ids = []
        # add bos token
        if add_special_tokens and self.add_bos_token:
            input_ids.append(self.tokenizer.bos_token_id)
            attention_mask.append(1)
            token_type_ids.append(0)
        
        for (prompt_text, response_text) in text_pairs:
            prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt_text))
            response_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(response_text))
            input_ids += prompt_ids
            attention_mask += [1] * len(prompt_ids)
            token_type_ids += [0] * len(prompt_ids)

            if add_special_tokens and self.add_sep_token:
                input_ids.append(self.tokenizer.sep_token_id)
                attention_mask.append(1)
                token_type_ids.append(1)
            
            input_ids += response_ids
            attention_mask += [1] * len(response_ids)
            token_type_ids += [1] * len(response_ids)

            if add_special_tokens and self.add_eos_token:
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)
                token_type_ids.append(1)

        tokenizer_out = {
            'input_ids': torch.LongTensor(self.pad(input_ids, pad_id = self.tokenizer.pad_token_id)),
            'attention_mask': torch.LongTensor(self.pad(input_ids, pad_id = 0)),
            'token_type_ids': torch.LongTensor(self.pad(input_ids, pad_id = 0))
        }
        tokenizer_out = self.get_response_label(tokenizer_out)

        return tokenizer_out

    def get_response_label(self, tokenizer_out):

        tokenizer_out['labels'] = torch.where(
            torch.bitwise_or(
                tokenizer_out['input_ids'] == self.tokenizer.pad_token_id,
                tokenizer_out['token_type_ids'] == 0
            ),
            -100,
            tokenizer_out['input_ids']
        )

        return tokenizer_out

    def pad(self, data_list, pad_id = 0):

        if len(data_list) < self.max_len:
            data_list += [pad_id] * (self.max_len - len(data_list))
        return data_list[: self.max_len]

    def get_generator(self):

        generator = Instruct_MTL_Generator(
            all_datas = self.all_datas,
            tokenize_func = None if self.pre_tokenize else self.tokenize
        )
        return generator

if __name__ == '__main__':

    # tokenizer = GPT2Tokenizer.from_pretrained('/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0')
    # out = tokenizer.encode_plus(
    #     'test a to b then c',
    #     'test d to e then f'
    # )
    # print(out['input_ids'], tokenizer.decode(out['input_ids']))
    # id1 = tokenizer.encode_plus(
    #     'test a to b then c',
    #     add_special_tokens = False
    # )['input_ids']
    # print(id1, tokenizer.decode(id1))
    # id2 = tokenizer.encode_plus(
    #     'test d to e then f',
    #     add_special_tokens = False
    # )['input_ids']
    # print(id2, tokenizer.decode(id2))
    # outid = tokenizer.build_inputs_with_special_tokens(id1, id2)
    # print(outid, tokenizer.decode(outid))

    config1 = Instruct_Dataset_Config(
        data_path = '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/7M_domains/code',
        tokenizer_pretrain_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0',
        tokenize_type = 'prompt_response'
    )
    config2 = Instruct_Dataset_Config(
        data_path = '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/7M_domains/subjective',
        tokenizer_pretrain_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0',
        tokenize_type = 'prompt_response'
    )
    # config = Instruct_Dataset_Config(
    #     data_path = '/home/smliu/datasets/instruct/BAAI/Infinity-Instruct/3M',
    #     tokenizer_pretrain_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0',
    #     tokenize_type = 'prompt_response'
    # )

    dataset = Instruct_MTL_Dataset(
        configs = [config1, config2]
    )
    dataset.load(max_sample = 100)
    print(dataset)