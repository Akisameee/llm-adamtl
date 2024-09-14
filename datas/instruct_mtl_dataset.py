import torch
from torch.utils.data import Dataset
import os
import csv
import re
from transformers import AutoTokenizer, AutoConfig, GPT2Tokenizer
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datas.dataset_parser import Dataset_Parser
from configs import Instruct_Dataset_Config

class Instruct_MTL_Dataset():

    def __init__(
        self,
        configs: list[Instruct_Dataset_Config]
    ):

        self.dataset_parsers = []
        for config in configs:
            dataset_parser = Dataset_Parser(config = config)
            self.dataset_parsers.append(dataset_parser)
        self.model_info = self.dataset_parsers[0].model_info

        self.tokenize = {
            'prompt_pad': self.tokenize_prompt_pad,
            'prompt_not_pad': self.tokenize_prompt_not_pad,
            'prompt_response': self.tokenize_prompt_response
        }[config.tokenize_type]

    def load(self):
        
        self.datas = {}

    def tokenize_prompt_response():

        pass

    def get_response_label(self, tokenizer_out):

        prompt_len = torch.sum(tokenizer_out['token_type_ids'])
        tokenizer_out['labels'] = torch.where(
            tokenizer_out['input_ids'] == self.tokenizer.pad_token_id,
            -100,
            tokenizer_out['input_ids']
        )
        tokenizer_out['labels'][:, :prompt_len] = -100

        return tokenizer_out

if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained('/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0')
    out = tokenizer.encode_plus(
        'test a to b then c',
        'test d to e then f'
    )
    print(out['input_ids'], tokenizer.decode(out['input_ids']))
    id1 = tokenizer.encode_plus(
        'test a to b then c',
        add_special_tokens = False
    )['input_ids']
    print(id1, tokenizer.decode(id1))
    id2 = tokenizer.encode_plus(
        'test d to e then f',
        add_special_tokens = False
    )['input_ids']
    print(id2, tokenizer.decode(id2))
    outid = tokenizer.build_inputs_with_special_tokens(id1, id2)
    print(outid, tokenizer.decode(outid))

    # dataset = Instruct_MTL_Dataset(
    #     configs = [
    #         Instruct_Dataset_Config(
    #             data_path = '/home'
    #         )
    #     ]
    # )