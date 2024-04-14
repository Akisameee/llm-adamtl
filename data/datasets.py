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

from configs import dataset_infos, model_infos

csv.field_size_limit(10000000)
TEST = 128

def has_chinese(str_check):

        for char in str_check:
            if '\u4e00' <= char <= '\u9fa5':
                return True
        return False

class Dataset_Parser(object):

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        dataset_name: str = None,
        remove_chinese: bool = False
    ) -> None:
        
        self.data_dir = data_dir
        self.dataset_name = dataset_name if dataset_name is not None else os.path.split(data_dir)[-1]
        self.dataset_info = dataset_infos[self.dataset_name]
        self.model_info = model_infos[model_name]
        self.remove_chinese = remove_chinese

    def is_data_valid(
        self,
        data: dict,
        remove_chinese: bool = None,
    ):
        
        for k, text in data.items():
            if remove_chinese:
                if has_chinese(text):
                    return False
                
        return True
        
    def parse_dataset(
        self,
        sub_paths = None,
        mode = 'train'
    ):
        if isinstance(sub_paths, str):
            sub_paths = [sub_paths]
        
        datas = []
        if self.dataset_name == 'hh-rlhf':
            for sub_path in sub_paths:
                datas += self.parse_hh_rlhf_dataset(os.path.join(self.data_dir, sub_path), mode = mode)
        elif self.dataset_name == 'sharegpt':
            datas = self.parse_sharegpt_dataset(self.data_dir, mode = mode)

        return datas

    def parse_sharegpt_dataset(self, data_path, mode):
        
        prompt_pattern = r'<s>Human: (.*?)</s>'
        response_pattern = r'<s>Assistant: (.*?)</s>'

        datas = []
        if mode == 'train':
            file_name = 'train_sft.csv'
        elif mode == 'eval':
            file_name = 'dev_sft.csv'
        elif mode == 'sharegpt':
            file_name = 'dev_sft_sharegpt.csv'
        else:
            raise NotImplementedError
        with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as file:
            for index, line in enumerate(csv.reader(file)):
                if index == 0:
                    continue
                line = line[0].strip('\n')

                prompt_text = re.findall(prompt_pattern, line, re.DOTALL)[0]
                response_text = re.findall(response_pattern, line, re.DOTALL)[0]
                if len(prompt_text) <= 0 or len(response_text) <= 0:
                    continue
                (
                    prompt_text,
                    response_text
                ) = self.add_instruct_prompt(
                    [(prompt_text, response_text)]
                )

                data = {
                    'prompt': prompt_text,
                    'response': response_text
                }
                if not self.is_data_valid(data, remove_chinese = self.remove_chinese):
                    continue
                
                datas.append(data)

                if mode and len(datas) == TEST:
                        break
                if not mode and len(datas) == TEST // 10:
                    break
        
        return datas

    def parse_hh_rlhf_dataset(self, data_path, mode):

        dataset_info = dataset_infos['hh-rlhf']
        dp_prefix = dataset_info['prompt_prefix']
        dr_prefix = dataset_info['response_prefix']
        instruct_pattern = re.compile(f'{dp_prefix}(.*?){dr_prefix}(.*?)', re.S)

        datas = []
        if mode == 'train':
            file_name = 'train.jsonl'
        elif mode == 'eval':
            file_name = 'test.jsonl'
        else:
            raise NotImplementedError
        with open(os.path.join(data_path, file_name), 'r', encoding = 'utf-8') as file:
            for index, line in enumerate(file):
                json_data = json.loads(line)

                chosen_splitted = self.hh_rlhf_split_text(json_data['chosen'], instruct_pattern)
                rejected_splitted = self.hh_rlhf_split_text(json_data['rejected'], instruct_pattern)

                chosen_prompt, chosen_response = self.add_instruct_prompt(chosen_splitted)
                rejected_prompt, rejected_response = self.add_instruct_prompt(rejected_splitted)

                if chosen_prompt != rejected_prompt:
                    continue

                data = {
                    'prompt': chosen_prompt,
                    'chosen': chosen_response,
                    'rejected': rejected_response
                }
                if not self.is_data_valid(data, remove_chinese = self.remove_chinese):
                    continue
                
                datas.append(data)

                if mode and len(datas) == TEST:
                        break
                if not mode and len(datas) == TEST // 10:
                    break
        
        return datas

    def hh_rlhf_split_text(self, text, instruct_pattern):

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

    def add_instruct_prompt(self, text_splitted):
            
        prompt_full = ''
        response_full = ''
        for idx, (prompt_text, response_text) in enumerate(text_splitted):
            prompt_text = prompt_text.strip()
            response_text = response_text.strip()
            prompt_full += self.model_info['prompt_prefix'] + prompt_text
            if idx == len(text_splitted) - 1:
                prompt_full += self.model_info['response_prefix']
                response_full = response_text
            else:
                prompt_full += self.model_info['response_prefix'] + response_text

        return prompt_full, response_full

if __name__ == '__main__':

    data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    # data_path = os.path.join('/home', 'smliu', 'datasets', 'instruct', 'sharegpt')
    model_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    model_path = '/home/share/models/huggingface/bit-dny/MindLLM'
    model_name = os.path.split(model_path)[-1]
    model_info = model_infos[model_name]
    parser = Dataset_Parser(
        data_dir = data_path,
        model_name = model_name,
        remove_chinese = False
    )
    res = parser.parse_dataset(
        sub_paths = ['helpful-base'],
        mode = 'train'
    )
    print(res)