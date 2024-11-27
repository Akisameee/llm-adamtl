import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import csv
import re
import numpy as np
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
from datasets import load_dataset

from configs import dataset_infos, model_infos, Dataset_Config, Panacea_PPO_Config

csv.field_size_limit(10000000)
TEST = 0

def has_chinese(str_check):

        for char in str_check:
            if '\u4e00' <= char <= '\u9fa5':
                return True
        return False

class Dataset_Parser(object):

    def __init__(
        self,
        config: Dataset_Config,
        tokenizer: PreTrainedTokenizer
    ) -> None:
        
        self.data_dir = config.data_path
        self.sub_paths = config.sub_data_path
        self.dataset_name = config.name
        self.dataset_info = config.dataset_info
        self.model_info = config.model_info
        self.remove_chinese = config.remove_chinese

        self.tokenizer = tokenizer

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
        mode = 'train',
        max_sample = None
    ):
        if isinstance(self.sub_paths, str):
            self.sub_paths = [self.sub_paths]
        
        max_sample = max_sample if max_sample is not None else 0
        datas = []
        if self.dataset_name == 'hh-rlhf':
            for sub_path in self.sub_paths:
                datas += self.parse_hh_rlhf_dataset(
                    os.path.join(self.data_dir, sub_path),
                    mode = mode,
                    max_sample = max_sample
                )
        elif self.dataset_name == 'sharegpt':
            datas = self.parse_sharegpt_dataset(
                self.data_dir,
                mode = mode,
                max_sample = max_sample
            )
        elif self.dataset_name == 'infinity-instruct':
            datas += self.parse_infinity_instruct_dataset(
                os.path.join(self.data_dir),
                mode = mode,
                max_sample = max_sample
            )
        elif self.dataset_name == 'sciq':
            datas += self.parse_sciqa_dataset(
                self.data_dir,
                mode = mode,
                max_sample = max_sample
            )
        elif self.dataset_name == 'bigbench':
            datas += self.parse_bigbench_dataset(
                self.data_dir,
                mode = mode,
                max_sample = max_sample
            )
        else:
            raise NotImplementedError

        return datas

    def parse_sharegpt_dataset(self, data_path, mode, max_sample = None):
        
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

                if mode and len(datas) == max_sample:
                    break
                # if not mode and len(datas) == max_sample // 10:
                #     break
        
        return datas

    def parse_hh_rlhf_dataset(self, data_path, mode, max_sample = None):

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

                if mode and len(datas) == max_sample:
                    break
                # if not mode and len(datas) == max_sample // 10:
                #     break
        
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
    
    def parse_infinity_instruct_dataset(self, data_path, mode = None, max_sample = None):

        dataset_raw = load_dataset(data_path, split = 'train')

        dataset_df = dataset_raw.data.to_pandas()
        dataset_df = dataset_df[dataset_df['langdetect'] == 'en']
        if max_sample != 0:
            dataset_df = dataset_df[: max_sample + 100]
        dataset_dicts = dataset_df.to_dict(orient = 'records')
        
        chats = []
        for dataset_dict in dataset_dicts:
            conversations = dataset_dict['conversations']
            if conversations.shape[0] % 2 != 0:
                continue
            msg = []
            role_flag = 0
            for text in conversations:
                if text['from'] == 'human':
                    if role_flag != 0:
                        continue
                    role_flag = 1
                    prompt_splitted = text['value']
                else:
                    if role_flag != 1:
                        continue
                    role_flag = 0
                    msg += [
                        {'role': 'user', 'content': prompt_splitted},
                        {'role': 'assistant', 'content': text['value']}
                    ]
                
            chat = self.apply_chat_template(msg, merge = False)
            chats.append(chat)

            if len(chats) == max_sample:
                break

        return chats
    
    def parse_sciqa_dataset(self, data_path, mode = 'train', max_sample = None):

        topic = os.path.split(data_path)[-1]
        json_path = os.path.join(data_path, f'sci_qa_{topic}_{mode}.json')
        chats = []
        with open(json_path, 'r') as f:
            raw_datas = json.load(f)
            for raw_data in raw_datas:
                # data = self.add_instruct_prompt(
                #     [(raw_data['instruct'], raw_data['response'])]
                # )
                chat = self.apply_chat_template([
                    {'role': 'user', 'content': raw_data['instruct']},
                    {'role': 'assistant', 'content': raw_data['response']}
                ])
                chats.append(chat)

                if len(chats) == max_sample:
                    break

        return chats
    
    def parse_bigbench_dataset(self, data_path, mode = 'train', max_sample = None):
        
        file_name = 'train.jsonl' if mode == 'train' else 'validation.jsonl'
        json_path = data_path + f'_{file_name}'
        chats = []
        with open(json_path, 'r') as f:
            for line in f.readlines():
                raw_data = json.loads(line)
                chat = self.apply_chat_template([
                    {'role': 'user', 'content': raw_data['instruction']},
                    {'role': 'assistant', 'content': raw_data['references'][0]}
                ])
                chats.append(chat)

                if len(chats) == max_sample:
                    break

        return chats

    def apply_chat_template(self, msg, merge = True):

        if self.tokenizer.chat_template is not None:
            if merge:
                assert msg[-1]['role'] == 'assistant'
                return (
                    self.tokenizer.apply_chat_template(msg[: -1], tokenize = False),
                    self.tokenizer.apply_chat_template([msg[-1]], tokenize = False)
                )
            else:
                return self.tokenizer.apply_chat_template(msg, tokenize = False)
        else:
            text_splitted = []
            for idx in range(0, len(msg), 2):
                p_msg = msg[idx]
                r_msg = msg[idx + 1]
                assert p_msg['role'] == 'user' and r_msg['role'] == 'assistant'
                text_splitted.append((p_msg['content'], r_msg['content']))

            return self.add_instruct_prompt(text_splitted, merge = merge)

    def add_instruct_prompt(self, text_splitted, merge = True):
        
        if merge:
            prompt_full = ''
            response_full = ''
            for idx, (prompt_text, response_text) in enumerate(text_splitted):
                prompt_text = prompt_text.strip()
                response_text = response_text.strip()
                prompt_full += self.model_info['prompt_prefix'] + prompt_text + self.model_info['prompt_suffix']
                if idx == len(text_splitted) - 1:
                    prompt_full += self.model_info['response_prefix']
                    response_full = response_text
                else:
                    prompt_full += self.model_info['response_prefix'] + response_text + self.model_info['response_suffix']

            return prompt_full, response_full
        else:
            res = []
            for idx, (prompt_text, response_text) in enumerate(text_splitted):
                prompt_text = prompt_text.strip()
                response_text = response_text.strip()
                res.append(
                    (
                        self.model_info['prompt_prefix'] + prompt_text + self.model_info['prompt_suffix'],
                        self.model_info['response_prefix'] + response_text + self.model_info['response_suffix']
                    )
                )
            
            return res

if __name__ == '__main__':

    config = Panacea_PPO_Config()
    dataset_cfg = config.dateset_cfg
    dataset_cfg.data_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')
    # data_path = os.path.join('/home', 'smliu', 'datasets', 'instruct', 'sharegpt')
    model_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    model_path = '/home/share/models/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    dataset_cfg.tokenizer_pretrain_path = model_path

    dataset_cfg.sub_data_path = ['helpful-base']

    parser = Dataset_Parser(
        dataset_cfg
    )
    res = parser.parse_dataset(
        mode = 'train'
    )
    print(res)