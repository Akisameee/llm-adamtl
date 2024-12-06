import torch
from torch.utils.data import Dataset
from datas.utils import pad_sequence_side
from typing import Literal, Optional
import os
import random
import re
from transformers import AutoTokenizer, AutoConfig, GPT2Tokenizer
from datasets import load_dataset
from copy import deepcopy

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datas.dataset_parser import Dataset_Parser
from configs import Instruct_Dataset_Config, Instruct_MTL_Config, dataset_infos

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
        self.model_config = AutoConfig.from_pretrained(configs[0].tokenizer_pretrain_path)
        self.max_len = int(min(
            self.tokenizer.model_max_length, self.model_config.max_position_embeddings
        ) / 2)
        
        self.subset_names = [
            os.path.split(config.data_path)[-1]
            for config in configs
        ]
        self.dataset_parsers = []
        for config in configs:
            dataset_parser = Dataset_Parser(config = config, tokenizer = self.tokenizer)
            self.dataset_parsers.append(dataset_parser)
        self.model_info = self.dataset_parsers[0].model_info

        self.pre_tokenize = False
        self.mode = None

    def get_tokenize_func(self):

        if self.mode == 'train':
            self.tokenizer.padding_side = 'right'
            return self.tokenize_prompt_response
        elif self.mode == 'val':
            self.tokenizer.padding_side = 'left'
            return self.tokenize_prompt
        else:
            return NotImplementedError

    def load(
        self,
        mode = 'train',
        max_sample = None,
        pre_tokenize = False
    ):
        self.pre_tokenize = pre_tokenize
        self.mode = mode
        self.tokenize_func = self.get_tokenize_func()

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
            datas.append(self.tokenize_func([text_pairs]))

        return datas

    def tokenize_prompt_response(
        self,
        text_pairs,
        padding = None,
        add_special_tokens = True
    ):

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

            if len(input_ids) > self.max_len:
                (
                    input_ids,
                    attention_mask,
                    token_type_ids
                ) = self.truncate_prompt(
                    input_ids,
                    attention_mask,
                    token_type_ids
                )

        if padding == None:
            tokenizer_out = {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(attention_mask),
                'token_type_ids': torch.LongTensor(token_type_ids)
            }
        else:
            tokenizer_out = {
                'input_ids': torch.LongTensor(self.pad(input_ids, padding_value = self.tokenizer.pad_token_id)),
                'attention_mask': torch.LongTensor(self.pad(attention_mask, padding_value = 0)),
                'token_type_ids': torch.LongTensor(self.pad(token_type_ids, padding_value = 0))
            }
        tokenizer_out = self.get_response_label(tokenizer_out)

        return tokenizer_out
    
    def tokenize_prompt(
        self,
        text_pairs,
        padding = None,
        add_special_tokens = True
    ):

        input_ids = []
        attention_mask = []
        token_type_ids = []
        # add bos token
        if add_special_tokens and self.add_bos_token:
            input_ids.append(self.tokenizer.bos_token_id)
            attention_mask.append(1)
            token_type_ids.append(0)
        
        for idx, (prompt_text, response_text) in enumerate(text_pairs):
            prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt_text))
            input_ids += prompt_ids
            attention_mask += [1] * len(prompt_ids)
            token_type_ids += [0] * len(prompt_ids)

            if idx < len(text_pairs) - 1:
                if add_special_tokens and self.add_sep_token:
                    input_ids.append(self.tokenizer.sep_token_id)
                    attention_mask.append(1)
                    token_type_ids.append(1)
            
                response_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(response_text))
                input_ids += response_ids
                attention_mask += [1] * len(response_ids)
                token_type_ids += [1] * len(response_ids)

                if add_special_tokens and self.add_eos_token:
                    input_ids.append(self.tokenizer.eos_token_id)
                    attention_mask.append(1)
                    token_type_ids.append(1)
            else:
                if add_special_tokens and self.add_eos_token:
                    input_ids.append(self.tokenizer.eos_token_id)
                    attention_mask.append(1)
                    token_type_ids.append(1)
                target = response_text

        if len(input_ids) > self.max_len:
            (
                input_ids,
                attention_mask,
                token_type_ids
            ) = self.truncate_prompt(
                input_ids,
                attention_mask,
                token_type_ids
            )

        if padding == None:
            tokenizer_out = {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(attention_mask),
                'token_type_ids': torch.LongTensor(token_type_ids),
                'target': target
            }
        else:
            tokenizer_out = {
                'input_ids': torch.LongTensor(self.pad(input_ids, padding_value = self.tokenizer.pad_token_id, side = 'left')),
                'attention_mask': torch.LongTensor(self.pad(attention_mask, padding_value = 0, side = 'left')),
                'token_type_ids': torch.LongTensor(self.pad(token_type_ids, padding_value = 0, side = 'left')),
                'target': target
            }

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

    def pad(self, sequences, padding_value = 0, side = Literal['right', 'left']):

        if side == 'right':
            if len(sequences) < self.max_len:
                sequences += [padding_value] * (self.max_len - len(sequences))
            return sequences[: self.max_len]
        else:
            if len(sequences) < self.max_len:
                sequences = [padding_value] * (self.max_len - len(sequences)) + sequences
            return sequences[len(sequences) - self.max_len:]
    
    def truncate_prompt(self, input_ids, *tensors):

        prompt_end_text = self.model_info['prompt_suffix'] + self.model_info['response_prefix']
        prompt_end_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt_end_text))
        prompt_end_len = len(prompt_end_ids)
        for idx in range(len(input_ids) - prompt_end_len, 0, -1):
            if prompt_end_ids == input_ids[idx: idx + prompt_end_len]:
                if idx < len(input_ids) - self.max_len:
                    idx = len(input_ids) - self.max_len
                break
        input_ids = input_ids[: idx - len(input_ids) + self.max_len] + input_ids[idx:]
        input_ids = input_ids[: self.max_len]
        truncated_tensors = []
        for tensor in tensors:
            tensor = tensor[: idx - len(input_ids) + self.max_len] + tensor[idx:]
            tensor = tensor[: self.max_len]
            truncated_tensors.append(tensor)

        return input_ids, *truncated_tensors

    def get_generator(self):

        if self.mode == 'train':
            generator = Instruct_MTL_Train_Generator(
                dataset = self,
                category_padding = True
            )
            return generator
        elif self.mode == 'val':
            generators = []
            for category_idx in range(len(self.all_datas)):
                generators.append(
                    Instruct_MTL_Val_Generator(
                        dataset = self,
                        category_idx = category_idx
                    )
                )
            return generators
        else:
            raise NotImplementedError
        
class Instruct_MTL_Train_Generator(Dataset):

    def __init__(self, dataset: Instruct_MTL_Dataset, category_padding = False) -> None:
        super().__init__()

        self.dataset = dataset
        self.all_datas = self.dataset.all_datas
        self.pre_tokenize = self.dataset.pre_tokenize
        self.category_padding = category_padding
        self.data_lens = [len(datas) for datas in self.all_datas]
        self.data_flags = [torch.zeros(data_len).long() for data_len in self.data_lens]

    def __getitem__(self, index):
        
        items = []
        for cls_idx, datas in enumerate(self.all_datas):
            data_index = index % self.data_lens[cls_idx]
            data_flag = self.data_flags[cls_idx]
            nonzero_indexs = (data_flag == 0).nonzero().squeeze()
            if len(nonzero_indexs.shape) == 0:
                data_flag.zero_()
                nonzero_indexs = (data_flag == 0).nonzero().squeeze()
            if data_flag[data_index]:
                data_index = nonzero_indexs[random.randint(0, nonzero_indexs.size(0) - 1)].item()
            data_flag[data_index] = 1
        
            if not self.pre_tokenize:
                items.append(self.dataset.tokenize_func([datas[data_index]]))
            else:
                items.append(datas[data_index])

        return items
    
    def __len__(self):

        if self.category_padding: 
            return max(self.data_lens)
        else:
            return min(self.data_lens)
    
    def create_mtl_collator(self):

        pad_token_id = self.dataset.tokenizer.pad_token_id

        def instruct_mtl_collator(batch):
        
            all_model_inputs = []
            n_task = len(batch[0])
            for task_idx in range(n_task):
                model_inputs = {
                    'input_ids': pad_sequence_side(
                        [data[task_idx]['input_ids'] for data in batch],
                        batch_first = True,
                        padding_value = pad_token_id
                    ),
                    'attention_mask': pad_sequence_side(
                        [data[task_idx]['attention_mask'] for data in batch],
                        batch_first = True,
                        padding_value = 0
                    ),
                    'token_type_ids': pad_sequence_side(
                        [data[task_idx]['token_type_ids'] for data in batch],
                        batch_first = True,
                        padding_value = 0
                    ),
                    'labels': pad_sequence_side(
                        [data[task_idx]['labels'] for data in batch],
                        batch_first = True,
                        padding_value = -100
                    )
                }
                all_model_inputs.append(model_inputs)

            return all_model_inputs
        
        return instruct_mtl_collator
    
class Instruct_MTL_Val_Generator(Dataset):

    def __init__(self, dataset: Instruct_MTL_Dataset, category_idx: int) -> None:
        super().__init__()

        self.dataset = dataset
        self.data_path = self.dataset.dataset_parsers
        self.datas = self.dataset.all_datas[category_idx]
        self.pre_tokenize = self.dataset.pre_tokenize

    def __getitem__(self, index):
        
        if not self.pre_tokenize:
            return self.dataset.tokenize_func([self.datas[index]])
        else:
            return self.datas[index]
    
    def __len__(self):

        return len(self.datas)
    
    def create_mtl_collator(self):

        pad_token_id = self.dataset.tokenizer.pad_token_id

        def instruct_mtl_collator(batch):
            
            model_inputs = {
                'input_ids': pad_sequence_side(
                    [data['input_ids'] for data in batch],
                    batch_first = True,
                    padding_value = pad_token_id,
                    side = 'left'
                ),
                'attention_mask': pad_sequence_side(
                    [data['attention_mask'] for data in batch],
                    batch_first = True,
                    padding_value = 0,
                    side = 'left'
                ),
                'token_type_ids': pad_sequence_side(
                    [data['token_type_ids'] for data in batch],
                    batch_first = True,
                    padding_value = 0,
                    side = 'left'
                ),
                'target_texts': [data['target'] for data in batch]
            }

            return model_inputs
        
        return instruct_mtl_collator

if __name__ == '__main__':

    # tokenizer = GPT2Tokenizer.from_pretrained('/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0')
    # msg = [
    #     {'role': 'user', 'content': 'Hi there!'},
    #     {'role': 'assistant', 'content': 'Nice to meet you!'},
    #     {'role': 'user', 'content': 'Can I ask a question?'}
    # ]
    # out = tokenizer.apply_chat_template(msg, tokenize = False)
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

    config = Instruct_MTL_Config()
    # config.base_dateset_cfg.tokenizer_pretrain_path = '/home/share/models/huggingface/meta-llama/Llama-2-7b-chat-hf'
    config.base_dateset_cfg.tokenizer_pretrain_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'

    config.base_dateset_cfg.data_path = '/home/smliu/datasets/instruct/bigbench'
    config.dataset_data_paths = []
    for sub_class in dataset_infos[config.base_dateset_cfg.name]['sub_classes']:
        config.dataset_data_paths.append(os.path.join(config.base_dateset_cfg.data_path, sub_class))

    dataset = Instruct_MTL_Dataset(
        configs = config.get_dataset_cfgs()
    )
    dataset.load(max_sample = 100, pre_tokenize = True)
    print(dataset)