import torch
from torch.utils.data import Dataset
import os
import csv
import re
from transformers import AutoTokenizer, AutoConfig

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import SFT_Train_Config, Instruct_Dataset_Config, RLHF_Config


csv.field_size_limit(10000000)
TEST = 1000

def instruct_collator(batch):

    input_ids = [data[0].squeeze() for data in batch]
    attention_masks = [data[1].squeeze() for data in batch]
    prompt_texts = [data[2] for data in batch]

    return input_ids, attention_masks, prompt_texts

def has_chinese(str_check):

    for char in str_check:
        if '\u4e00' <= char <= '\u9fa5':
            return True
    return False

class Generator_Dataset(Dataset):

    def __init__(self, datas):
        
        self.datas = datas

    def __getitem__(self, index):

        return tuple(map(lambda t: t[index], self.datas))
    
    def __len__(self):

        return len(self.datas[0])

class Instruct_Dataset():

    def __init__(
            self,
            config: Instruct_Dataset_Config,
            data_path: str
        ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_pretrain_path, padding_side = config.padding_side)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = config.max_len

        self.data_path = data_path
        self.prompt_only = True if config.tokenize_type in ['prompt_pad', 'prompt_not_pad'] else False
        self.tokenize = {
            'prompt_pad': self.tokenize_prompt_pad,
            'prompt_not_pad': self.tokenize_prompt_not_pad,
            'prompt_response': self.tokenize_prompt_response
        }[config.tokenize_type]
        self.remove_chinese = config.remove_chinese
        if config.tokenizer_pretrain_path.endswith('LocutusqueXFelladrin-TinyMistral248M-Instruct'):
            self.add_instruct_prompt = lambda p, r: ('<|USER|> ' + p, '<|ASSISTANT|> ' + r) \
            if not self.prompt_only else ('<|USER|> ' + p + ' <|ASSISTANT|> ', '<|ASSISTANT|> ' + r)
        else:
            self.add_instruct_prompt = lambda p, r: ('Human: ' + p, 'Assistant: ' + r) \
            if not self.prompt_only else ('Human: ' + p + ' Assistant: ', 'Assistant: ' + r)

        self.datas = self.read_sharegpt_dataset(self.data_path)
        
    def tokenize_prompt_pad(self, prompt_texts, response_texts):
        
        tokenizer_out = self.tokenizer.batch_encode_plus(
            prompt_texts,
            padding = True,
            truncation = True,
            max_length = self.max_len,
            return_tensors = 'pt',
            return_token_type_ids = False
        )
        input_ids = tokenizer_out['input_ids']
        attention_mask = tokenizer_out['attention_mask']

        return [input_ids, attention_mask, prompt_texts]
    
    def tokenize_prompt_not_pad(self, prompt_texts, response_texts):

        input_ids = []
        attention_mask = []
        for prompt_text in prompt_texts:
            tokenizer_out = self.tokenizer.encode_plus(
                prompt_text,
                padding = 'do_not_pad',
                truncation = True,
                max_length = self.max_len,
                return_tensors = 'pt',
                return_token_type_ids = False
            )
            if tokenizer_out['input_ids'].size(-1) > 0.75 * self.max_len:
                continue
            input_ids.append(tokenizer_out['input_ids'])
            attention_mask.append(tokenizer_out['attention_mask'])

        return [input_ids, attention_mask, prompt_texts]
    
    def tokenize_prompt_response(self, prompt_texts, response_texts):

        datas = []
        for prompt_text, response_text in zip(prompt_texts, response_texts):
            tokenizer_out = self.tokenizer.encode_plus(
                prompt_text,
                response_text,
                padding = True,
                truncation = True,
                max_length = self.max_len,
                return_tensors = 'pt',
                return_token_type_ids = True
            )
            tokenizer_out = self.get_response_label(tokenizer_out)
            tokenizer_out['prompt_text'] = prompt_text
            tokenizer_out['response_text'] = response_text
            datas.append(tokenizer_out)

        return datas

    def get_response_label(self, tokenizer_out):

        prompt_len = torch.sum(tokenizer_out['token_type_ids'])
        tokenizer_out['labels'] = torch.where(
            tokenizer_out['input_ids'] == self.tokenizer.pad_token_id,
            -100,
            tokenizer_out['input_ids']
        )
        tokenizer_out['labels'][:, :prompt_len] = -100

        return tokenizer_out
        
    def read_sharegpt_dataset(self, data_path):
        
        prompt_texts = []
        response_texts = []
        prompt_pattern = r'<s>Human: (.*?)</s>'
        response_pattern = r'<s>Assistant: (.*?)</s>'

        with open(data_path, 'r', encoding='utf-8') as file:
            for index, line in enumerate(csv.reader(file)):
                if index == 0:
                    continue
                line = line[0].strip('\n')

                prompt_text = re.findall(prompt_pattern, line, re.DOTALL)[0]
                response_text = re.findall(response_pattern, line, re.DOTALL)[0]
                (
                    prompt_text,
                    response_text
                ) = self.add_instruct_prompt(
                    prompt_text,
                    response_text
                )

                if len(prompt_text) <= 0 or len(response_text) <= 0:
                    continue

                if self.remove_chinese:
                    if has_chinese(prompt_text):
                        continue
                
                prompt_texts.append(prompt_text)
                response_texts.append(response_text)
                if len(prompt_texts) == TEST:
                    break
        
        if len(prompt_texts) > 0:
            datas = self.tokenize(prompt_texts, response_texts)
        else:
            datas = None
        return datas

    def get_generator(self):

        return Generator_Dataset(self.datas)

if __name__ == '__main__':

    config = RLHF_Config()
    config.dateset_cfg.tokenize_type = 'prompt_pad'
    dataset = Instruct_Dataset(config.dateset_cfg, config.dateset_cfg.train_data_path)
    instruct_ds = dataset.get_generator()
    loader = torch.utils.data.DataLoader(instruct_ds, batch_size = 8)
    for batch in loader:
        print(batch[0].shape, batch[1].shape)
    print(dataset.datas)
