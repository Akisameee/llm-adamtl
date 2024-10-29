import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import csv
import re
import random
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

topic_names = ['physics', 'chemistry', 'biology', 'geography']

ds_split = 'validation'
sciq_ds = load_dataset('/home/smliu/datasets/instruct/sciq-topic/origin', split = ds_split)
sciq_ds = sciq_ds.data.to_pandas()
sciq_dict = sciq_ds.to_dict(orient = 'records')

qa_pairs_topic = [[], [], [], []]
for sci_qa in sciq_dict:
    instruct = ''
    response = ''
    question = sci_qa['question']
    support = sci_qa['support']
    choices = [
        sci_qa['correct_answer'],
        sci_qa['distractor1'],
        sci_qa['distractor2'],
        sci_qa['distractor3']
    ]
    random.shuffle(choices)
    answer_idx = choices.index(sci_qa['correct_answer'])
    answer_choice = {
        0:'A', 1:'B', 2:'C', 3:'D'
    }[answer_idx]

    instruct = f'{question} Choose the correct answer: ' + \
        f'A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}'
    response = f'{answer_choice}. {choices[answer_idx]}.' + (f' {support}' if len(support) > 0 else '')

    qa_pairs_topic[topic_names.index(sci_qa['topic'])].append({
        'instruct': instruct,
        'response': response
    })

for idx, qa_pairs in enumerate(qa_pairs_topic):
    with open(f'./sci_qa_{topic_names[idx]}_{ds_split}.json', 'w') as f:
        json.dump(qa_pairs, f)

for idx, qa_pairs in enumerate(qa_pairs_topic):
    with open(f'./sci_qa_{topic_names[idx]}_{ds_split}.json', 'r') as f:
        data = json.load(f)
        print(data[:3])