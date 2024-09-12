import torch
from torch.utils.data import Dataset
import os
import csv
import re
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datas.dataset_parser import Dataset_Parser
from datas.instruct_dataset import Instruct_Dataset
from configs import SFT_Train_Config, Instruct_Dataset_Config, RLHF_Config

class Instruct_MTL_Dataset(Instruct_Dataset):

    def __init__(self, config: Instruct_Dataset_Config):
        super().__init__(config)

    def load():
        pass

if __name__ == '__main__':

    dataset = load_dataset('BAAI/Infinity-Instruct', '7M_domains', split = 'train')
    print(dataset)