import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import copy

from configs.pefts import Peft_Config, Lora_Config, Panacea_SVD_Config
from modules.pefts.base import Base_Adapter

def get_adapter_iter(model: nn.Module, return_name: bool = False):

    if not return_name:
        return filter(lambda m: isinstance(m, Base_Adapter), model.modules())
    else:
        return filter(lambda m: isinstance(m[1], Base_Adapter), model.named_modules())

def freeze_except_adapters():
    pass

def set_all_adapters(
    model: nn.Module,
    enable: bool,
    base_enable: bool = False
):
    for module in model.children():
        if isinstance(module, Base_Adapter):
            module.set_adapter(enable = enable, base_enable = base_enable)
        elif len(list(module.children())) > 0:
            set_all_adapters(module, enable = enable, base_enable = base_enable)

def compute_consine_similarities(tensors: list[torch.FloatTensor], dim: int):
    
    assert len(tensors) > 1
    cosine_similarities = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            x, y = tensors[i], tensors[j]
            cosine_similarities.append(F.cosine_similarity(x, y, dim = dim))

    cosine_similarities = torch.stack(cosine_similarities, dim = 0).mean(dim = 0)
    return cosine_similarities