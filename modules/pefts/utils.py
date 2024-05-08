import sys
import os
sys.path.insert(0, '/home/smliu/RLHF')
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import copy
import itertools
import string

from configs.pefts import Peft_Config, Lora_Config, SVD_Lora_Config
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

def compute_consine_similarities(grads: list[torch.FloatTensor], dim: int):
    
    assert len(grads) > 1
    cosine_similarities = []
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            x, y = grads[i], grads[j]
            cosine_similarities.append(F.cosine_similarity(x, y, dim = dim))

    cosine_similarities = torch.stack(cosine_similarities, dim = 0).mean(dim = 0)
    return cosine_similarities

def batch_dot(a: torch.Tensor, b: torch.Tensor, dim: int):

    input_shape = ''.join(itertools.islice(
        itertools.cycle(string.ascii_lowercase),
        len(a.shape)
    ))
    output_shape = input_shape.replace(input_shape[dim], '')
    return torch.einsum(
        f'{input_shape},{input_shape}->{output_shape}',
        a, b
    )

def compute_conflict_scores(
    grads: list[torch.FloatTensor],
    params: torch.FloatTensor,
    weight: torch.Tensor,
    dim: int
):

    assert len(grads) > 1 and len(grads) == len(weight)
    t_dim = len(grads)
    assert len(weight.squeeze().shape) == 1
    assert all(t.shape == params.shape and len(t.shape) == 2 for t in grads) and dim < 2
    sh_ts_conflict_scores = torch.zeros(t_dim, grads[0].shape[0 if dim == 1 else 0]).to(weight.device)
    
    magnitudes = [torch.norm(grad, dim = dim) ** 2 for grad in grads]
    # cross_dot_prod = [[torch.einsum() for j in range(i, t_dim)] for i in range(t_dim - 1)]
    for i in range(t_dim):
        for j, w_task in enumerate(weight.squeeze()):
            if i == j: continue
            sh_ts_conflict_scores[i] += w_task * (magnitudes[i] - batch_dot(grads[j], grads[i], dim = dim))

    return sh_ts_conflict_scores

if __name__ == '__main__':

    compute_conflict_scores(
        grads = [torch.nn.init.normal_(torch.rand(8, 2048), std = 1e-4) for i in range(3)],
        params = torch.rand(8, 2048),
        weight = torch.FloatTensor([0.2, 0.5, 0.3]),
        dim = 1
    )