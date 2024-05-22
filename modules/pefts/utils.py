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

    if len(weight.shape) > 1:
        weight = weight.squeeze()
    assert len(weight.shape) == 1

    assert len(grads) > 1 and len(grads) == len(weight)
    t_dim = len(grads)
    assert all(t.shape == params.shape and len(t.shape) == 2 for t in grads) and dim < 2
    n_r = grads[0].shape[0 if dim == 1 else 0]
    sh_ts_conflict_scores = torch.zeros(t_dim, n_r).to(weight.device)
    ts_ts_conflict_scores = torch.zeros(t_dim, t_dim, n_r).to(weight.device)
    
    cross_dot_prod = torch.zeros(t_dim, t_dim, n_r)
    for i in range(t_dim):
        for j in range(i, t_dim):
            if i == j:
                cross_dot_prod[i, j, :] = torch.norm(grads[i], dim = dim) ** 2
            else:
                dot_prod = batch_dot(grads[j], grads[i], dim = dim)
                cross_dot_prod[i, j, :] = dot_prod
                cross_dot_prod[j, i, :] = dot_prod
    
    for i, w_i in enumerate(weight):
        sh_ts_penality = 0
        for j, w_j in enumerate(weight):
            for k, w_k in enumerate(weight):
                sh_ts_penality += w_j * w_k * cross_dot_prod[k, j, :]

        sh_ts_conflict_scores[i] = w_i * cross_dot_prod[i, i, :] - sh_ts_penality
    
    for i, w_i in enumerate(weight):
        for j, w_j in enumerate(weight):
            if i == j: continue
            ts_ts_conflict_scores[i, j, :] = w_j * cross_dot_prod[j, j, :] - w_i * cross_dot_prod[i, i, :] - \
                (w_j - w_i) * cross_dot_prod[j, i, :]

    return sh_ts_conflict_scores, ts_ts_conflict_scores

if __name__ == '__main__':

    compute_conflict_scores(
        grads = [torch.nn.init.normal_(torch.rand(8, 2048), std = 1e-4) for i in range(3)],
        params = torch.rand(8, 2048),
        weight = torch.FloatTensor([0.2, 0.5, 0.3]),
        dim = 1
    )