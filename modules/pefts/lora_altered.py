import sys
import os
sys.path.insert(0, '/home/smliu/llm-adamtl')
import torch
import torch.nn as nn
import math
import numpy as np

from configs.pefts import Peft_Config, Lora_Config, Lora_Altered_Config

from modules.pefts.base import Base_Adapter
from modules.pefts.lora import Lora_Linear
from modules.pefts.utils import compute_conflict_scores, compute_consine_similarities

class Lora_Linear_Altered(Base_Adapter):

    def __init__(
        self,
        config: Lora_Altered_Config,
        base_layer: nn.Module,
    ) -> None:
        
        if not isinstance(base_layer, (nn.Linear, Lora_Linear)):
            raise TypeError(f'Expected base_layer type \'torch.nn.Linear\' or \'Lora_Linear\', but got \'{type(base_layer)}\'.')
        if config.r < 0:
            raise ValueError(f'Expected r >= 0, but got r = {config.r}.')
        if config.pref_dim <= 1:
            raise ValueError(f'Expected pref_dim > 1, but got pref_dim = {config.pref_dim}.')
        if config.pref_r < 0:
            raise ValueError(f'Expected pref_r >= 0, but got pref_r = {config.pref_r}.')
        if config.r == 0 and config.pref_r == 0:
            raise ValueError(f'At least one of r and pref_r should be > 0.')
        self.r = config.r
        self.pref_dim = config.pref_dim
        self.pref_r = config.pref_r

        if isinstance(base_layer, nn.Linear):
            super().__init__(base_layer)
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
            self.lora_alpha = config.lora_alpha
            self.lora_A = nn.Parameter(torch.FloatTensor(self.r + self.pref_dim * self.pref_r, self.in_features))
            self.lora_B = nn.Parameter(torch.FloatTensor(self.out_features, self.r + self.pref_dim * self.pref_r))
            self.reset_lora_weight()
        else:
            raise NotImplementedError

        if config.lora_dropout > 0.0:
            self.dropout = nn.Dropout(p = config.lora_dropout)
        else:
            self.dropout = nn.Identity()
        self.scaling = config.lora_alpha / self.r
        self.merged = False
        self.set_adapter(enable = True)
        self.pref_vec = torch.zeros(self.pref_dim)
        self.name = None

        self.records = {
            'conflict_cos_sims': [],
            'sh_ts_conflict_scores': [],
            'ts_ts_conflict_scores': [],
            'task_flags': []
        }
        self.task_flag = torch.zeros(self.r + self.pref_dim * self.pref_r).long() - 1
        for pref_r_idx in range(self.pref_r):
            self.task_flag[
                self.r + pref_r_idx * self.pref_dim:
                self.r + (pref_r_idx + 1) * self.pref_dim
            ] = torch.arange(self.pref_dim)

    def reset_lora_weight(self, init_weights: bool = True):

        if not init_weights:
            return
        
        nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_task_flag(self):

        return self.task_flag.clone()

    def get_delta_weights(self):

        device = self.lora_B.device
        dtype = self.lora_B.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lora_weight_A = self.lora_A
        lora_weight_B = self.lora_B

        if cast_to_fp32:
            lora_weight_A = lora_weight_A.float()
            lora_weight_B = lora_weight_B.float()

        delta_weights = lora_weight_B @ lora_weight_A * self.scaling

        if cast_to_fp32:
            delta_weights = delta_weights.to(dtype = dtype)
            self.lora_A.data = lora_weight_A.to(dtype)
            self.lora_B.data = lora_weight_B.to(dtype)

        return delta_weights

    def train(self, mode: bool = True):

        self.dropout.train(mode)
        if mode:
            self.unmerge()
        else:
            self.merge()

    def compute_scores(self, grad_dict: dict):

        if self.name is None:
            lora_A_key = list(filter(lambda k: k.endswith('lora_A'), grad_dict.keys()))[0]
            lora_B_key = list(filter(lambda k: k.endswith('lora_B'), grad_dict.keys()))[0]
        else:
            lora_A_key = self.name + '.lora_A'
            lora_B_key = self.name + '.lora_B'

        lora_AB_grads = [torch.cat(
                [
                    lora_A_grad,
                    lora_B_grad.T
                ], dim = 1
            ) for lora_A_grad, lora_B_grad in zip(
                grad_dict[lora_A_key],
                grad_dict[lora_B_key]
            )
        ]
        
        # compute conflict scores
        lora_AB = torch.cat(
            [
                self.lora_A.data,
                self.lora_B.data.T
            ], dim = 1
        )
        conflict_cos_sim = compute_consine_similarities(lora_AB_grads, dim = 1)
        sh_ts_conflict_score, ts_ts_conflict_score = compute_conflict_scores(
            grads = lora_AB_grads,
            params = lora_AB,
            weight = self.pref_vec.to(lora_AB_grads[0].device),
            dim = 1
        )
        self.record_step(**dict(
            conflict_cos_sims = conflict_cos_sim,
            sh_ts_conflict_scores = sh_ts_conflict_score,
            ts_ts_conflict_scores = ts_ts_conflict_score,
            task_flags = self.get_task_flag()
        ))

    def restore_gradient(self, grad_dict: dict):

        if self.name is None:
            lora_A_key = list(filter(lambda k: k.endswith('lora_A'), grad_dict.keys()))[0]
            lora_B_key = list(filter(lambda k: k.endswith('lora_B'), grad_dict.keys()))[0]
        else:
            lora_A_key = self.name + '.lora_A'
            lora_B_key = self.name + '.lora_B'

        lora_A_grads = []
        lora_B_grads = []
        for pref_idx, (lora_A_grad, lora_B_grad) in enumerate(zip(
            grad_dict[lora_A_key],
            grad_dict[lora_B_key]
        )):
            (
                lora_A_grad_mask,
                lora_B_grad_mask
            ) = self.get_grad_mask(pref_idx, device = lora_A_grad.device)
            lora_A_grads.append(lora_A_grad * lora_A_grad_mask)
            lora_B_grads.append(lora_B_grad * lora_B_grad_mask)

        grad_dict[lora_A_key] = torch.stack(lora_A_grads, dim = 0).sum(dim = 0)
        grad_dict[lora_B_key] = torch.stack(lora_B_grads, dim = 0).sum(dim = 0)

        return grad_dict
    
    def get_grad_mask(self, pref_idx: int, device):
        
        device = self.device if device is None else device

        grad_mask_map = torch.zeros(self.pref_dim + 1, device = device)
        grad_mask_map[0] = 1
        if self.pref_vec[pref_idx].item() != 0:
            grad_mask_map[pref_idx + 1] = 1 / self.pref_vec[pref_idx].item()
        grad_mask = grad_mask_map.index_select(0, self.task_flag + 1)

        lora_A_grad_mask = grad_mask.unsqueeze(1).expand(-1, self.in_features)
        lora_B_grad_mask = grad_mask.unsqueeze(0).expand(self.out_features, -1)

        return lora_A_grad_mask, lora_B_grad_mask

    def forward(self, x: torch.Tensor, *args, **kwargs):
        
        if self.disabled:
            if self.merged:
                self.unmerge()
            res = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            res = self.base_layer(x, *args, **kwargs)
        else:
            res = self.base_layer(x, *args, **kwargs)
            res = res + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return res
    
if __name__ == '__main__':

    linear_layer = nn.Linear(in_features = 1024, out_features = 256, bias = False)
    config = Lora_Altered_Config(
        r = 8,
        lora_alpha = 32,
        pref_dim = 4,
        pref_r = 1
    )
    lora_layer = Lora_Linear_Altered(
        config = config,
        base_layer = linear_layer
    )

    test_tensor = torch.rand(4, 1024)
    out = lora_layer.forward(test_tensor)
    print(out)
    lora_layer.merge()
    out = lora_layer.forward(test_tensor)
    print(out)
    lora_layer.unmerge()
    out = lora_layer.forward(test_tensor)
    print(out)
    target = torch.ones(4, 256).float()
    (target - out).sum().backward()
    print(lora_layer.lora_A.weight.grad)
    print(lora_layer.lora_B.weight.grad)
    print(lora_layer.base_layer.weight.grad)
