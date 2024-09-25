import sys
sys.path.insert(0, '/home/smliu/RLHF')
import torch
import torch.nn as nn
import math
import numpy as np
import random
# from configs.peft_configs import Peft_Config, Lora_Config
import peft

from configs.pefts import SVD_Lora_Config
from modules.pefts.base import Base_Adapter
from modules.pefts.lora import Lora_Linear
from modules.pefts.panacea_svd import SVD_Lora_Linear
from modules.pefts.utils import compute_consine_similarities, compute_conflict_scores

class SVD_Lora_Linear_Altered(Base_Adapter):

    def __init__(
        self,
        config: SVD_Lora_Config,
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
            self.lora_diag = nn.Parameter(torch.zeros(self.r + self.pref_dim * self.pref_r, 1), requires_grad = True)
            # self.pref_scaling = nn.Parameter(torch.FloatTensor(self.pref_r), requires_grad = True)
            self.reset_lora_weight(init_strategy = config.init_strategy)
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
            'diags': [],
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

    def reset_lora_weight(self, init_strategy = None):
        
        # nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
        nn.init.normal_(self.lora_A, mean = 0.0, std = 0.02)
        if init_strategy == 'b_zero':
            nn.init.zeros_(self.lora_B)
        else:
            nn.init.normal_(self.lora_B, mean = 0.0, std = 0.02)
        
        if init_strategy == 'diag_zero':
            nn.init.zeros_(self.lora_diag)
        else:
            nn.init.normal_(self.lora_diag, mean = 0.0, std = 0.02)
            # nn.init.normal_(self.lora_diag[-self.pref_dim * self.pref_r:], mean = 0.0, std = 0.5)
            # nn.init.normal_(self.lora_diag, mean = 0.0, std = 0.5)

    def get_task_flag(self):

        return self.task_flag.clone()
    
    def compute_scores(self, grad_dict: dict):

        if self.name is None:
            lora_A_key = list(filter(lambda k: k.endswith('lora_A'), grad_dict.keys()))[0]
            lora_B_key = list(filter(lambda k: k.endswith('lora_B'), grad_dict.keys()))[0]
            lora_diag_key = list(filter(lambda k: k.endswith('lora_diag'), grad_dict.keys()))[0]
        else:
            lora_A_key = self.name + '.lora_A'
            lora_B_key = self.name + '.lora_B'
            lora_diag_key = self.name + '.lora_diag'

        lora_AB_grads = [torch.cat(
                [
                    lora_A_grad,
                    lora_diag_grad,
                    lora_B_grad.T
                ], dim = 1
            ) for lora_A_grad, lora_diag_grad, lora_B_grad in zip(
                grad_dict[lora_A_key],
                grad_dict[lora_diag_key],
                grad_dict[lora_B_key]
            )
        ]
        
        # compute conflict scores
        lora_AB = torch.cat(
            [
                self.lora_A.data,
                self.lora_diag.data,
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
            diags = self.lora_diag.data.squeeze(),
            conflict_cos_sims = conflict_cos_sim,
            sh_ts_conflict_scores = sh_ts_conflict_score,
            ts_ts_conflict_scores = ts_ts_conflict_score,
            task_flags = self.get_task_flag()
        ))

    def restore_gradient(self, grad_dict: dict):

        if self.name is None:
            lora_A_key = list(filter(lambda k: k.endswith('lora_A'), grad_dict.keys()))[0]
            lora_B_key = list(filter(lambda k: k.endswith('lora_B'), grad_dict.keys()))[0]
            lora_diag_key = list(filter(lambda k: k.endswith('lora_diag'), grad_dict.keys()))[0]
        else:
            lora_A_key = self.name + '.lora_A'
            lora_B_key = self.name + '.lora_B'
            lora_diag_key = self.name + '.lora_diag'

        lora_A_grads = []
        lora_B_grads = []
        lora_diag_grads = []
        for pref_idx, (lora_A_grad, lora_diag_grad, lora_B_grad) in enumerate(zip(
            grad_dict[lora_A_key],
            grad_dict[lora_diag_key],
            grad_dict[lora_B_key]
        )):
            (
                lora_A_grad_mask,
                lora_B_grad_mask,
                lora_diag_grad_mask
            ) = self.get_grad_mask(pref_idx, device = lora_A_grad.device)
            lora_A_grads.append(lora_A_grad * lora_A_grad_mask)
            lora_B_grads.append(lora_B_grad * lora_B_grad_mask)
            lora_diag_grads.append(lora_diag_grad * lora_diag_grad_mask)

        grad_dict[lora_A_key] = torch.stack(lora_A_grads, dim = 0).sum(dim = 0)
        grad_dict[lora_B_key] = torch.stack(lora_B_grads, dim = 0).sum(dim = 0)
        grad_dict[lora_diag_key] = torch.stack(lora_diag_grads, dim = 0).sum(dim = 0)

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
        lora_diag_grad_mask = grad_mask.unsqueeze(1)

        return lora_A_grad_mask, lora_B_grad_mask, lora_diag_grad_mask

    @torch.no_grad()
    def get_delta_weights(self):

        pref_vecs = torch.cat(
            [torch.Tensor([1]).to(self.pref_vec.device), self.pref_vec],
            dim = 0
        ).index_select(0, self.task_flag + 1).to(self.lora_diag.device)
        lora_diag = self.lora_diag * pref_vecs.unsqueeze(1)
        lora_weight_A_diag = (self.lora_A * lora_diag)
        
        delta_weights = self.lora_B @ lora_weight_A_diag * self.scaling

        return delta_weights.data
    
    def set_pref_vec(self, pref_vec: torch.Tensor):
        
        is_merged = self.merged
        if is_merged:
            self.unmerge()

        pref_vec = pref_vec.squeeze()
        if len(pref_vec.shape) != 1:
            raise ValueError(f'Expected pref_vec to be 1 dimension, but got {pref_vec.shape}.')
        if pref_vec.shape[0] != self.pref_dim:
            raise ValueError(f'Expected pref_vec_len = {self.pref_dim}, but got {pref_vec.shape[0]}.')
        self.pref_vec = pref_vec.clone().to(self.task_flag.device)

        if is_merged:
            self.merge()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        
        if self.disabled:
            if self.merged:
                self.unmerge()
            res = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            res = self.base_layer(x, *args, **kwargs)
        else:
            res = self.base_layer(x, *args, **kwargs)
            pref_vecs = torch.cat(
                [torch.Tensor([1]).to(self.pref_vec.device), self.pref_vec],
                dim = 0
            ).index_select(0, self.task_flag + 1).to(self.lora_diag.device)
            lora_daig = self.lora_diag * pref_vecs.unsqueeze(1)
            res = res + (self.dropout(x) @ (self.lora_A * lora_daig).T @ self.lora_B.T) * self.scaling

        return res

    def to_task_specific(self, idx: int, task_idx: int):

        assert idx < self.r + self.pref_r * self.pref_dim and idx >= 0
        assert task_idx < self.pref_dim and task_idx >= 0
        if self.task_flag[idx] == task_idx:
            raise ValueError(f'Index {idx} has already set to task {task_idx} specific.')
        
        self.task_flag[idx] = task_idx
        # self.lora_diag.data[idx] *= 2

    def to_shared(self, idx: int):

        assert idx < self.r + self.pref_r * self.pref_dim and idx >= 0
        if self.task_flag[idx] == -1:
            raise ValueError(f'Index {idx} has already set to shared.')

        self.task_flag[idx] = -1
        # self.lora_diag.data[idx] *= 0.5

if __name__ == '__main__':
    
    linear_layer = nn.Linear(in_features = 1024, out_features = 256, bias = False)
    svd_lora_layer = SVD_Lora_Linear_Altered(
        config = SVD_Lora_Config(
            r = 6,
            pref_r = 2,
            pref_dim = 2,
            lora_dropout = 0.1
        ),
        base_layer = linear_layer
    )
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, svd_lora_layer.parameters()))
    svd_lora_layer.eval()
    svd_lora_layer.train()
    svd_lora_layer.set_pref_vec(torch.FloatTensor([0.1, 0.9]))

    for i in range(100):
        out = svd_lora_layer.forward(torch.rand(4, 20, 1024))
        target = torch.ones(4, 20, 256).float()
        (target - out).sum().backward()
        # print(svd_lora_layer.lora_A.grad)
        # print(svd_lora_layer.lora_B.grad)
        # print(svd_lora_layer.lora_diag.grad)
        print(svd_lora_layer.get_grad_mask(pref_idx=0, device='cpu'))
        print(svd_lora_layer.get_grad_mask(pref_idx=1, device='cpu'))
        optimizer.step()
        optimizer.zero_grad()
        if i + 1 == 20:
            svd_lora_layer.to_task_specific(idx = 0, task_idx = 1)
            
        if i + 1 == 25:
            svd_lora_layer.to_task_specific(idx = 3, task_idx = 0)
            
        if i + 1 == 30:
            svd_lora_layer.to_shared(idx = 0)
            
        if i + 1 == 40:
            svd_lora_layer.to_task_specific(idx = 4, task_idx = 1)
        
        if i + 1 == 50:
            svd_lora_layer.to_shared(idx = 3)

            

    