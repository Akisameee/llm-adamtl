import sys
sys.path.insert(0, '/home/smliu/RLHF')
import torch
import torch.nn as nn
import math
import numpy as np
import random
# from configs.peft_configs import Peft_Config, Lora_Config
import peft

from configs.pefts import Panacea_SVD_Config
from modules.pefts.base import Base_Adapter
from modules.pefts.lora import Lora_Linear
from modules.pefts.utils import compute_consine_similarities

class Panacea_SVD_Linear(Base_Adapter):

    def __init__(
        self,
        config: Panacea_SVD_Config,
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
            self.pref_scaling = nn.Parameter(torch.FloatTensor(self.pref_r), requires_grad = True)
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
        self.pref_vec = None

        self.diags = []
        self.conflict_cos_sims = []
        self.grad_conflict_scores = []
        self.split_flags = []
        self.r_index = torch.arange(self.r + self.pref_r)

    def reset_lora_weight(self, init_strategy = None):
        
        # nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
        nn.init.normal_(self.lora_A, mean = 0.0, std = 0.02)
        nn.init.normal_(self.lora_B, mean = 0.0, std = 0.02)
        if init_strategy == 'b_zero':
            nn.init.zeros_(self.lora_B)
        
        nn.init.normal_(self.lora_diag[: self.r], mean = 0.0, std = 0.1)
        nn.init.normal_(self.pref_scaling, mean = 0.0, std = 0.1)
        if init_strategy == 'diag_zero':
            nn.init.zeros_(self.lora_diag[: self.r])
            nn.init.zeros_(self.pref_scaling)

    def get_split_flag(self):

        split_flag = torch.zeros(self.r + self.pref_r)
        split_flag[self.r: ] += 1
        return split_flag.index_select(0, self.r_index.argsort())

    def restore_gradient(self, grad_dict: dict):

        lora_A_key = list(filter(lambda k: k.endswith('lora_A'), grad_dict.keys()))[0]
        lora_B_key = list(filter(lambda k: k.endswith('lora_B'), grad_dict.keys()))[0]

        lora_A_grads = []
        lora_B_grads = []
        conflict_grads = []
        conflict_params = []
        for pref_idx, (lora_A_grad, lora_B_grad) in enumerate(zip(grad_dict[lora_A_key], grad_dict[lora_B_key])):
            lora_A_grad_mask, lora_B_grad_mask = self.get_grad_mask(pref_idx, device = lora_A_grad.device)
            lora_A_grads.append(lora_A_grad * lora_A_grad_mask)
            lora_B_grads.append(lora_B_grad * lora_B_grad_mask)
            lora_A_conflict_grads = torch.cat(
                [lora_A_grad[: self.r, :]] + \
                [lora_A_grad[self.r + (i * self.pref_dim) + pref_idx, :].unsqueeze(0) for i in range(self.pref_r)],
                dim = 0
            )
            lora_B_conflict_grads = torch.cat(
                [lora_B_grad[:, : self.r]] + \
                [lora_B_grad[:, self.r + (i * self.pref_dim) + pref_idx].unsqueeze(1) for i in range(self.pref_r)],
                dim = 1
            )
            conflict_grads.append(
                torch.cat(
                    [
                        lora_A_conflict_grads,
                        lora_B_conflict_grads.T
                    ], dim = 1
                )
            )

            if self.pref_r > 0:
                lora_A_conflict_params = torch.cat(
                    [self.lora_A.data[self.r + (i * self.pref_dim) + pref_idx, :].unsqueeze(0) for i in range(self.pref_r)],
                    dim = 0
                ).to(lora_A_grad.device)
                lora_B_conflict_params = torch.cat(
                    [self.lora_B.data[:, self.r + (i * self.pref_dim) + pref_idx].unsqueeze(1) for i in range(self.pref_r)],
                    dim = 1
                ).to(lora_A_grad.device)
                conflict_params.append(
                    torch.cat(
                        [
                            lora_A_conflict_params,
                            lora_B_conflict_params.T
                        ], dim = 1
                    )
                )

        grad_dict[lora_A_key] = torch.stack(lora_A_grads, dim = 0).sum(dim = 0)
        grad_dict[lora_B_key] = torch.stack(lora_B_grads, dim = 0).sum(dim = 0)
        
        # compute conflict gradient scores
        remap_index = self.r_index.argsort()

        conflict_cos_sim = compute_consine_similarities(conflict_grads, dim = 1)
        if self.pref_r > 0:
            conflict_params_cos_sim = compute_consine_similarities(conflict_params, dim = 1)
            conflict_cos_sim[self.r: ] = torch.min(
                conflict_cos_sim[self.r: ],
                conflict_params_cos_sim
            )
        conflict_cos_sim = conflict_cos_sim.index_select(0, remap_index)
        
        diag = torch.cat(
            [
                self.lora_diag.data[: self.r].squeeze(),
                self.pref_scaling.data
            ], dim = 0
        ).to(conflict_cos_sim.device)
        diag = diag.index_select(0, remap_index)
        grad_conflict_score = conflict_cos_sim * diag.abs()

        self.diags.append(diag)
        self.conflict_cos_sims.append(conflict_cos_sim)
        self.grad_conflict_scores.append(grad_conflict_score.to(lora_A_grad.device))
        # print(self.grad_conflict_scores[-1])
        # split_flag = torch.zeros(self.r + self.pref_r).to(lora_A_grad.device)
        # split_flag[self.r: ] += 1
        # self.split_flags.append(split_flag.index_select(0, remap_index))
        self.split_flags.append(self.get_split_flag())

        return grad_dict
    
    def get_grad_mask(self, pref_idx: int, device):
        
        device = self.device if device is None else device

        pref_grad_mask = torch.zeros(self.pref_dim, device = device)
        pref_grad_mask[pref_idx] = 1
        grad_mask = torch.cat([torch.ones(self.r, device = device), pref_grad_mask.repeat(self.pref_r)], dim = 0)

        lora_A_grad_mask = grad_mask.unsqueeze(1).expand(-1, self.in_features)
        lora_B_grad_mask = grad_mask.unsqueeze(0).expand(self.out_features, -1)

        # pref_scaling_grad_mask = 

        return lora_A_grad_mask, lora_B_grad_mask

    def get_delta_weights(self):

        # self.set_pref_vec(self.pref_vec)
        pref_scaling = self.pref_scaling.unsqueeze(1).repeat(1, self.pref_dim).view(-1, 1)
        lora_daig_scaled = torch.cat(
            [
                self.lora_diag[: self.r, :],
                self.lora_diag[self.r: , :] * pref_scaling
            ], dim = 0
        )
        lora_weight_A_diag = (self.lora_A * lora_daig_scaled)
        
        delta_weights = self.lora_B @ lora_weight_A_diag * self.scaling

        return delta_weights.data

    def merge(self, safe_mode: bool = False):

        if self.merged:
            return
        
        delta_weights = self.get_delta_weights()
        if safe_mode:
            orig_weights = self.base_layer.weight.data.clone()
            orig_weights += delta_weights
            self.base_layer.weight.data = orig_weights
        else:
            self.base_layer.weight.data += delta_weights
        self.merged = True

    def unmerge(self):

        if not self.merged:
            return
        
        delta_weights = self.get_delta_weights()
        self.base_layer.weight.data -= delta_weights
        self.merged = False

    def train(self, mode: bool = True):
        
        self.dropout.train(mode)
        if mode:
            self.unmerge()
        else:
            self.merge()

    def set_pref_vec(self, pref_vec: torch.Tensor):
        
        self.pref_vec = pref_vec
        if self.pref_r == 0:
            return
        
        is_merged = self.merged
        if is_merged:
            self.unmerge()

        pref_vec = pref_vec.squeeze()
        if len(pref_vec.shape) != 1:
            raise ValueError(f'Expected pref_vec to be 1 dimension, but got {pref_vec.shape}.')
        if pref_vec.shape[0] != self.pref_dim:
            raise ValueError(f'Expected pref_vec_len = {self.pref_dim}, but got {pref_vec.shape[0]}.')
        
        pref_vec = pref_vec.unsqueeze(1).repeat(self.pref_r, 1)
        self.lora_diag.data[self.r: , :] = pref_vec.to(self.lora_diag.data.device)

        if is_merged:
            self.merge()

    def forward(self, x: torch.Tensor, *args, **kwargs):
        
        # print(f'x.device{x.device}, self.lora_A.device{self.lora_A.device}')
        if self.disabled:
            if self.merged:
                self.unmerge()
            res = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            res = self.base_layer(x, *args, **kwargs)
        else:
            res = self.base_layer(x, *args, **kwargs)
            # self.set_pref_vec(self.pref_vec)
            pref_scaling = self.pref_scaling.unsqueeze(1).repeat(1, self.pref_dim).view(-1, 1)
            lora_daig_scaled = torch.cat(
                [
                    self.lora_diag[: self.r, :],
                    self.lora_diag[self.r: , :].detach() * pref_scaling
                ], dim = 0
            )
            res = res + (self.dropout(x) @ (self.lora_A * lora_daig_scaled).T @ self.lora_B.T) * self.scaling

        return res
    
    def clear_gradient_scores(self):

        self.grad_conflict_scores.clear()
    
    def split_tensors(
        self,
        r_idx: int,
        lora_A_tensor: torch.Tensor,
        lora_B_tensor: torch.Tensor,
        lora_diag_tensor: torch.Tensor,
        pref_scaling_tensor: torch.Tensor
    ):
        lora_A_tensor = torch.cat(
            [
                lora_A_tensor[:r_idx, :],
                lora_A_tensor[r_idx + 1:, :],
                lora_A_tensor[r_idx, :].unsqueeze(0).repeat(self.pref_dim, 1)
            ], dim = 0
        )
        lora_B_tensor = torch.cat(
            [
                lora_B_tensor[:, :r_idx],
                lora_B_tensor[:, r_idx + 1:],
                lora_B_tensor[:, r_idx].unsqueeze(1).repeat(1, self.pref_dim)
            ], dim = 1
        )
        pref_scaling_tensor = torch.cat(
            [
                pref_scaling_tensor,
                lora_diag_tensor[r_idx]
            ], dim = 0
        )
        lora_diag_tensor = torch.cat(
            [
                lora_diag_tensor[:r_idx],
                lora_diag_tensor[r_idx + 1:],
                lora_diag_tensor[-self.pref_dim:]
            ], dim = 0
        )
        return (
            lora_A_tensor,
            lora_B_tensor,
            lora_diag_tensor,
            pref_scaling_tensor
        )

    def split(self, idx: int, packed_tensors = []):

        assert idx < self.r + self.pref_r and idx >= 0
        if not torch.any(self.r_index[: self.r] == idx):
            raise ValueError(f'Index {idx} has already splitted.')

        r_idx = torch.nonzero(self.r_index == idx)[0].item()
        (
            lora_A_splitted,
            lora_B_splitted,
            lora_diag_splitted,
            pref_scaling_splitted
        ) = self.split_tensors(
            r_idx = r_idx,
            lora_A_tensor = self.lora_A.data,
            lora_B_tensor = self.lora_B.data,
            lora_diag_tensor = self.lora_diag.data,
            pref_scaling_tensor = self.pref_scaling.data
        )
        self.lora_A = nn.Parameter(lora_A_splitted, requires_grad = True)
        self.lora_B = nn.Parameter(lora_B_splitted, requires_grad = True)
        self.lora_diag = nn.Parameter(lora_diag_splitted, requires_grad = True)
        self.pref_scaling = nn.Parameter(pref_scaling_splitted, requires_grad = True)
        
        self.r_index[r_idx: self.r + self.pref_r - 1] = self.r_index[r_idx + 1: self.r + self.pref_r].clone()
        self.r_index[self.r + self.pref_r - 1] = idx
        self.r -= 1
        self.pref_r += 1

        splitted_tensors = []
        for tensors in packed_tensors:
            splitted_tensors.append(self.split_tensors(r_idx = idx, *tensors))

        return splitted_tensors
    
    def unsplit_tensors(
        self,
        r_idx: int,
        remain_idx: int,
        lora_A_tensor: torch.Tensor,
        lora_B_tensor: torch.Tensor,
        lora_diag_tensor: torch.Tensor,
        pref_scaling_tensor: torch.Tensor
    ):
        pref_r_idx = r_idx - self.r
        lora_A_tensor = torch.cat(
            [
                lora_A_tensor[self.r + self.pref_dim * pref_r_idx + remain_idx, :].unsqueeze(0),
                lora_A_tensor[:self.r + self.pref_dim * pref_r_idx, :],
                lora_A_tensor[self.r + self.pref_dim * (pref_r_idx + 1):, :],
            ], dim = 0
        )
        lora_B_tensor = torch.cat(
            [
                lora_B_tensor[:, self.r + self.pref_dim * pref_r_idx + remain_idx].unsqueeze(1),
                lora_B_tensor[:, :self.r + self.pref_dim * pref_r_idx],
                lora_B_tensor[:, self.r + self.pref_dim * (pref_r_idx + 1):],
            ], dim = 1
        )
        lora_diag_tensor = torch.cat(
            [
                pref_scaling_tensor[pref_r_idx].unsqueeze(0).unsqueeze(0),
                lora_diag_tensor[: -self.pref_dim]
            ], dim = 0
        )
        pref_scaling_tensor = torch.cat(
            [
                pref_scaling_tensor[: pref_r_idx],
                pref_scaling_tensor[pref_r_idx + 1:]
            ], dim = 0
        )
        return (
            lora_A_tensor,
            lora_B_tensor,
            lora_diag_tensor,
            pref_scaling_tensor
        )

    def unsplit(self, idx: int, remain_idx: int = None, packed_tensors = []):

        assert idx < self.r + self.pref_r and idx >= 0
        if remain_idx is None:
            remain_idx = 0
        assert remain_idx < self.pref_dim and remain_idx >= 0
        if not torch.any(self.r_index[self.r: ] == idx):
            raise ValueError(f'Index {idx} has already unsplitted.')

        r_idx = torch.nonzero(self.r_index == idx)[0].item()
        (
            lora_A_unsplitted,
            lora_B_unsplitted,
            lora_diag_unsplitted,
            pref_scaling_unsplitted
        ) = self.unsplit_tensors(
            r_idx = r_idx,
            remain_idx = remain_idx,
            lora_A_tensor = self.lora_A.data,
            lora_B_tensor = self.lora_B.data,
            lora_diag_tensor = self.lora_diag.data,
            pref_scaling_tensor = self.pref_scaling.data
        )
        self.lora_A = nn.Parameter(lora_A_unsplitted, requires_grad = True)
        self.lora_B = nn.Parameter(lora_B_unsplitted, requires_grad = True)
        self.lora_diag = nn.Parameter(lora_diag_unsplitted, requires_grad = True)
        self.pref_scaling = nn.Parameter(pref_scaling_unsplitted, requires_grad = True)

        self.r_index[1: r_idx + 1] = self.r_index[0: r_idx].clone()
        self.r_index[0] = idx
        self.r += 1
        self.pref_r -= 1

        unsplitted_tensors = []
        for tensors in packed_tensors:
            unsplitted_tensors.append(self.unsplit_tensors(r_idx = idx, remain_idx = remain_idx, *tensors))

        return unsplitted_tensors

if __name__ == '__main__':
    
    linear_layer = nn.Linear(in_features = 1024, out_features = 256, bias = False)
    svd_lora_layer = Panacea_SVD_Linear(
        config = Panacea_SVD_Config(
            r = 8,
            pref_r = 0,
            pref_dim = 2
        ),
        base_layer = linear_layer
    )
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, svd_lora_layer.parameters()))
    svd_lora_layer.train()
    svd_lora_layer.set_pref_vec(torch.FloatTensor([0.1, 0.9]))

    for i in range(100):
        out = svd_lora_layer.forward(torch.rand(4, 20, 1024))
        target = torch.ones(4, 20, 256).float()
        (target - out).sum().backward()
        # print(svd_lora_layer.lora_A.grad)
        # print(svd_lora_layer.lora_B.grad)
        # print(svd_lora_layer.lora_diag.grad)
        # print(svd_lora_layer.pref_scaling.grad)
        # print(svd_lora_layer.get_grad_mask(pref_idx=0, device='cpu'))
        # print(svd_lora_layer.get_grad_mask(pref_idx=1, device='cpu'))
        optimizer.step()
        optimizer.zero_grad()
        if i + 1 == 20:
            svd_lora_layer.split(idx = 0)
            
        if i + 1 == 25:
            svd_lora_layer.split(idx = 3)
            
        if i + 1 == 30:
            svd_lora_layer.unsplit(idx = 0)
            
        if i + 1 == 40:
            svd_lora_layer.split(idx = 4)
        
        if i + 1 == 50:
            svd_lora_layer.unsplit(idx = 3)

            

    