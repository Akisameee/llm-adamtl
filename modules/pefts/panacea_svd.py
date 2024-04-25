import sys
sys.path.insert(0, '/home/smliu/RLHF')
import torch
import torch.nn as nn
import math
import numpy as np
# from configs.peft_configs import Peft_Config, Lora_Config
import peft

from configs.pefts import Panacea_SVD_Config
from modules.pefts.base import Base_Adapter
from modules.pefts.lora import Lora_Linear

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
            self.reset_lora_weight(init_weights = True)
        else:
            raise NotImplementedError

        if config.lora_dropout > 0.0:
            self.dropout = nn.Dropout(p = config.lora_dropout)
        else:
            self.dropout = nn.Identity()
        self.scaling = config.lora_alpha / self.r
        self.merged = False
        self.set_adapter(enable = True)
        self.grad_conflict_scores = []

    def reset_lora_weight(self, init_weights: bool = True):

        if not init_weights:
            return
        
        # nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
        nn.init.normal_(self.lora_A, mean = 0.0, std = 0.02)
        nn.init.zeros_(self.lora_B)
        # nn.init.normal_(self.lora_B, mean = 0.0, std = 0.02)
        nn.init.normal_(self.lora_diag[: self.r], mean = 0.0, std = 0.02)
        nn.init.normal_(self.pref_scaling, mean = 0.0, std = 0.5)

    def restore_gradient(self, grad_dict: dict):

        lora_A_key = list(filter(lambda k: k.endswith('lora_A'), grad_dict.keys()))[0]
        lora_B_key = list(filter(lambda k: k.endswith('lora_B'), grad_dict.keys()))[0]

        lora_A_grads = []
        lora_B_grads = []
        for pref_idx, (lora_A_grad, lora_B_grad) in enumerate(zip(grad_dict[lora_A_key], grad_dict[lora_B_key])):
            lora_A_grad_mask, lora_B_grad_mask = self.get_grad_mask(pref_idx, device = lora_A_grad.device)
            lora_A_grads.append(lora_A_grad * lora_A_grad_mask)
            lora_B_grads.append(lora_B_grad * lora_B_grad_mask)

        grad_dict[lora_A_key] = torch.stack(lora_A_grads, dim = 0).sum(dim = 0)
        grad_dict[lora_B_key] = torch.stack(lora_B_grads, dim = 0).sum(dim = 0)

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

        pref_scaling = self.pref_scaling.unsqueeze(1).repeat(1, self.pref_dim).view(-1, 1)
        lora_daig_scaled = torch.cat(
            [
                self.lora_diag[: self.r, :],
                self.lora_diag[self.r: , :] * pref_scaling
            ],
            dim = 0
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
        
        pref_vec = pref_vec.squeeze()
        if len(pref_vec.shape) != 1:
            raise ValueError(f'Expected pref_vec to be 1 dimension, but got {pref_vec.shape}.')
        if pref_vec.shape[0] != self.pref_dim:
            raise ValueError(f'Expected pref_vec_len = {self.pref_dim}, but got {pref_vec.shape[0]}.')
        
        pref_vec = pref_vec.unsqueeze(1).repeat(self.pref_r, 1)
        self.lora_diag.data[self.r: , :] = pref_vec.to(self.lora_diag.data.device)

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
            pref_scaling = self.pref_scaling.unsqueeze(1).repeat(1, self.pref_dim).view(-1, 1)
            lora_daig_scaled = torch.cat(
                [
                    self.lora_diag[: self.r, :],
                    self.lora_diag[self.r: , :].detach() * pref_scaling
                ],
                dim = 0
            )
            res = res + (self.dropout(x) @ (self.lora_A * lora_daig_scaled).T @ self.lora_B.T) * self.scaling

        return res

if __name__ == '__main__':

    linear_layer = nn.Linear(in_features = 1024, out_features = 256, bias = False)
    svd_lora_layer = Panacea_SVD_Linear(
        config = Panacea_SVD_Config(
            r = 6,
            pref_r = 2,
            pref_dim = 2
        ),
        base_layer = linear_layer
    )

    # svd_lora_layer.set_pref_vec(torch.FloatTensor([0.1, 0.9, 0.2, 0.3])
    svd_lora_layer.set_pref_vec(torch.FloatTensor([0.1, 0.9]))
    print(svd_lora_layer.base_layer.weight.data)
    svd_lora_layer.eval()
    print(svd_lora_layer.base_layer.weight.data)
    svd_lora_layer.train()
    print(svd_lora_layer.base_layer.weight.data)
    out = svd_lora_layer.forward(torch.rand(4, 20, 1024))
    target = torch.ones(4, 20, 256).float()
    (target - out).sum().backward()
    print(svd_lora_layer.lora_A.grad)
    print(svd_lora_layer.lora_B.grad)
    print(svd_lora_layer.lora_diag.grad)
    print(svd_lora_layer.pref_scaling.grad)
    print(svd_lora_layer.get_grad_mask(pref_idx=0, device='cpu'))
    print(svd_lora_layer.get_grad_mask(pref_idx=1, device='cpu'))