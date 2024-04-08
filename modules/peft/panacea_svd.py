import sys
sys.path.insert(0, '/home/smliu/RLHF')
import torch
import torch.nn as nn
import math
import numpy as np
# from configs.peft_configs import Peft_Config, Lora_Config

from modules.peft.base import Base_Adapter
from modules.peft.lora import Lora_Linear

class Panacea_SVD_Linear(Base_Adapter):

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        pref_dim: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ) -> None:
        if not isinstance(base_layer, (nn.Linear, Lora_Linear)):
            raise TypeError(f'Expected base_layer type \'torch.nn.Linear\' or \'Lora_Linear\', but got \'{type(base_layer)}\'.')
        if r <= 0:
            raise ValueError(f'Expected r > 0, but got r = {r}.')
        if pref_dim <= 1:
            raise ValueError(f'Expected pref_dim > 1, but got pref_dim = {pref_dim}.')
        self.r = r
        self.pref_dim = pref_dim

        if isinstance(base_layer, nn.Linear):
            super().__init__(base_layer)
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
            self.lora_alpha = lora_alpha
            self.lora_A = nn.Linear(in_features = self.in_features, out_features = r + pref_dim, bias = False)
            self.lora_B = nn.Linear(in_features = r + pref_dim, out_features = self.out_features, bias = False)
            self.lora_diag = nn.Parameter(torch.zeros(r + pref_dim, 1), requires_grad = True)
            # self.pref_vec = torch.zeros(pref_dim).float()
            self.pref_scaling = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
            self.reset_lora_weight(init_weights = True)
        else:
            raise NotImplementedError

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(p = lora_dropout)
        else:
            self.dropout = nn.Identity()
        self.scaling = lora_alpha / r
        self.merged = False
        self.set_adapter(enable = True)

    def reset_lora_weight(self, init_weights: bool = True):

        if not init_weights:
            return
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a = math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # nn.init.normal_(self.lora_B.weight, mean = 0.0, std = 0.02)
        nn.init.normal_(self.lora_diag[: self.r], mean = 0.0, std = 0.02)
        nn.init.normal_(self.pref_scaling, mean = 0.0, std = 0.5)

    def get_delta_weights(self):

        lora_weight_A = self.lora_A.weight
        lora_weight_B = self.lora_B.weight

        lora_weight_A_diag = torch.cat(
            [
                lora_weight_A[: self.r, :] * self.lora_diag[: self.r, :],
                lora_weight_A[self.r: , :] * self.lora_diag[self.r: , :] * self.pref_scaling
            ],
            dim = 0
        )
        delta_weights = lora_weight_B @ lora_weight_A_diag * self.scaling

        return delta_weights

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

        self.lora_A.train(mode)
        self.lora_B.train(mode)
        self.dropout.train(mode)
        # if mode:
        #     self.unmerge()
        # else:
        #     self.merge()

    def set_pref_vec(self, pref_vec):

        if isinstance(pref_vec, (torch.Tensor, np.ndarray)):
            pref_vec_len = pref_vec.shape[0]
        elif isinstance(pref_vec, list):
            pref_vec_len = len(pref_vec)
            if pref_vec_len != self.pref_dim:
                raise ValueError(f'Expected pref_vec_len = {self.pref_dim}, but got {pref_vec_len}.')
        else:
            raise TypeError(f'Expected pref_vec type (list, numpy.ndarray, torch.Tensor), but got {type(pref_vec)}')

        self.lora_diag.data[self.r: , :] = torch.FloatTensor(pref_vec.squeeze()).unsqueeze(1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        
        if self.disabled:
            if self.merged:
                self.unmerge()
            res = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            res = self.base_layer(x, *args, **kwargs)
        else:
            res = self.base_layer(x, *args, **kwargs)
            lora_A_out = self.lora_A(self.dropout(x)).T
            lora_A_diag_out = torch.cat(
                [
                    lora_A_out[: self.r, :] * self.lora_diag[: self.r, :],
                    lora_A_out[self.r: , :] * self.lora_diag[self.r: , :].detach() * self.pref_scaling
                ],
                dim = 0
            )
            res = res + self.lora_B(lora_A_diag_out.T) * self.scaling

        return res

if __name__ == '__main__':

    linear_layer = nn.Linear(in_features = 1024, out_features = 256, bias = False)
    svd_lora_layer = Panacea_SVD_Linear(
        base_layer = linear_layer,
        r = 8,
        pref_dim = 2,
        lora_alpha = 32,
        lora_dropout = 0.1
    )

    svd_lora_layer.set_pref_vec(torch.FloatTensor([0.1, 0.9]))
    svd_lora_layer.merge()
    svd_lora_layer.unmerge()
    out = svd_lora_layer.forward(torch.rand(4, 1024))
    target = torch.ones(4, 256).float()
    (target - out).sum().backward()
    print(svd_lora_layer.lora_A.weight.grad)
    print(svd_lora_layer.lora_B.weight.grad)
    print(svd_lora_layer.lora_diag.grad)
    print(svd_lora_layer.pref_scaling.grad)