import sys
import os
sys.path.insert(0, '/home/smliu/RLHF')
import torch
import torch.nn as nn
import math
import numpy as np

from configs.pefts import Peft_Config, Lora_Config

from modules.pefts.base import Base_Adapter

class Lora_Linear(Base_Adapter):

    def __init__(
        self,
        config: Lora_Config,
        base_layer: nn.Module,
    ) -> None:
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f'Expected base_layer type \'torch.nn.Linear\', but got \'{type(base_layer)}\'.')
        super().__init__(base_layer)
        if config.r <= 0:
            raise ValueError(f'Expected r > 0, but got r = {config.r}.')

        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.lora_alpha = config.lora_alpha
        self.lora_A = nn.Linear(in_features = self.in_features, out_features = config.r, bias = False)
        self.lora_B = nn.Linear(in_features = config.r, out_features = self.out_features, bias = False)
        self.reset_lora_weight(init_weights = True)
        if config.lora_dropout > 0.0:
            self.dropout = nn.Dropout(p = config.lora_dropout)
        else:
            self.dropout = nn.Identity()
        self.scaling = config.lora_alpha / config.r
        self.merged = False
        self.set_adapter(enable = True)

    def reset_lora_weight(self, init_weights: bool = True):

        if not init_weights:
            return
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a = math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weights(self):

        device = self.lora_B.weight.device
        dtype = self.lora_B.weight.dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lora_weight_A = self.lora_A.weight
        lora_weight_B = self.lora_B.weight

        if cast_to_fp32:
            lora_weight_A = lora_weight_A.float()
            lora_weight_B = lora_weight_B.float()

        delta_weights = lora_weight_B @ lora_weight_A * self.scaling

        if cast_to_fp32:
            delta_weights = delta_weights.to(dtype = dtype)
            self.lora_A.weight.data = lora_weight_A.to(dtype)
            self.lora_B.weight.data = lora_weight_B.to(dtype)

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
        # print('merged')

    def unmerge(self):

        if not self.merged:
            return
        
        delta_weights = self.get_delta_weights()
        self.base_layer.weight.data -= delta_weights
        self.merged = False
        # print('unmerged')

    def train(self, mode: bool = True):

        self.lora_A.train(mode)
        self.lora_B.train(mode)
        self.dropout.train(mode)
        if mode:
            self.unmerge()
        else:
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
            res = res + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

        return res
    
if __name__ == '__main__':

    linear_layer = nn.Linear(in_features = 1024, out_features = 256, bias = False)
    config = Lora_Config(
        use_peft = True,
        r = 8,
        lora_alpha = 32
    )
    lora_layer = Lora_Linear(
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
