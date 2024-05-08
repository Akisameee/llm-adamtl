import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import math
import numpy as np
from configs.pefts import Peft_Config, Lora_Config

class Base_Adapter(nn.Module):

    def __init__(
        self,
        base_layer: nn.Module
    ) -> None:
        super().__init__()

        self.base_layer = base_layer
        self.disabled = False

    @property
    def device(self):

        return self.base_layer.device

    def set_adapter(self, enable: bool, base_enable: bool = False):

        if enable:
            self.base_layer.requires_grad_(False)
            self.disabled = False
        else:
            self.base_layer.requires_grad_(base_enable)
            self.base_layer.requires_grad_(base_enable)
            self.disabled = True

    def get_delta_weights():

        raise NotImplementedError

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
        
        # self.train(mode)
        super().train(mode = mode)
        if mode:
            self.unmerge()
        else:
            self.merge()

    def forward(self, x: torch.Tensor, *args, **kwargs):

        return self.base_layer(x, *args, **kwargs)
    
    def post_init(self):

        self.to(self.base_layer.device)