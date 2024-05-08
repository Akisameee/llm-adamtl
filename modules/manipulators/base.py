import torch
import torch.nn as nn
from accelerate import Accelerator
from typing import Dict, List, Tuple, Union
from collections import OrderedDict
import os
from torch.optim.optimizer import Optimizer as Optimizer
from accelerate.utils import broadcast
import torch.distributed as dist
import json

from logger import Logger
from modules.pefts import SVD_Lora_Linear
from modules.manipulators.weighted_loss import WeightedLoss_Mixin

class Base_Manipulator(WeightedLoss_Mixin):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        logger: Logger,
        **kwargs
    ) -> None:
        
        self.model = model
        self.accelerator = accelerator
        self.use_ddp = self.accelerator.num_processes > 1
        self.optimizer = optimizer
        self.logger = logger
        self.n_gradient_accumulation_step = self.accelerator.gradient_accumulation_steps
        self.gradient_accumulation_step = 0
        self.grad_dict = {}
        self.max_norm = kwargs.pop('max_norm', None)
        self.weighted_loss_type = kwargs.pop('weighted_loss_type', None)

        self.restore_step = 0

    @property
    def device(self):
        return self.accelerator.device
    
    def clear(
        self
    ):
        self.gradient_accumulation_step = 0
        self.grad_dict.clear()
        self.optimizer.zero_grad()

    def get_parameters(
        self
    ):
        return filter(lambda p: p.requires_grad, self.model.parameters())
    
    def get_named_parameters(
        self
    ):
        return filter(lambda p: p[1].requires_grad, self.model.named_parameters())

    def accumulate_gradient(
        self
    ):
        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                self.grad_dict[name] += param.grad.detach().cpu()
            else:
                self.grad_dict[name] = param.grad.detach().cpu()
            param.grad.zero_()

        # for k in self.grad_dict.keys():
        #     if k is not None:
        #         break
        # print(f'pid: {os.getpid()}, grad: {self.grad_dict[k]}')

    def restore_gradient(
        self
    ):
        self.restore_step += 1
        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                param.grad = self.grad_dict[name].to(param.device)
    
    def step(
        self
    ):
        self.gradient_accumulation_step += 1
        if self.gradient_accumulation_step == self.n_gradient_accumulation_step:
            if self.n_gradient_accumulation_step > 1:
                self.restore_gradient()
            self.optimizer.step()
            self.clear()
    
    def backward(
        self,
        losses: torch.Tensor
    ):
        weigthed_loss = self.get_weighted_loss(losses)

        self.accelerator.backward(weigthed_loss)
        if self.max_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        if self.n_gradient_accumulation_step > 1:
            self.accumulate_gradient()
        
        return weigthed_loss, losses

# scalarize loss before backward(joint training)
class Base_Weight_Manipulator(Base_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        logger: Logger,
        **kwargs
    ) -> None:
        super().__init__(model, accelerator, optimizer, logger, **kwargs)

        self.pref_dim = kwargs.pop('pref_dim')
        self.pref_vec = torch.FloatTensor(self.pref_dim).to(self.device)

    def set_pref_vec(
        self,
        pref_vec
    ):
        assert len(self.pref_vec) == len(pref_vec)
        
        for i in range(self.pref_dim):
            self.pref_vec[i] = pref_vec[i]

