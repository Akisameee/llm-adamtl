import torch
import torch.nn as nn
from accelerate import Accelerator
from typing import Dict, List, Tuple, Union

class Base_Manipulator():

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        pref_dim: int,
        max_norm: float = None
    ) -> None:
        
        self.model = model
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.n_gradient_accumulation_step = self.accelerator.gradient_accumulation_steps
        self.gradient_accumulation_step = 0
        self.grad_dict = {}

        self.pref_dim = pref_dim
        self.pref_vec = torch.FloatTensor(self.pref_dim).to(self.device)
        self.max_norm = max_norm

    @property
    def device(self):
        return self.accelerator.device

    def set_pref_vec(
        self,
        pref_vec
    ):
        assert len(self.pref_vec) == len(pref_vec)
        
        for i in range(self.pref_dim):
            self.pref_vec[i] = pref_vec[i]
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        
        return torch.sum(losses)
    
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

    def restore_gradient(
        self
    ):
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

    # override this
    def get_weighted_loss(
        self,
        losses: torch.Tensor
    ):
        return losses.backward()
    
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
        
        return weigthed_loss