import torch
import torch.nn as nn
from accelerate import Accelerator
from typing import Dict, List, Tuple, Union
import os

from torch.optim.optimizer import Optimizer as Optimizer

class Base_Manipulator():

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> None:
        
        self.model = model
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.n_gradient_accumulation_step = self.accelerator.gradient_accumulation_steps
        self.gradient_accumulation_step = 0
        self.grad_dict = {}
        self.max_norm = kwargs.pop('max_norm', None)

    @property
    def device(self):
        return self.accelerator.device
    
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

        # for k in self.grad_dict.keys():
        #     if k is not None:
        #         break
        # print(f'pid: {os.getpid()}, grad: {self.grad_dict[k]}')

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

# scalarize loss before backward
class Base_Weight_Manipulator(Base_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        **kwargs
    ) -> None:
        super().__init__(model, accelerator, optimizer, **kwargs)

        self.pref_dim = kwargs.pop('pref_dim')
        self.pref_vec = torch.FloatTensor(self.pref_dim).to(self.device)

    def set_pref_vec(
        self,
        pref_vec
    ):
        assert len(self.pref_vec) == len(pref_vec)
        
        for i in range(self.pref_dim):
            self.pref_vec[i] = pref_vec[i]

# backward multi-objective losses separately
class Base_MO_Manipulator(Base_Weight_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        **kwargs
    ) -> None:
        super().__init__(model, accelerator, optimizer, **kwargs)

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        return losses

    def accumulate_gradient(
        self,
        obj_idx: int
    ):
        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                assert len(self.grad_dict[name]) >= obj_idx
                if len(self.grad_dict[name]) == obj_idx:
                    self.grad_dict[name].append(param.grad.detach().cpu())
                else:
                    self.grad_dict[name][obj_idx] += param.grad.detach().cpu()
            else:
                assert obj_idx == 0
                self.grad_dict[name] = [param.grad.detach().cpu()]
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
            self.restore_gradient()
            self.optimizer.step()
            self.clear()
    
    def backward(
        self,
        losses: torch.Tensor
    ):
        assert len(losses) == self.pref_dim
        weighted_losses = self.get_weighted_loss(losses)
        
        for obj_idx, weighted_loss in enumerate(weighted_losses):
            self.accelerator.backward(
                weighted_loss,
                retain_graph = obj_idx != self.pref_dim - 1
            )
            self.accumulate_gradient(obj_idx)
        
        if self.max_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        return torch.sum(weighted_losses), weighted_losses