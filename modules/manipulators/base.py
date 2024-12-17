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
        self.curr_losses = torch.zeros(1)
        self.prev_losses = []

    @property
    def device(self):
        return self.accelerator.device
    
    def clear(
        self
    ):
        self.gradient_accumulation_step = 0
        self.grad_dict.clear()
        self.prev_losses.append(self.curr_losses.clone())
        self.curr_losses.zero_()
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
                # param.grad = self.grad_dict[name].to(param.device)
                if isinstance(self.grad_dict[name], list):
                    param.grad = torch.stack(self.grad_dict[name], dim = 0).sum(dim = 0).to(param.device)
                else:
                    param.grad = self.grad_dict[name].to(param.device)
    
    def gather_losses(
        self
    ):
        device = self.curr_losses.device
        losses = self.curr_losses.to(self.device)

        nan_mask = torch.isnan(losses)
        losses[nan_mask] = 0  
        n_non_nan = (~nan_mask).float()

        losses_gather = self.accelerator.reduce(losses, reduction = 'sum')
        n_non_nan = self.accelerator.reduce(n_non_nan, reduction = 'sum')

        losses_gather = torch.where(n_non_nan > 0, losses_gather / n_non_nan, float('nan'))
        self.curr_losses = losses_gather.to(device)
    
    def step(
        self
    ):
        self.gradient_accumulation_step += 1
        if self.gradient_accumulation_step == self.n_gradient_accumulation_step:
            self.gather_losses()
            if self.n_gradient_accumulation_step > 1:
                self.restore_gradient()
            self.optimizer.step()
            self.clear()
    
    def backward(
        self,
        loss: torch.Tensor
    ):
        # weighted_loss = self.get_weighted_loss(losses)

        self.curr_losses += loss.item()
        self.accelerator.backward(loss)
        if self.max_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        if self.n_gradient_accumulation_step > 1:
            self.accumulate_gradient()
        
        return loss

# scalarize loss before backward(joint training)
class Base_MTL_Manipulator(Base_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        logger: Logger,
        n_task: int,
        **kwargs
    ) -> None:
        super().__init__(model, accelerator, optimizer, logger, **kwargs)

        self.n_task = n_task
        self.curr_losses = torch.zeros(n_task)
        self.pref_vec = torch.FloatTensor(self.n_task).to(self.device)
        self.task_step = 0

    def set_pref_vec(
        self,
        pref_vec
    ):
        assert len(self.pref_vec) == len(pref_vec)
        
        for i in range(self.n_task):
            self.pref_vec[i] = pref_vec[i]

    def step(
        self
    ):
        self.gradient_accumulation_step += 1
        self.task_step = 0
        if self.gradient_accumulation_step == self.n_gradient_accumulation_step:
            self.gather_losses()
            self.restore_gradient()
            if self.max_norm is not None and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
            
            self.optimizer.step()
            self.clear()

    def accumulate_gradient(
        self,
        task_idx: int
    ):
        # flag = 4
        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                assert len(self.grad_dict[name]) >= task_idx
                if len(self.grad_dict[name]) == task_idx:
                    self.grad_dict[name].append(param.grad.detach().cpu())
                else:
                    self.grad_dict[name][task_idx] += param.grad.detach().cpu()
            else:
                assert task_idx == 0
                self.grad_dict[name] = [param.grad.detach().cpu()]
            # self.accelerator.wait_for_everyone()
            # if flag:
            #     print(f'{os.getpid()}: {name}\n{param.shape}\n{param}\n{param.grad}')
            #     flag -= 1
            param.grad.zero_()

    def backward(
        self,
        losses: torch.Tensor
    ):
        assert len(losses) == self.n_task
        weighted_losses = self.get_weighted_loss(losses)
        
        for task_idx, weighted_loss in enumerate(weighted_losses):
            self.accelerator.backward(
                weighted_loss,
                retain_graph = task_idx != self.n_task - 1
            )
            if self.use_ddp and task_idx != self.n_task - 1:
                # ddp reducer gradient sync
                self.model.reducer._rebuild_buckets()
                self.model.reducer.prepare_for_backward([])
            
            self.accumulate_gradient(task_idx)
        
        # if self.max_norm is not None and self.accelerator.sync_gradients:
        #     self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        return torch.sum(weighted_losses), weighted_losses

    def backward_single(
        self,
        loss: torch.Tensor,
        task_idx: int
    ):
        assert task_idx == self.task_step
        self.curr_losses[task_idx] += loss.item() / self.n_gradient_accumulation_step
        self.accelerator.backward(
            loss,
            retain_graph = self.task_step != self.n_task - 1
        )
        if self.use_ddp and self.task_step != self.n_task - 1:
            # ddp reducer gradient sync
            self.model.reducer._rebuild_buckets()
            self.model.reducer.prepare_for_backward([])
        
        self.accumulate_gradient(self.task_step)
        self.task_step += 1
        
        return loss
