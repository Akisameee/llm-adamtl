import torch
import torch.nn as nn
from accelerate import Accelerator
from typing import Dict, List, Tuple, Union
from collections import OrderedDict
import os
from torch.optim.optimizer import Optimizer as Optimizer
from accelerate.utils import broadcast

from logger import Logger
from modules.pefts import Panacea_SVD_Linear
from modules.manipulators.utils import get_random_split

class Base_Manipulator():

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
        self.optimizer = optimizer
        self.logger = logger
        self.n_gradient_accumulation_step = self.accelerator.gradient_accumulation_steps
        self.gradient_accumulation_step = 0
        self.grad_dict = {}
        self.max_norm = kwargs.pop('max_norm', None)

        self.restore_step = 0

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

# backward multi-objective losses separately
class Base_MO_Manipulator(Base_Weight_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        logger: Logger,
        **kwargs
    ) -> None:
        super().__init__(model, accelerator, optimizer, logger, **kwargs)

        self.svd_lora_type = kwargs.pop('svd_lora_type', None)
        split_percentage = kwargs.pop('svd_lora_split_percentage', None)
        self.svd_lora_layers = {name: module for name, module in model.named_modules() if isinstance(module, Panacea_SVD_Linear)}
        print(self.svd_lora_layers.values())
        n_pref_r = sum(module.pref_r for module in self.svd_lora_layers.values())
        if split_percentage is None:
            self.n_split = n_pref_r
        else:
            n_r = sum(module.r for module in self.svd_lora_layers.values())
            self.n_split = int(n_r * split_percentage)
        
        if self.svd_lora_type is None or self.svd_lora_type == 'adaptive':
            # for module in model.modules():
            #     if isinstance(module, Panacea_SVD_Linear):
            #         module.split(0)
            #         print(module.r_index)
            #         break
            pass
        elif self.svd_lora_type == 'random':
            n_svd_lora = sum(1 for module in self.svd_lora_layers.values())
            max_r = min(module.r + module.pref_r for module in self.svd_lora_layers.values())
            random_split = get_random_split(n_svd_lora, self.n_split, max_r)
            random_split = broadcast(torch.LongTensor(random_split).to(self.device)).cpu().tolist()
            print(random_split)
            
            idx = 0
            for module in self.svd_lora_layers.values():
                n_split_op = module.pref_r - random_split[idx]
                if module.pref_r > random_split[idx]:
                    for i in range(abs(n_split_op)):
                        module.unsplit(module.r + i)
                else:
                    for i in range(abs(n_split_op)):
                        module.split(i)
                # print(random_split[idx], module.r_index)
                idx += 1
            assert idx != len(random_split) - 1
        else:
            raise NotImplementedError

        # self.model = self.accelerator.prepare(self.model)
        self.n_adapt_step = kwargs.pop('n_adapt_step', 4)
        self.grad_conflict_scores = {}
        for name, module in self.svd_lora_layers.items():
            print(f'{name}, ({module.r}, {module.pref_r})')
            for r_idx in range(module.r + module.pref_r):
                self.grad_conflict_scores[f'{name}_{r_idx}'] = 0

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
        self.restore_step += 1
        for name, module in self.svd_lora_layers.items():
            grad_dict = {k: v for k, v in self.grad_dict.items() if k.startswith(name)}
            grad_dict = module.restore_gradient(grad_dict)
            self.grad_dict.update(grad_dict)
            
        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                if isinstance(self.grad_dict[name], list):
                    param.grad = torch.stack(self.grad_dict[name], dim = 0).sum(dim = 0).to(param.device)
                else:
                    param.grad = self.grad_dict[name].to(param.device)
    
    def adapt_svd_lora(
        self
    ):
        if self.svd_lora_type == 'adaptive':
            for name, module in self.svd_lora_layers.items():
                scores = torch.stack(module.grad_conflict_scores).mean(dim = 0)
                self.grad_conflict_scores.update(
                    {f'{name}_{r_idx}': scores[r_idx].item() for r_idx in range(module.r + module.pref_r)}
                )
            sorted_layers = sorted(
                self.grad_conflict_scores.items(),
                key = lambda layer_item: layer_item[1]
            )
            sorted_layers = {
                name: True if idx < self.n_split else False \
                for idx, (name, score) in enumerate(sorted_layers)
            }
            split_names = []
            unsplit_names = []
            for name, module in self.svd_lora_layers.items():
                split_flag = module.get_split_flag()
                for r_idx, splitted in enumerate(split_flag):
                    if not splitted and sorted_layers[f'{name}_{r_idx}']:
                        module.split(r_idx)
                        split_names.append(f'{name}_{r_idx}')
                    elif splitted and not sorted_layers[f'{name}_{r_idx}']:
                        module.unsplit(r_idx)
                        unsplit_names.append(f'{name}_{r_idx}')
            
            split_info = 'Module splitted:\n' + '\n'.join(split_names) \
                if len(split_names) > 0 else 'No module splitted.'
            unsplit_info = 'Module unsplitted:\n' + '\n'.join(unsplit_names) \
                if len(unsplit_names) > 0 else 'No module unsplitted.'
            self.logger.info(split_info + '\n' + unsplit_info)
    
    def step(
        self
    ):
        self.gradient_accumulation_step += 1
        if self.gradient_accumulation_step == self.n_gradient_accumulation_step:
            self.restore_gradient()
            if self.restore_step % self.n_adapt_step == 0:
                self.adapt_svd_lora()
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