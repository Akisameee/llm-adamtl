import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
import os
from torch.optim.optimizer import Optimizer as Optimizer
from accelerate.utils import broadcast
import torch.distributed as dist
from functools import reduce

from logger import Logger
from modules.pefts import SVD_Lora_Linear_Altered, Lora_Linear_Altered
from modules.manipulators.base import Base_MTL_Manipulator
from modules.manipulators.utils import get_random_splits

# backward multi-objective losses separately
class ADA_Manipulator(Base_MTL_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        logger: Logger,
        **kwargs
    ) -> None:
        super().__init__(model, accelerator, optimizer, logger, **kwargs)

        self.ratio = kwargs.pop('ratio', 0.1)
        self.n_top = int(self.n_task * self.ratio)
        self.lora_layers = {
            name: module 
            for name, module in model.named_modules()
            if isinstance(
                module,
                (SVD_Lora_Linear_Altered, Lora_Linear_Altered)
            )
        }

    def restore_gradient(
        self
    ):
        self.restore_step += 1

        for name, layer in self.lora_layers.items():
            grad_dict = {k: v for k, v in self.grad_dict.items() if k.startswith(name)}
            grads = layer.get_grads(grad_dict)
            sims = torch.zeros(layer.r, self.n_task, self.n_task)
            cos_sims = torch.zeros_like(sims)
            for i in range(self.n_task):
                for j in range(i, self.n_task):
                    sim = torch.einsum('ij,ij->i', grads[i], grads[j])
                    cos_sim = F.cosine_similarity(grads[i], grads[j], dim = -1)
                    sims[:, i, j] = sim
                    sims[:, j, i] = sim
                    cos_sims[:, i, j] = cos_sim
                    cos_sims[:, j, i] = cos_sim

            scores = sims.sum(dim = -1)
            layer.record_step(cf_scores = scores, cos_sims = cos_sims.sum(dim = -1))
            for t_idx in range(self.n_task):
                if torch.sum(scores[t_idx] < 0) <= self.n_top:
                    unuse_idxs = torch.where(scores[t_idx] < 0)[0].tolist()
                else:
                    _, unuse_idxs = torch.topk(-scores[t_idx], self.n_top)
                    unuse_idxs = unuse_idxs.tolist()
                
                # self.accelerator.wait_for_everyone()
                # print(f'pid: {os.getpid()}, unuse_idxs: {unuse_idxs}')
                for unuse_idx in unuse_idxs:
                    for p_name, g in grad_dict.items():
                        g[t_idx][unuse_idx].zero_()

            self.grad_dict.update(grad_dict)

        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                if isinstance(self.grad_dict[name], list):
                    param.grad = torch.stack(self.grad_dict[name], dim = 0).sum(dim = 0).to(param.device)
                else:
                    param.grad = self.grad_dict[name].to(param.device)