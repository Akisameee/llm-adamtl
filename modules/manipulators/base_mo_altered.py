import torch
import torch.nn as nn
from accelerate import Accelerator
import os
from torch.optim.optimizer import Optimizer as Optimizer
from accelerate.utils import broadcast
import torch.distributed as dist

from logger import Logger
from modules.pefts import SVD_Lora_Linear_Altered
from modules.manipulators.base import Base_Weight_Manipulator
from modules.manipulators.utils import get_random_split

# backward multi-objective losses separately
class Base_MO_Manipulator_Altered(Base_Weight_Manipulator):

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
        self.svd_lora_layers = {name: module for name, module in model.named_modules() if isinstance(module, SVD_Lora_Linear_Altered)}
        # print(self.svd_lora_layers.values())
        n_pref_r = sum(module.pref_r for module in self.svd_lora_layers.values())
        self.n_rs = [sum(sum(module.task_flag == -1).item() for module in self.svd_lora_layers.values())]
        for t_idx in range(self.pref_dim):
            self.n_rs.append(sum(sum(module.task_flag == t_idx).item() for module in self.svd_lora_layers.values()))
        if split_percentage is None:
            self.n_split = n_pref_r
        else:
            n_r = sum(module.r for module in self.svd_lora_layers.values())
            self.n_split = int(n_r * split_percentage)
        
        if self.svd_lora_type is None or self.svd_lora_type == 'adaptive':
            pass
        elif self.svd_lora_type == 'random':
            # TODO
            pass
            # n_svd_lora = len(self.svd_lora_layers.values())
            # max_r = min(module.r + module.pref_r for module in self.svd_lora_layers.values())
            # random_split = get_random_split(n_svd_lora, self.n_split, max_r)
            # random_split = broadcast(torch.LongTensor(random_split).to(self.device)).cpu().tolist()
            # print(random_split)
            
            # idx = 0
            # for module in self.svd_lora_layers.values():
            #     n_split_op = module.pref_r - random_split[idx]
            #     if module.pref_r > random_split[idx]:
            #         for i in range(abs(n_split_op)):
            #             module.to_shared(module.r + i)
            #     else:
            #         for i in range(abs(n_split_op)):
            #             module.to_task_specific(i)
            #     idx += 1
            # assert idx != len(random_split) - 1
        else:
            raise NotImplementedError

        self.n_adapt_step = kwargs.pop('n_adapt_step', 128)
        self.conflict_scores = {}
        for name, module in self.svd_lora_layers.items():
            module.name = name
            print(f'{name}, ({module.get_task_flag().tolist()})')
            for r_idx, t_flag in enumerate(module.get_task_flag().tolist()):
                self.conflict_scores[f'{name}_{r_idx}'] = {
                    'ts_flag': t_flag,
                    'sh_ts_score': torch.zeros(self.pref_dim),
                    'ts_ts_score': 0
                }
        # self.n_rs = len(self.conflict_scores)

    def accumulate_gradient(
        self,
        obj_idx: int
    ):
        # flag = 1
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
            # self.accelerator.wait_for_everyone()
            # if flag:
            #     print(f'{os.getpid()}: {name}\n{param.shape}\n{param}\n{param.grad}')
            #     flag -= 1
            param.grad.zero_()

    def restore_gradient(
        self
    ):
        self.restore_step += 1
        if self.restore_step % self.n_adapt_step == 0:
            for name, module in self.svd_lora_layers.items():
                grad_dict = {k: v for k, v in self.grad_dict.items() if k.startswith(name)}
                module.compute_scores(grad_dict)
            self.adapt_svd_lora()
            for name, module in self.svd_lora_layers.items():
                grad_dict = {k: v for k, v in self.grad_dict.items() if k.startswith(name)}
                grad_dict = module.restore_gradient(grad_dict)
                self.grad_dict.update(grad_dict)
        else:
            for name, module in self.svd_lora_layers.items():
                grad_dict = {k: v for k, v in self.grad_dict.items() if k.startswith(name)}
                module.compute_scores(grad_dict)
                grad_dict = module.restore_gradient(grad_dict)
                self.grad_dict.update(grad_dict)
        
        # flag = 4
        for name, param in self.get_named_parameters():
            if name in self.grad_dict.keys():
                if isinstance(self.grad_dict[name], list):
                    param.grad = torch.stack(self.grad_dict[name], dim = 0).sum(dim = 0).to(param.device)
                else:
                    param.grad = self.grad_dict[name].to(param.device)
                # self.accelerator.wait_for_everyone()
                # if flag:
                #     print(f'{os.getpid()}: {name}\n{param.shape}\n{param}\n{param.grad}')
                #     flag -= 1
    
    def adapt_svd_lora(
        self
    ):
        if self.svd_lora_type == 'adaptive':
            
            for name, module in self.svd_lora_layers.items():
                sh_ts_conflict_score = torch.stack(module.records['sh_ts_conflict_scores']).mean(dim = 0)
                for r_idx, ts_flag in enumerate(module.get_task_flag().tolist()):
                    m_scores = self.conflict_scores[f'{name}_{r_idx}']
                    assert ts_flag == m_scores['ts_flag']
                    m_scores['sh_ts_score'] = sh_ts_conflict_score[:, r_idx].squeeze()
                    
            sh_ts_ranks = [
                {
                    name: rank for rank, (name, s) in \
                    enumerate(sorted(
                        self.conflict_scores.items(),
                        key = lambda item: item[1]['sh_ts_score'][t_idx],
                        reverse = True
                    ))
                }
                for t_idx in range(self.pref_dim)
            ]
            sh_params = dict(filter(lambda item: item[1]['ts_flag'] == -1, self.conflict_scores.items()))
            ts_param_groups = [
                dict(filter(lambda item: item[1]['ts_flag'] == t_idx, self.conflict_scores.items()))
                for t_idx in range(self.pref_dim)
            ]

            sh_params_new = {}
            for name, m_scores in sh_params.items():
                costs = torch.zeros(self.pref_dim)
                for t_idx in range(self.pref_dim):
                    if sh_ts_ranks[t_idx][name] < self.n_rs[t_idx + 1]:
                        costs[t_idx] = self.pref_vec[t_idx] * m_scores['sh_ts_score'][t_idx]
                if torch.any(costs != 0):
                    ts_idx = torch.argmax(costs).item()
                    m_scores['ts_flag'] = ts_idx
                    ts_param_groups[ts_idx][name] = m_scores
                else:
                    sh_params_new[name] = m_scores
            sh_params = sh_params_new
            print(len(sh_params), [len(ts_params) for ts_params in ts_param_groups])

            for t_idx, ts_params in enumerate(ts_param_groups):
                n_to_sh = len(ts_params) - self.n_rs[t_idx + 1]
                if n_to_sh > 0:
                    to_sh_list = [
                        name for name, rank in \
                        sorted(
                            filter(lambda item: item[0] in ts_params.keys(), sh_ts_ranks[t_idx].items()),
                            key = lambda item: item[1]
                        )
                    ][-n_to_sh: ]
                    for name in to_sh_list:
                        m_scores = ts_params.pop(name)
                        m_scores['ts_flag'] = -1
                        sh_params[name] = m_scores
            print(len(sh_params), [len(ts_params) for ts_params in ts_param_groups])
            
            for params in [sh_params, *ts_param_groups]:
                self.conflict_scores.update(params)
            self.update_svd_lora_layers()
    
    def update_svd_lora_layers(self):

        update_param_infos = []
        for name, module in self.svd_lora_layers.items():
            task_flag = module.get_task_flag()
            for r_idx, t_flag in enumerate(task_flag):
                m_scores = self.conflict_scores[f'{name}_{r_idx}']
                target_t_flag = m_scores['ts_flag']
                if t_flag != target_t_flag:
                    if target_t_flag != -1:
                        module.to_task_specific(r_idx, task_idx = target_t_flag)
                    else:
                        module.to_shared(r_idx)
                    update_param_infos.append(f'{name}_{r_idx} {t_flag}->{target_t_flag}')

        self.logger.info(
            'svd lora layer update:\n' + \
            '\n'.join(update_param_infos)
        )
    
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
            if self.use_ddp and obj_idx != self.pref_dim - 1:
                # ddp reducer gradient sync
                self.model.reducer._rebuild_buckets()
                self.model.reducer.prepare_for_backward([])
            
            self.accumulate_gradient(obj_idx)
        
        if self.max_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        return torch.sum(weighted_losses), weighted_losses

