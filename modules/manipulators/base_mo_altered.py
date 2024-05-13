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
from modules.manipulators.utils import get_random_splits

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
        random_init = kwargs.pop('svd_lora_random_init', False)
        self.svd_lora_layers = {name: module for name, module in model.named_modules() if isinstance(module, SVD_Lora_Linear_Altered)}
        # print(self.svd_lora_layers.values())
        self.n_rs = [sum(sum(module.task_flag == -1).item() for module in self.svd_lora_layers.values())]
        for t_idx in range(self.pref_dim):
            self.n_rs.append(sum(sum(module.task_flag == t_idx).item() for module in self.svd_lora_layers.values()))
        
        if split_percentage is not None:
            n_tsr = int(sum(self.n_rs) * split_percentage)
            n_rs_new = [sum(self.n_rs) - n_tsr * self.pref_dim] + [n_tsr] * self.pref_dim
            self.n_rs = n_rs_new
        
        self.logger.info(
            f'shared params: {self.n_rs[0]}\n' + \
            '\n'.join([f'task {t_idx} specific params: {self.n_rs[t_idx + 1]}' for t_idx in range(self.pref_dim)])
        )
        
        if random_init:

            n_svd_lora = len(self.svd_lora_layers.values())
            max_rs = [len(module.task_flag) for module in self.svd_lora_layers.values()]
            random_splits = get_random_splits(n_svd_lora, self.n_rs[1:], max_rs, self.pref_dim)

            random_splits = broadcast(torch.LongTensor(random_splits).transpose(0, 1).to(self.device)).cpu()
            print(random_splits)
            
            for module, n_tsr in zip(self.svd_lora_layers.values(), random_splits):
                n_to_tsrs = torch.LongTensor(
                    [sum(module.task_flag == t_idx).item() for t_idx in range(self.pref_dim)]
                ) - n_tsr
                for t_idx, n_to_tsr in enumerate(n_to_tsrs):
                    if n_to_tsr > 0:
                        for _ in range(n_to_tsr):
                            first_ts_idx = (module.task_flag == t_idx).nonzero()[0].item()
                            module.to_shared(first_ts_idx)
                    else:
                        for _ in range(-n_to_tsr):
                            first_sh_idx = (module.task_flag == -1).nonzero()[0].item()
                            module.to_task_specific(first_sh_idx, t_idx)
            
            assert all(
                sum(sum(module.task_flag == t_idx).item() for module in self.svd_lora_layers.values()) == \
                self.n_rs[t_idx + 1] for t_idx in range(self.pref_dim)
            )

        self.n_adapt_step = kwargs.pop('n_adapt_step', 128)
        self.conflict_scores = {}
        for name, module in self.svd_lora_layers.items():
            module.name = name
            self.accelerator.wait_for_everyone()
            print(f'{name}, ({module.get_task_flag().tolist()})')
            for r_idx, t_flag in enumerate(module.get_task_flag().tolist()):
                self.conflict_scores[f'{name}_{r_idx}'] = {
                    'ts_flag': t_flag,
                    'sh_ts_score': torch.zeros(self.pref_dim),
                    'ts_ts_score': torch.zeros(self.pref_dim, self.pref_dim)
                }

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

                weights = torch.linspace(0.5, 1, self.restore_step)
                sh_ts_conflict_score = torch.stack(module.records['sh_ts_conflict_scores'])
                ts_ts_conflict_score = torch.stack(module.records['ts_ts_conflict_scores'])

                # sh_ts_conflict_score = sh_ts_conflict_score.mean(dim = 0)
                sh_ts_weights = weights.view(-1, 1, 1).expand_as(sh_ts_conflict_score)
                sh_ts_conflict_score = torch.sum(sh_ts_conflict_score * sh_ts_weights, dim = 0) / sh_ts_weights.sum(dim = 0)

                # ts_ts_conflict_score = ts_ts_conflict_score.mean(dim = 0)
                ts_ts_weights = weights.view(-1, 1, 1, 1).expand_as(ts_ts_conflict_score)
                ts_ts_conflict_score = torch.sum(ts_ts_conflict_score * ts_ts_weights, dim = 0) / ts_ts_weights.sum(dim = 0)

                for r_idx, ts_flag in enumerate(module.get_task_flag().tolist()):
                    m_scores = self.conflict_scores[f'{name}_{r_idx}']
                    assert ts_flag == m_scores['ts_flag']
                    m_scores['sh_ts_score'] = sh_ts_conflict_score[:, r_idx].squeeze()
                    m_scores['ts_ts_score'] = ts_ts_conflict_score[:, :, r_idx].squeeze()
                    
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
            
            sh_params, ts_param_groups = self.convers_sh_ts(
                sh_ts_ranks = sh_ts_ranks,
                sh_params = sh_params,
                ts_param_groups = ts_param_groups
            )

            # ts_param_groups = self.convers_ts_ts(
            #     ts_param_groups = ts_param_groups
            # )

            for params in [sh_params, *ts_param_groups]:
                self.conflict_scores.update(params)
            self.update_svd_lora_layers()

    def convers_sh_ts(
        self,
        sh_ts_ranks: list[dict],
        sh_params: dict,
        ts_param_groups: list[dict]
    ):
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

        return sh_params, ts_param_groups
    
    def convers_ts_ts(
        self,
        ts_param_groups: list[dict]
    ):
        pass
        
    
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

