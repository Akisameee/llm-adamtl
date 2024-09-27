import torch
import torch.nn as nn
from accelerate import Accelerator
import os
from torch.optim.optimizer import Optimizer as Optimizer
from accelerate.utils import broadcast
import torch.distributed as dist
from functools import reduce

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
        
        self.task_step = 0

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
                
                sh_ts_conflict_score = torch.stack(module.records['sh_ts_conflict_scores'])
                ts_ts_conflict_score = torch.stack(module.records['ts_ts_conflict_scores'])

                sh_ts_conflict_score = sh_ts_conflict_score.mean(dim = 0)
                ts_ts_conflict_score = ts_ts_conflict_score.mean(dim = 0)

                # weights = torch.linspace(0.5, 1, self.restore_step)
                # sh_ts_weights = weights.view(-1, 1, 1).expand_as(sh_ts_conflict_score)
                # sh_ts_conflict_score = torch.sum(sh_ts_conflict_score * sh_ts_weights, dim = 0) / sh_ts_weights.sum(dim = 0)
                # ts_ts_weights = weights.view(-1, 1, 1, 1).expand_as(ts_ts_conflict_score)
                # ts_ts_conflict_score = torch.sum(ts_ts_conflict_score * ts_ts_weights, dim = 0) / ts_ts_weights.sum(dim = 0)

                for r_idx, ts_flag in enumerate(module.get_task_flag().tolist()):
                    m_scores = self.conflict_scores[f'{name}_{r_idx}']
                    assert ts_flag == m_scores['ts_flag']
                    m_scores['sh_ts_score'] = sh_ts_conflict_score[:, r_idx].squeeze()
                    m_scores['ts_ts_score'] = ts_ts_conflict_score[:, :, r_idx].squeeze()
                    
            sh_params = dict(filter(lambda item: item[1]['ts_flag'] == -1, self.conflict_scores.items()))
            ts_param_groups = [
                dict(filter(lambda item: item[1]['ts_flag'] == t_idx, self.conflict_scores.items()))
                for t_idx in range(self.pref_dim)
            ]
            
            sh_params, ts_param_groups = self.convers_sh_ts(self.conflict_scores)
            print(len(sh_params), [len(ts_params) for ts_params in ts_param_groups])

            # ts_param_groups = self.convers_ts_ts(
            #     ts_param_groups = ts_param_groups
            # )

            for params in [sh_params, *ts_param_groups]:
                self.conflict_scores.update(params)
            self.update_svd_lora_layers()

    def convers_sh_ts(
        self,
        conflict_scores: dict
    ):
        sh_ts_ranks = [
            {
                name: rank for rank, (name, s) in \
                enumerate(sorted(
                    conflict_scores.items(),
                    key = lambda item: item[1]['sh_ts_score'][t_idx],
                    reverse = True
                ))
            }
            for t_idx in range(self.pref_dim)
        ]
        
        n_rank_top = [n_r for n_r in self.n_rs]
        ts_param_groups_new = [
            {name: conflict_scores[name] for name, rank in sh_ts_ranks[t_idx].items() if rank < n_rank_top[t_idx + 1]}
            for t_idx in range(self.pref_dim)
        ]
        conflict_params = reduce(
            lambda a, b: a & set(b.keys()),
            ts_param_groups_new,
            set(ts_param_groups_new[0].keys())
        )
        while len(conflict_params) > 0:
            for name in conflict_params:
                m_scores = conflict_scores[name]
                target_idxs = torch.LongTensor([name in ts_params for ts_params in ts_param_groups_new])
                sh_ts_score = m_scores['sh_ts_score'].clone()
                sh_ts_score[~target_idxs] = float('-inf')
                target_idx = torch.argmax(sh_ts_score).item()
                conflict_scores[name]['ts_flag'] = target_idx
                target_idxs[target_idx] = 0
                for t_idx, flag in enumerate(target_idxs):
                    if flag:
                        del sh_ts_ranks[t_idx][name]
                        n_rank_top[t_idx + 1] += 1
            ts_param_groups_new = [
                {name: conflict_scores[name] for name, rank in sh_ts_ranks[t_idx].items() if rank < n_rank_top[t_idx + 1]}
                for t_idx in range(self.pref_dim)
            ]
            conflict_params = reduce(
                lambda a, b: a & set(b.keys()),
                ts_param_groups_new,
                set(ts_param_groups_new[0].keys())
            )
        for t_idx, ts_params in enumerate(ts_param_groups_new):
            for name, m_score in ts_params.items():
                m_score['ts_flag'] = t_idx

        sh_params_new = {}
        for name, m_scores in filter(
            lambda item: all([item[0] not in ts_params.keys() for ts_params in ts_param_groups_new]),
            conflict_scores.items()
        ):
            m_scores = conflict_scores[name]
            m_scores['ts_flags'] = -1
            sh_params_new[name] = m_scores

        return sh_params_new, ts_param_groups_new
    
    def convers_ts_ts(
        self,
        conflict_scores: dict
    ):
        ts_param_groups_new = [{} for _ in range(self.pref_dim)]
        ts_ts_ranks = [{} for _ in range(self.pref_dim)]
        for t_idx_1 in range(self.pref_dim):
            for t_idx_2 in range(t_idx_1 + 1, self.pref_dim):
                ts_ts_rank = {
                    name: rank for rank, (name, s) in \
                    enumerate(sorted(
                        {**ts_param_groups_new[t_idx_1], **ts_param_groups_new[t_idx_2]},
                        key = lambda item: item[1]['ts_ts_score'][t_idx_1][t_idx_2],
                        reverse = True
                    ))
                }
                ts_ts_ranks[t_idx_1][t_idx_2] = ts_ts_rank
                tsrs = self.n_rs[t_idx_1 + 1] + self.n_rs[t_idx_2 + 1]
                ts_ts_ranks[t_idx_2][t_idx_1] = {name: tsrs - rank for name, rank in ts_ts_rank.items()}

        # ts_param_groups_new = [
        #     {name: conflict_scores[name] for name, rank in sh_ts_ranks[t_idx].items() if rank < n_rank_top[t_idx + 1]}
        #     for t_idx in range(self.pref_dim)
        # ]
        # conflict_params = reduce(
        #     lambda a, b: a & set(b.keys()),
        #     ts_param_groups_new,
        #     set(ts_param_groups_new[0].keys())
        # )

        return ts_param_groups_new
            
    
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
        self.task_step = 0
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
    
    def backward_single(
        self,
        loss: torch.Tensor
    ):
        
        self.accelerator.backward(
            loss,
            retain_graph = self.task_step != self.pref_dim - 1
        )
        if self.use_ddp and self.task_step != self.pref_dim - 1:
            # ddp reducer gradient sync
            self.model.reducer._rebuild_buckets()
            self.model.reducer.prepare_for_backward([])
        
        self.accumulate_gradient(self.task_step)

        self.task_step += 1
        
        if self.max_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        return loss

