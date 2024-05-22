import torch
import torch.nn as nn
from accelerate import Accelerator
import os
from torch.optim.optimizer import Optimizer as Optimizer
from accelerate.utils import broadcast
import torch.distributed as dist

from logger import Logger
from modules.pefts import SVD_Lora_Linear
from modules.manipulators.base import Base_Weight_Manipulator
from modules.manipulators.utils import get_random_split

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
        self.svd_lora_layers = {name: module for name, module in model.named_modules() if isinstance(module, SVD_Lora_Linear)}
        self.optimizer_param_maps = self.get_optimizer_param_maps(pack_params = True)
        # print(self.svd_lora_layers.values())
        n_pref_r = sum(module.pref_r for module in self.svd_lora_layers.values())
        if split_percentage is None:
            self.n_split = n_pref_r
        else:
            n_r = sum(module.r for module in self.svd_lora_layers.values())
            self.n_split = int(n_r * split_percentage)
        
        if self.svd_lora_type is None or self.svd_lora_type == 'adaptive':
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
                        module.to_shared(module.r + i)
                else:
                    for i in range(abs(n_split_op)):
                        module.to_task_specific(i)
                idx += 1
            assert idx != len(random_split) - 1
            self.update_params()
        else:
            raise NotImplementedError

        # self.model = self.accelerator.prepare(self.model)
        self.n_adapt_step = kwargs.pop('n_adapt_step', 128)
        self.grad_conflict_scores = {}
        for name, module in self.svd_lora_layers.items():
            print(f'{name}, ({module.r}, {module.pref_r})')
            for r_idx in range(module.r + module.pref_r):
                self.grad_conflict_scores[f'{name}_{r_idx}'] = 0

    def get_optimizer_param_maps(
        self,
        pack_params: bool = False
    ):
        optimizer_param_maps = {}
        idx_s = 0
        init_param_list = list(self.get_named_parameters())
        module_names = [n for n, m in self.model.named_modules()]
        assert len(init_param_list) == sum(len(param_group['params']) for param_group in self.optimizer.param_groups)
        for g_idx, param_group in enumerate(self.optimizer.param_groups):
            for idx, param in enumerate(param_group['params']):
                if param is init_param_list[idx_s + idx][1]:
                    name = init_param_list[idx_s + idx][0]
                else:
                    for name, model_param in self.get_named_parameters():
                        if param is model_param:
                            # param_map[name] = idx
                            break
                if pack_params:
                    for split_time in range(len(name.split('.'))):
                        names = name.rsplit('.', split_time + 1)
                        if names[0] in module_names:
                            m_name = names[0]
                            p_name = '.'.join(names[1:])
                            break
                    if m_name in module_names:
                        if m_name in optimizer_param_maps.keys():
                            assert g_idx == optimizer_param_maps[m_name][0]
                            optimizer_param_maps[m_name][1][p_name] = idx
                        else:
                            optimizer_param_maps[m_name] = (g_idx, {p_name: idx})
                    else: raise ValueError(f'Invalid Param {name}.')
                else: optimizer_param_maps[name] = (g_idx, idx)

            idx_s += len(param_group['params'])
        
        return optimizer_param_maps
    
    def update_params(
        self
    ):
        if self.use_ddp:
            # Build parameters for reducer.
            parameters, expect_sparse_gradient = self.model._build_params_for_reducer()
            # Verify model equivalence.
            dist._verify_params_across_processes(self.model.process_group, parameters)
            # Sync params and buffers. Ensures all DDP models start off at the same value.
            self.model._sync_params_and_buffers(authoritative_rank=0)
            # In debug mode, build a mapping of parameter index -> parameter.
            if dist.get_debug_level() != dist.DebugLevel.OFF:
                param_to_name_mapping = self.model._build_param_to_name_mapping(parameters)
            else:
                param_to_name_mapping = {}
            # Builds reducer.
            self.model._ddp_init_helper(parameters, expect_sparse_gradient, param_to_name_mapping)

        current_params = {n: m for n, m in self.get_named_parameters()}
        for m_name, param_map in self.optimizer_param_maps.items():
            for p_name, p_idx in param_map[1].items():
                param_group = self.optimizer.param_groups[param_map[0]]
                name = m_name + '.' + p_name
                if not current_params[name] is param_group['params'][p_idx]:
                    if len(self.optimizer.state) > 0:
                        param_state = self.optimizer.state.pop(param_group['params'][p_idx])
                        self.optimizer.state[current_params[name]] = param_state
                    param_group['params'][p_idx] = current_params[name]
        
        torch.cuda.empty_cache()

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        return losses

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
        for name, module in self.svd_lora_layers.items():
            grad_dict = {k: v for k, v in self.grad_dict.items() if k.startswith(name)}
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
    
    def get_svd_lora_optimizer_states(
        self
    ):
        optimizer_state_dict = self.optimizer.state_dict()
        optimizer_states = optimizer_state_dict['state']
        svd_lora_states = {}
        for name, module in self.svd_lora_layers.items():
            param_idxs = self.optimizer_param_maps[name][1]
            svd_lora_state = (
                optimizer_states[param_idxs['lora_A']],
                optimizer_states[param_idxs['lora_B']],
                optimizer_states[param_idxs['lora_diag']],
                optimizer_states[param_idxs['pref_scaling']]
            )
            svd_lora_states[name] = svd_lora_state
        return svd_lora_states
    
    def update_optimizer_states(
        self,
        optimizer_states_new: dict
    ):
        optimizer_state_dict = self.optimizer.state_dict()
        optimizer_state_dict['state'].update(optimizer_states_new)
 
        self.optimizer.load_state_dict(optimizer_state_dict)
    
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
            svd_lora_optimizer_states = self.get_svd_lora_optimizer_states()
            optimizer_states_new = {}
            for name, module in self.svd_lora_layers.items():
                split_flag = module.get_task_flag()
                optimizer_state = svd_lora_optimizer_states[name]
                
                splitted_flag = 0
                for r_idx, splitted in enumerate(split_flag):
                    if not splitted and sorted_layers[f'{name}_{r_idx}']:
                        optimizer_state = module.to_task_specific(r_idx, optimizer_state)
                        split_names.append(f'{name}_{r_idx}')
                        splitted_flag = 1
                    elif splitted and not sorted_layers[f'{name}_{r_idx}']:
                        optimizer_state = module.to_shared(r_idx, 0, optimizer_state)
                        unsplit_names.append(f'{name}_{r_idx}')
                        splitted_flag = 1
                if splitted_flag:
                    param_map = self.optimizer_param_maps[name]
                    for p_idx, state in zip(param_map[1].values(), optimizer_state):
                        optimizer_states_new[p_idx] = state
            self.update_optimizer_states(optimizer_states_new)
            self.update_params()

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
            if self.use_ddp and obj_idx != self.pref_dim - 1:
                # ddp reducer gradient sync
                self.model.reducer._rebuild_buckets()
                self.model.reducer.prepare_for_backward([])
            
            self.accumulate_gradient(obj_idx)
        
        if self.max_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.get_parameters(), self.max_norm)
        
        return torch.sum(weighted_losses), weighted_losses