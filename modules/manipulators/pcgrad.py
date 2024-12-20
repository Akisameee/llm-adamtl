import copy
import random
from modules.manipulators.base import *

class PCGrad(Base_MTL_Manipulator):
    '''Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    '''
    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        logger: Logger,
        n_task: int,
        **kwargs
    ):
        super().__init__(model, accelerator, optimizer, logger, n_task, **kwargs)
        reduction = kwargs.pop('reduction', 'sum')
        assert reduction in ['mean', 'sum']
        self.reduction = reduction

    def restore_gradient(self):
        
        self.restore_step += 1
        
        # shared part
        shared_parameters_name = []
        shared_parameters = []
        for name, param in self.get_named_parameters():
            shared_parameters_name.append(name)
            shared_parameters.append(param)
        shared_grads = []
        for t_idx in range(self.n_task):
            shared_grads.append(
                tuple([self.grad_dict[n][t_idx] for n in shared_parameters_name])
            )

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for param, g in zip(shared_parameters, non_conflict_shared_grads):
            param.grad = g.to(param.device)
        
        # TODO: task specific part
        # if task_specific_parameters is not None:
        #     task_specific_grads = torch.autograd.grad(
        #         losses.sum(), task_specific_parameters
        #     )
        #     if isinstance(task_specific_parameters, torch.Tensor):
        #         task_specific_parameters = [task_specific_parameters]
        #     for p, g in zip(task_specific_parameters, task_specific_grads):
        #         p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [
                        torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                        for grad_i, grad_j in zip(g_i, g_j)
                    ]
                )
                if g_i_g_j < 0:
                    g_j_norm_square = (
                        torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                    )
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == 'mean':
            merged_grad = [g / self.n_tasks for g in merged_grad]

        return merged_grad

    # def backward(
    #     self,
    #     losses: torch.Tensor,
    #     parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
    #     shared_parameters: Union[
    #         List[torch.nn.parameter.Parameter], torch.Tensor
    #     ] = None,
    #     task_specific_parameters: Union[
    #         List[torch.nn.parameter.Parameter], torch.Tensor
    #     ] = None,
    #     **kwargs,
    # ):
    #     self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
    #     # make sure the solution for shared params has norm <= self.eps
    #     if self.max_norm > 0:
    #         torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
    #     return None, {}  # NOTE: to align with all other weight methods