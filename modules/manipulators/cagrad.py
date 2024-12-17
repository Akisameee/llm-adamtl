import numpy as np
from scipy.optimize import minimize
from modules.manipulators.base import *

class CAGrad(Base_MTL_Manipulator):

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
        self.c = kwargs.pop('c', 0.4)
        self.max_norm = kwargs.pop('max_norm', 1.0)

    def restore_gradient(self):
        
        self.restore_step += 1

        # NOTE: we allow only shared params for now. Need to see paper for other options.
        shared_parameters_name = []
        shared_parameters = []
        grad_dims = []
        for name, param in self.get_named_parameters():
            shared_parameters_name.append(name)
            shared_parameters.append(param)
            grad_dims.append(param.data.numel())
            
        grads_vec = torch.Tensor(sum(grad_dims), self.n_task)

        # for i in range(self.n_task):
        #     if i < self.n_task:
        #         losses[i].backward(retain_graph=True)
        #     else:
        #         losses[i].backward()
        #     self.grad2vec(shared_parameters, grads_vec, grad_dims, i)
        #     # multi_task_model.zero_grad_shared_modules()
        #     for p in shared_parameters:
        #         p.grad = None

        for task_idx in range(self.n_task):
            grads = [self.grad_dict[n][task_idx] for n in shared_parameters_name]
            self.grad2vec(grads, grads_vec, grad_dims, task_idx)

        g, GTG, w_cpu = self.cagrad(grads_vec, alpha = self.c, rescale = 1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def cagrad(self, grads, alpha = 0.5, rescale = 1):
        
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_task) / self.n_task
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_task).dot(A).dot(b.reshape(self.n_task, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_task).dot(A).dot(x.reshape(self.n_task, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(grads: List[torch.Tensor], grads_vec, grad_dims, task_idx):

        grads_vec[:, task_idx].fill_(0.0)
        cnt = 0

        for grad in grads:
            assert grad.data.numel() == grad_dims[cnt]
            grad_cur = grad.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            grads_vec[beg:en, task_idx].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters: List[torch.nn.Parameter], newgrad, grad_dims):

        newgrad = newgrad * self.n_task  # to match the sum loss
        cnt = 0

        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone().to(param.device)
            cnt += 1
