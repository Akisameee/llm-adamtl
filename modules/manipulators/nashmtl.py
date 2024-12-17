import numpy as np
import cvxpy as cp
from modules.manipulators.base import *

class NashMTL(Base_MTL_Manipulator):

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

        self.n_optim_iter = kwargs.pop('n_optim_iter', 20)
        self.update_weights_every = kwargs.pop('update_weights_every', 1)
        self.max_norm = kwargs.pop('max_norm', 1.0)

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1, ))
        self.init_gtg = self.init_gtg = np.eye(self.n_task)
        self.timestep = 0.0
        self.prvs_alpha = np.ones(self.n_task, dtype = np.float32)

    def _stop_criteria(self, gtg, alpha_t):

        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):

        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.n_optim_iter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t
            
            try:
                # TODO: Invalid Solver
                self.prob.solve(solver = cp.ECOS, warm_start = True, max_iters = 100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):

        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):

        self.alpha_param = cp.Variable(shape = (self.n_task, ), nonneg = True)
        self.prvs_alpha_param = cp.Parameter(
            shape = (self.n_task, ), value = self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape = (self.n_task, self.n_task), value = self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1, ), value = np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_task):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def restore_gradient(self):
        
        self.restore_step += 1

        # extra_outputs = dict()
        if self.timestep == 0:
            self._init_optim_problem()

        if (self.timestep % self.update_weights_every) == 0:
            self.timestep += 1

            grads = {}
            for t_idx in range(self.n_task):
                grads[t_idx] = torch.cat([torch.flatten(gs[t_idx]) for name, gs in self.grad_dict.items()])

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1, ))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.timestep += 1
            alpha = self.prvs_alpha

        for name, param in self.get_named_parameters():
            assert name in self.grad_dict.keys()
            grad_tensors = [gt * w for gt, w in zip(self.grad_dict[name], alpha)]
            param.grad = torch.stack(grad_tensors, dim = 0).sum(dim = 0).to(param.device)
        
        # weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        # extra_outputs["weights"] = alpha
        # extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        # return weighted_loss, extra_outputs