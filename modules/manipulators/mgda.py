from modules.manipulators.base import *
from modules.manipulators.min_max_solver import *
from modules.manipulators.utils import gradient_normalizers

class MGDA(Base_MTL_Manipulator):
    '''Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    '''
    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: Optimizer,
        logger: Logger,
        n_task: int,
        # params = 'shared',
        **kwargs
    ):
        super().__init__(model, accelerator, optimizer, logger, n_task, **kwargs)
        self.solver = MinNormSolver()
        # assert params in ['shared', 'last', 'rep']
        # self.params = params
        normalization = kwargs.pop('normalization', 'none')
        assert normalization in ['norm', 'loss', 'loss+', 'none']
        self.normalization = normalization

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def restore_gradient(self):

        self.restore_step += 1

        grads = {}
        for t_idx in range(self.n_task):
            grads[t_idx] = [torch.flatten(gs[t_idx]) for name, gs in self.grad_dict.items()]

        gn = gradient_normalizers(grads, None, self.normalization)

        for t in range(self.n_task):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        sol = sol * self.n_task  # make sure it sums to self.n_tasks

        for name, param in self.get_named_parameters():
            assert name in self.grad_dict.keys()
            grad_tensors = [gt * w for gt, w in zip(self.grad_dict[name], sol)]
            param.grad = torch.stack(grad_tensors, dim = 0).sum(dim = 0).to(param.device)
        
        # weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])
        # return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))