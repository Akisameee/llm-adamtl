import torch.nn.functional as F
from modules.manipulators.base import *

class FAMO(Base_MTL_Manipulator):
    '''Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h'''

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
    
        self.min_losses = torch.zeros(n_task)
        self.w = torch.tensor(
            [0.0] * n_task,
            # device = self.device,
            requires_grad = True
        )
        self.w_opt = torch.optim.Adam(
            [self.w],
            lr = kwargs.pop('w_lr', 0.025),
            weight_decay = kwargs.pop('gamma', 1e-5)
        )
        self.max_norm = kwargs.pop('max_norm', 1.0)
    
    def set_min_losses(self, losses):

        self.min_losses = losses
    
    def restore_gradient(self):
        
        nan_mask = torch.isnan(self.curr_losses)

        if self.restore_step > 0:
            self.update(self.prev_losses[-1])
        self.restore_step += 1

        z = F.softmax(self.w, -1)
        D = self.curr_losses - self.min_losses + 1e-8
        c = ((z[~nan_mask] / D[~nan_mask])).sum().detach()
        weights = z / c

        # print(f'pid: {os.getpid()}\nw: {weights}\nlosses: {self.curr_losses}')
        for name, param in self.get_named_parameters():
            assert name in self.grad_dict.keys()
            grad_tensors = [gt * w for gt, w in zip(self.grad_dict[name], weights)]
            param.grad = torch.stack(grad_tensors, dim = 0).sum(dim = 0).to(param.device)


    def update(self, prev_losses: torch.FloatTensor):

        delta = (prev_losses - self.min_losses + 1e-8).log() - \
            (self.curr_losses - self.min_losses + 1e-8).log()
        delta[torch.isnan(delta)] = 0.0
        
        with torch.enable_grad():
            d = torch.autograd.grad(
                F.softmax(self.w, -1),
                self.w,
                grad_outputs = delta.detach()
            )[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
