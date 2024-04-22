from modules.manipulators.base import *

class Linear_Scalarization(Base_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        pref_dim: int,
        max_norm: float = None
    ) -> None:
        super().__init__(model, accelerator, optimizer, pref_dim, max_norm)

    def get_weighted_loss(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        loss = torch.sum(losses * self.pref_vec)
        return loss
        

class ScaleInvariant_Linear_Scalarization(Base_Manipulator):

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        pref_dim: int,
        max_norm: float = None
    ) -> None:
        super().__init__(model, accelerator, optimizer, pref_dim, max_norm)

    def get_weighted_loss(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        loss = torch.sum(torch.log(losses) * self.pref_vec)
        return loss