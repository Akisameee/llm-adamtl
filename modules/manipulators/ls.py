from accelerate import Accelerator
from torch.nn.modules import Module
from modules.manipulators.base import *

class Linear_Scalarization(Base_Manipulator):

    def __init__(self, model: Module, accelerator: Accelerator, pref_dim: int) -> None:
        super().__init__(model, accelerator, pref_dim)

    def get_weighted_loss(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        loss = torch.sum(losses * self.pref_vec)
        return loss
        

class ScaleInvariant_Linear_Scalarization(Base_Manipulator):

    def __init__(self, model: Module, accelerator: Accelerator, pref_dim: int) -> None:
        super().__init__(model, accelerator, pref_dim)

    def get_weighted_loss(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        loss = torch.sum(torch.log(losses) * self.pref_vec)
        return loss