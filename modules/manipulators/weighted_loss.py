# from modules.manipulators.base import *
import torch
import torch.nn as nn

class WeightedLoss_Mixin():

    def linear_scalarization(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        loss = torch.sum(losses * self.pref_vec)
        return loss
    
    def scaleinvariant_linear_scalarization(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        loss = torch.sum(torch.log(losses) * self.pref_vec)
        return loss
    
    def mo_linear_scalarization(self, losses: torch.Tensor):
        
        if self.pref_vec.device != losses.device:
            self.pref_vec.to(losses.device)
        
        weighted_losses = losses * self.pref_vec
        return weighted_losses
    
    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):

        if self.weighted_loss_type == None:
            return torch.sum(losses)
        elif self.weighted_loss_type == 'ls':
            return self.linear_scalarization(losses)
        elif self.weighted_loss_type == 'sils':
            return self.scaleinvariant_linear_scalarization(losses)
        elif self.weighted_loss_type == 'mols':
            return self.mo_linear_scalarization(losses)
        else:
            raise NotImplementedError

# class Weight_Linear_Scalarization(Base_Weight_Manipulator):

#     def __init__(
#         self,
#         model: nn.Module,
#         accelerator: Accelerator,
#         optimizer: torch.optim.Optimizer,
#         **kwargs
#     ) -> None:
#         super().__init__(model, accelerator, optimizer, **kwargs)

#     def get_weighted_loss(self, losses: torch.Tensor):
        
#         if self.pref_vec.device != losses.device:
#             self.pref_vec.to(losses.device)
        
#         loss = torch.sum(losses * self.pref_vec)
#         return loss
        

# class Weight_ScaleInvariant_Linear_Scalarization(Base_Weight_Manipulator):

#     def __init__(
#         self,
#         model: nn.Module,
#         accelerator: Accelerator,
#         optimizer: torch.optim.Optimizer,
#         **kwargs
#     ) -> None:
#         super().__init__(model, accelerator, optimizer, **kwargs)

#     def get_weighted_loss(self, losses: torch.Tensor):
        
#         if self.pref_vec.device != losses.device:
#             self.pref_vec.to(losses.device)
        
#         loss = torch.sum(torch.log(losses) * self.pref_vec)
#         return loss
    
# class MO_Linear_Scalarization(Base_MO_Manipulator):

#     def __init__(
#         self,
#         model: nn.Module,
#         accelerator: Accelerator,
#         optimizer: torch.optim.Optimizer,
#         **kwargs
#     ) -> None:
#         super().__init__(model, accelerator, optimizer, **kwargs)

#     def get_weighted_loss(self, losses: torch.Tensor):
        
#         if self.pref_vec.device != losses.device:
#             self.pref_vec.to(losses.device)
        
#         weighted_losses = losses * self.pref_vec
#         return weighted_losses