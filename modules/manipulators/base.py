import torch
import torch.nn as nn
from accelerate import Accelerator
from typing import Dict, List, Tuple, Union

class Base_Manipulator():

    def __init__(
        self,
        model: nn.Module,
        accelerator: Accelerator,
        pref_dim: int
    ) -> None:
        
        self.model = model
        self.accelerator = accelerator

        self.pref_dim = pref_dim
        self.pref_vec = torch.FloatTensor(self.pref_dim).to(self.device)

    @property
    def device(self):
        return self.accelerator.device

    def set_pref_vec(
        self,
        pref_vec
    ):
        assert len(self.pref_vec) == len(pref_vec)
        
        for i in range(self.pref_dim):
            self.pref_vec[i] = pref_vec[i]
    
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        
        return torch.sum(losses)
    
    def backward(
        self,
        losses: torch.Tensor,
        params: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None
    ):
        
        weigthed_loss = self.get_weighted_loss(losses)
        self.accelerator.backward(weigthed_loss)
        return weigthed_loss