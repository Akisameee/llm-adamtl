import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from configs.peft import Peft_Config, Lora_Config

from modules.peft.base import Base_Adapter
from modules.peft import Lora_Linear, Panacea_SVD_Linear

adapter_maps = {
    'lora': {
        nn.Linear: Lora_Linear
    },
    'panacea_svd': {
        nn.Linear: Panacea_SVD_Linear,
        Lora_Linear: Panacea_SVD_Linear
    }
}

def replace_peft_layers(
    model: nn.Module,
    peft_config: Peft_Config,
    return_info: bool = False
):
    for name, module in model._modules.items():
        # print(name, module)
        if len(list(module.children())) > 0:
            model._modules[name] = replace_peft_layers(module, peft_config)
        else:
            if name in peft_config.target_modules:
                adapter_map = adapter_maps[peft_config.adapter_name]
                if type(module) in adapter_map.keys():
                    peft_module = adapter_map[type(module)](
                        base_layer = module,
                        r = peft_config.r,
                        lora_alpha = peft_config.lora_alpha,
                        lora_dropout = peft_config.lora_dropout
                    )
                else:
                    raise NotImplementedError
                model._modules[name] = peft_module
            else:
                model._modules[name].requires_grad_(False)
    
    if return_info:
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_non_trainable_params = total_params - total_trainable_params
        peft_info = f'{type(model)} Peft Info:\n' + \
        f'total parameters: {total_params}\n' + \
        f'trainable parameters: {total_trainable_params}\n' + \
        f'non-trainable parameters: {total_non_trainable_params}'
        return model, peft_info
    else:
        return model

def freeze_except_adapters():
    pass

def set_all_adapters():
    pass