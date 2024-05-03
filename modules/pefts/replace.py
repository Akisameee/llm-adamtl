import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn

from configs.pefts import Peft_Config, Lora_Config, Panacea_SVD_Config
from modules.pefts import Lora_Linear, Panacea_SVD_Linear

adapter_maps = {
    'lora': {
        nn.Linear: Lora_Linear
    },
    'panacea': {
        nn.Linear: Panacea_SVD_Linear,
        Lora_Linear: Panacea_SVD_Linear
    }
}

def replace_peft_layers(
    model: nn.Module,
    peft_config: Peft_Config,
    return_info: bool = False,
    **kwargs
):
    model = _replace_peft_layers(model, peft_config)

    if return_info:
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_non_trainable_params = total_params - total_trainable_params
        peft_info = f'Peft Info:\n' + \
        f'total parameters: {total_params}\n' + \
        f'trainable parameters: {total_trainable_params}\n' + \
        f'non-trainable parameters: {total_non_trainable_params}'
        return model, peft_info
    else:
        return model
    

def _replace_peft_layers(
    model: nn.Module,
    peft_config: Peft_Config
):
    for name, module in model._modules.items():
        # print(name, module)
        if len(list(module.children())) > 0:
            model._modules[name] = _replace_peft_layers(module, peft_config)
        else:
            if name in peft_config.target_modules:
                adapter_map = adapter_maps[peft_config.adapter_name]
                if type(module) in adapter_map.keys():
                    peft_module = adapter_map[type(module)](
                        config = peft_config,
                        base_layer = module
                    )
                else:
                    raise NotImplementedError
                model._modules[name] = peft_module
            else:
                model._modules[name].requires_grad_(False)
    
    return model