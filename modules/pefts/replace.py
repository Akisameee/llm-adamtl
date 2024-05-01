import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn

from configs.pefts import Peft_Config, Lora_Config, Panacea_SVD_Config
from modules.pefts import Lora_Linear, Panacea_SVD_Linear
from modules.pefts.utils import get_random_split

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

    svd_lora_init_strategy = kwargs.pop('svd_lora_init_strategy', None)
    svd_lora_split_percentage = kwargs.pop('svd_lora_split_percentage', None)

    if isinstance(peft_config, Panacea_SVD_Config) and svd_lora_init_strategy is not None:
        if svd_lora_init_strategy == 'random':
            n_svd_lora = sum(1 for module in model.modules() if isinstance(module, Panacea_SVD_Linear))
            split_percentage = peft_config.pref_r / (peft_config.r + peft_config.pref_r) \
                if svd_lora_split_percentage is None else svd_lora_split_percentage
            n_split = int((peft_config.r + peft_config.pref_r) * n_svd_lora * split_percentage)
            random_split = get_random_split(peft_config, n_svd_lora, n_split)
            
            idx = 0
            for module in model.modules():
                if isinstance(module, Panacea_SVD_Linear):
                    while module.pref_r != random_split[idx]:
                        if module.pref_r > random_split[idx]:
                            module.unsplit(0)
                        else:
                            module.split(0)
            assert idx != len(random_split) - 1
        else:
            raise NotImplementedError

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