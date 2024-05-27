import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import numpy as np
print(torch.cuda.device_count())
print(torch.version.cuda)

# points = np.linspace(0, 40242, 3, endpoint=True)
# points = np.round(points, 0).astype(int)
# print(points)
# n_epoch = 11
# x = torch.range(0, 1, step = 1 / (n_epoch - 1))
# print(x)
# x = torch.linspace(0, 1, steps = n_epoch)
# print(x)

# x = torch.tensor([1,6,3,2,5,7,0,4])

# y = x.argsort()
# print(y)
# print(x.index_select(0, y))

from trl import AutoModelForCausalLMWithValueHead
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

def get_model(
    model_path: str,
    device_map = 'auto'
):
    full_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model = full_model.pretrained_model
            
    dtype = model.dtype
    no_split_module_classes = model._get_no_split_modules(device_map)
    if device_map != "sequential":
        max_memory = get_balanced_memory(
            model,
            dtype=dtype,
            low_zero=(device_map == "balanced_low_0"),
            no_split_module_classes = no_split_module_classes,
        )

    # max_memory[0] *= 0.9
    model.tie_weights()
    device_map = infer_auto_device_map(
        model = model,
        max_memory = max_memory,
        dtype = dtype,
        no_split_module_classes = no_split_module_classes,
        # verbose = True
    )
    model = dispatch_model(
        model = model,
        device_map = device_map
    )
    print(model.hf_device_map)

    full_model.pretrained_model = model
    full_model.v_head.to(model.device)

    return model


model_path = '/home/share/models/huggingface/meta-llama/Llama-2-7b-chat-hf'
model = get_model(model_path)
print(model)