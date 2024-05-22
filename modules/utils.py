import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from beartype import beartype
from beartype.typing import List
import math
from typing import Union, Literal, Optional
from einops import rearrange
import os
import numpy as np
from transformers import AutoModel, AutoConfig, DebertaV2ForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, dispatch_model, infer_auto_device_map, Accelerator
from accelerate.utils import get_balanced_memory

from configs import LM_Config, RM_Config
from modules.pefts import replace_peft_layers
from modules.base import BaseLM, BaseRM, BaseLMWithValueHeads

lora_module_classes = ["Lora_linear"]

def split_peft_kwargs(kwargs):

    supported_kwargs = {}
    unsupported_kwargs = {}

    for key, value in kwargs.items():
        if key in [
            'svd_lora_init_strategy',
            'svd_lora_split_percentage'
        ]:
            supported_kwargs[key] = value
        else:
            unsupported_kwargs[key] = value

    return supported_kwargs, unsupported_kwargs

def get_model(
    config: LM_Config | RM_Config,
    dispatch: bool = True,
    device_map = 'auto',
    **kwargs
):
    peft_kwargs, model_kwargs = split_peft_kwargs(kwargs)

    if config.model_class is None:
        if isinstance(config, LM_Config):
            model_class = BaseLM
        elif isinstance(config, RM_Config):
            model_class = BaseRM
        else:
            raise NotImplementedError
    else:
        model_class = config.model_class
    
    if model_class in [BaseLM, BaseRM, BaseLMWithValueHeads]:
        full_model = model_class(config, **model_kwargs)
        model = full_model.pretrained_model
    else:
        # full_model = model_class.from_pretrained(config.model_pretrain_path, **kwargs)
        full_model = model_class.from_pretrained(config.model_pretrain_path)
        if model_class in [AutoModelForCausalLMWithValueHead]:
            model = full_model.pretrained_model
        else:
            model = full_model
    
    peft_info = None
    if config.peft_cfg is not None:
        if config.peft_cfg.target_modules is not None:
            model, peft_info = replace_peft_layers(
                model = model,
                peft_config = config.peft_cfg,
                return_info = True,
                **peft_kwargs
            )
            
    if dispatch:
        dtype = model.dtype
        if isinstance(model, DebertaV2ForSequenceClassification):
            setattr(model, "_no_split_modules", ["DebertaV2Layer"])
        
        no_split_module_classes = model._get_no_split_modules(device_map) + lora_module_classes
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

    if model_class in [BaseLM, BaseRM, BaseLMWithValueHeads, AutoModelForCausalLMWithValueHead]:
        full_model.pretrained_model = model
        if model_class == AutoModelForCausalLMWithValueHead:
            full_model.v_head.to(model.device)
        else:
            full_model.post_init()
    else:
        full_model = model
    
    return full_model, peft_info

def eval_decorator(fn):

    def inner(self, *args, **kwargs):

        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    
    return inner

def top_p(logits, thres = 0.9):

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):

    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# def masked_mean(seq, mask = None, dim = 1, keepdim = False):

#     if mask is None:
#         if dim is None:
#             return seq.mean()
#         else:
#             return seq.mean(dim = dim)
#     else:
#         if seq.ndim == 3:
#             mask = rearrange(mask, 'b n -> b n 1')

#         masked_seq = seq.masked_fill(~mask, 0.)
#         numer = masked_seq.sum(dim = dim, keepdim = keepdim)
#         denom = mask.sum(dim = dim, keepdim = keepdim)

#         masked_mean = numer / denom.clamp(min = 1e-3)
#         masked_mean = masked_mean.masked_fill(denom == 0, 0.)
#         return masked_mean

def masked_mean(values: torch.Tensor, mask: torch.Tensor = None, dim: bool = None, keepdim: bool = False) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if mask is None:
        mask = torch.ones_like(values, device = values.device)
    if dim is not None:
        return torch.sum(values * mask, dim = dim, keepdim = keepdim) / torch.sum(mask, dim = dim, keepdim = keepdim)
    else:
        return torch.sum(values * mask) / torch.sum(mask)

def gumbel_noise(t):

    noise = torch.zeros_like(t).uniform_(0, 1)

    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):

    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def log(t, eps = 1e-20):

    return torch.log(t.clamp(min = eps))

def log_prob(prob, indices):

    assert prob.shape[:2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)

def shift(t, value = 0, shift = 1, dim = -1):

    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value = value)

def default(val, d):

    if val is not None:
        return val
    return d() if callable(d) else d

def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance

def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: Union[int, float],
    dim: int = -1
) -> torch.Tensor:

    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        pad_tensor = pad_value * torch.ones(*pad_size, dtype = tensor.dtype, device = tensor.device)

        return torch.cat([tensor, pad_tensor], dim = dim)
    
def merge_dict(
    unmerged_dicts: list[dict],
    reduce: Optional[Literal['mean', 'sum']] = None
):
    keys = unmerged_dicts[0].keys()
    merged_dict = {}
    for key in keys:
        merged_datas = [unmerged_dict[key] for unmerged_dict in unmerged_dicts]
        if reduce == 'mean':
            merged_data = sum(merged_datas) / len(merged_datas)
        elif reduce == 'sum':
            merged_data = sum(merged_datas)
        elif reduce is None:
            merged_data = merged_datas
        else:
            raise NotImplementedError
        merged_dict[key] = merged_data

    return merged_dict


@beartype
class ExperienceDataset(Dataset):
    def __init__(
        self,
        data: List[torch.Tensor],
        device = None
    ):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):

        return self.data[0].shape[0]

    def __getitem__(self, index):
        
        return tuple(map(lambda t: t[index].to(self.device), self.data))

class Reward_Collector():

    def __init__(
        self,
        accelerator: Accelerator,
        pref_dim: int = 1
    ) -> None:
        
        self.accelerator = accelerator
        self.pref_dim = pref_dim
        if self.pref_dim >= 1:
            self.rewards = [[] * self.pref_dim]
            self.unsynced_rewards = [[] * self.pref_dim]
        else:
            raise ValueError(f'Expected pref_dim > 0, but got pref_dim = {pref_dim}')
        
    def update(self, reward: float, pref_idx: int = 0):

        assert pref_idx < self.pref_dim and pref_idx >= 0
        self.unsynced_rewards[pref_idx].append(reward)

    def get_var_mean(self, pref_idx: int = 0):

        assert pref_idx < self.pref_dim and pref_idx >= 0
        rewards = self.rewards[pref_idx] + self.unsynced_rewards[pref_idx]
        rewards = np.array(rewards)
        if len(rewards) > 0:
            return np.var(rewards), np.mean(rewards)
        else:
            return 1, 0
        
    def sync(self):

        for pref_idx in range(self.pref_dim):
            gathered_rewards = self.accelerator.gather_for_metrics(self.unsynced_rewards[pref_idx])
            self.rewards[pref_idx] += gathered_rewards
            self.unsynced_rewards[pref_idx].clear()