import torch
import torch.nn as nn
import os
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from peft import get_peft_model
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2Model, AutoConfig, AutoTokenizer, AutoModel, GPTNeoForCausalLM, DebertaV2ForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead
from configs import SFT_Train_Config, LM_Config, RM_Config, RLHF_Config, rm_infos
from modules.utils import eval_decorator, top_k, top_p, gumbel_sample, masked_mean
from modules.peft import replace_peft_layers

lora_module_classes = ["Lora_linear"]

def get_model(
    config: LM_Config | RM_Config,
    dispatch: bool = True,
    device_map = 'auto'
):
    if config.model_class is None:
        if isinstance(config, LM_Config):
            model_class = BaseLM
        elif isinstance(config, RM_Config):
            model_class = RewardLM
        else:
            raise NotImplementedError
    else:
        model_class = config.model_class
    
    if model_class in [BaseLM, RewardLM]:
        full_model = model_class(config)
        model = full_model.lm
    else:
        full_model = model_class.from_pretrained(config.model_pretrain_path)
        if model_class == AutoModelForCausalLMWithValueHead:
            model = full_model.pretrained_model
        else:
            model = full_model
    
    if dispatch:
        if isinstance(model, DebertaV2ForSequenceClassification):
            setattr(model, "_no_split_modules", ["DebertaV2Layer"])
            
        dtype = model.dtype
        peft_info = None
        if config.peft_cfg is not None:
            if config.peft_cfg.target_modules is not None:
                model, peft_info = replace_peft_layers(
                    model = model,
                    peft_config = config.peft_cfg,
                    return_info = True
                )
        
        no_split_module_classes = model._get_no_split_modules(device_map) + lora_module_classes
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
                no_split_module_classes = no_split_module_classes,
            )

        max_memory[0] *= 0.9
        model.tie_weights()
        device_map = infer_auto_device_map(
            model = model,
            max_memory = max_memory,
            no_split_module_classes = no_split_module_classes,
            # verbose = True
        )
        model = dispatch_model(
            model = model,
            device_map = device_map
        )
        # print(model.hf_device_map)

    if model_class in [BaseLM, RewardLM]:
        full_model.lm = model
    elif model_class == AutoModelForCausalLMWithValueHead:
        full_model.pretrained_model = model
        full_model.v_head.to(model.device)
    else:
        full_model = model
    
    return full_model, peft_info


class BaseLM(nn.Module):

    def __init__(self, config: LM_Config, dispatch = False, **kwargs):
        super(BaseLM, self).__init__()
        
        # if dispatch:
        #     self.lm, _  = get_dispatched_model(
        #         config = config,
        #         model_class = AutoModelForCausalLM
        #     )
        # else:
        self.lm = AutoModelForCausalLM.from_pretrained(config.model_pretrain_path, **kwargs)
            
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_pretrain_path)
        self.generation_config = config.generation_config
        if self.generation_config.pad_token_id == None:
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        self.hidden_dim = 1024
        self.device = config.device
        # if self.device == 'cuda':
        #     self = nn.DataParallel(self, device_ids=config.device_ids)
        # self.to(self.device)

    def set_freeze(self, freeze):

        for p in self.parameters():
            p.requires_grad = not freeze

    def decode_single(self, input_ids, attention_mask, prompt_mask):

        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        prompt_mask = prompt_mask.squeeze()

        response_mask = torch.where(prompt_mask == 0, 1, 0) * attention_mask
        prompt_ids = input_ids * attention_mask * prompt_mask
        prompt_ids = prompt_ids[:torch.sum(prompt_mask)]
        response_ids = input_ids * attention_mask * response_mask
        response_ids = response_ids[torch.sum(prompt_mask):]
        
        # all_text = self.tokenizer.convert_ids_to_tokens(input_ids)
        # all_text = ''.join(all_text)
        # prompt = self.tokenizer.convert_ids_to_tokens(prompt_ids)
        # prompt = ''.join(prompt)
        # response = self.tokenizer.convert_ids_to_tokens(response_ids)
        # response = ''.join(response)

        all_text = self.tokenizer.decode(input_ids)
        prompt = self.tokenizer.decode(prompt_ids)
        response = self.tokenizer.decode(response_ids)

        return prompt, response
        

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            labels = None,
        ):

        lm_out = self.lm.forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            # token_type_ids = token_type_ids,
            labels = labels,
            return_dict = True,
            output_hidden_states = True
        )
        
        return lm_out.logits, lm_out.hidden_states[-1], lm_out.loss
    
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        max_new_tokens = None,
        return_seq_without_prompt = True,
        **kwargs
    ):
        
        if max_new_tokens:
            self.generation_config.max_new_tokens = max_new_tokens
            
        prompt_len = prompt.shape[-1]

        response = self.lm.generate(
            input_ids = prompt,
            generation_config = self.generation_config
        )

        if return_seq_without_prompt:
            response = response[:, prompt_len:]
        # print('Response:', self.tokenizer.decode(response.squeeze()))
        
        return response
    
class RewardLMWithoutLMHead(nn.Module):

    def __init__(self, config):
        super(RewardLMWithoutLMHead, self).__init__()
        
        self.lm = AutoModelForCausalLM.from_pretrained(config.model_pretrain_path)

        self.regressor = nn.Sequential(
            nn.Linear(2, 1, bias = False),
            Rearrange('... 1 -> ...')
        )

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            labels = None,
            sample = False
        ):

        lm_out = self.lm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,
            output_hidden_states = True
        )

        pooled = masked_mean(lm_out.hidden_states[-1], attention_mask, dim = 1)
        reward = self.regressor(pooled)

        return reward


class RewardLM(nn.Module):

    def __init__(
        self,
        config: RM_Config,
        dispatch = False,
        **kwargs
    ):
        super(RewardLM, self).__init__()

        lm_config = AutoConfig.from_pretrained(config.model_pretrain_path)
        self.single_forward = False if lm_config.pad_token_id is not None else True
        self.lm = AutoModelForSequenceClassification.from_pretrained(config.model_pretrain_path, **kwargs)
        self.model_info = rm_infos[os.path.split(config.model_pretrain_path)[-1]]
        self.uni_info = rm_infos['universal']

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_pretrain_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = lm_config.max_position_embeddings
        self.device = config.device
        # self.to(self.device)
    
    def set_freeze(self, freeze):

        for p in self.parameters():
            p.requires_grad = not freeze

    def replace_instruct_prompts(
        self,
        target_str: str
    ):
        target_str = target_str.replace(self.uni_info['prompt_prefix'], self.model_info['prompt_prefix'])
        target_str = target_str.replace(self.uni_info['response_prefix'], self.model_info['response_prefix'])

        return target_str
        
    def encode_single(self, prompt: str, response: str):

        prompt = self.replace_instruct_prompts(prompt)
        # response = self.replace_instruct_prompts(response)
        tokenizer_out = self.tokenizer(
            prompt,
            response,
            # padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        ).to(self.device)

        return tokenizer_out
    
    def encode_batch(self, prompt_texts, response_texts, return_padding = False):
        
        if return_padding:
            prompt_texts = [
                self.replace_instruct_prompts(prompt_text)
                for prompt_text in prompt_texts
            ]
            # response_texts = [
            #     self.replace_instruct_prompts(response_text)
            #     for response_text in response_texts
            # ]
            tokenizer_out = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs = list(zip(prompt_texts, response_texts)),
                padding = True,
                max_length = self.max_len,
                truncation = True,
                return_tensors = 'pt',
                return_token_type_ids = True
            ).to(self.device)
            return tokenizer_out
        else:
            tokenizer_outs = {}
            for prompt_text, response_text in zip(prompt_texts, response_texts):
                tokenizer_out = self.encode_single(prompt_text, response_text)
                for k, v in tokenizer_out:
                    if k in tokenizer_out.keys():
                        tokenizer_outs[k].append(v)
                    else:
                        tokenizer_outs[k] = [v]
            
            return tokenizer_outs

    def get_rewards(
        self,
        prompts,
        responses
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
            responses = [responses]

        rewards = []
        for prompt, response in zip(prompts, responses):
            tokenizer_out = self.encode_single(
                prompt = prompt,
                response = response
            )
            reward = self.forward(
                **tokenizer_out
            )
            rewards.append(reward)
        rewards = torch.cat(rewards)
        
        return rewards

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            **kwargs
        ):
        assert input_ids.shape[0] == 1

        lm_out = self.lm(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )

        reward = lm_out.logits[0]

        return reward
    

if __name__ == '__main__':

    model_path = '/home/share/models/huggingface/bit-dny/MindLLM'
    # model_path = '/home/smliu/Pretrain_Models/LocutusqueXFelladrin-TinyMistral248M-Instruct'
    config = RLHF_Config()
    config.dateset_cfg.tokenizer_pretrain_path = model_path
    config.model_cfg.model_pretrain_path = model_path
    config.ref_cfg.model_pretrain_path = model_path

    model = get_model(
        config = config.model_cfg,
        model_class = AutoModelForCausalLMWithValueHead,
    )

    test_out = model.forward()
