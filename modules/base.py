import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
import inspect
# sys.path.insert(0, '/home/smliu/RLHF')
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, AutoModelForSequenceClassification, LlamaForCausalLM, GPTNeoModel
from trl import AutoModelForCausalLMWithValueHead
from trl.models.modeling_base import PreTrainedModelWrapper

from configs import LM_Config, RM_Config, Model_Config, model_infos, rm_infos

class Base_Warpper(nn.Module):

    supported_args = ()

    def __init__(
        self,
        config: Model_Config
    ) -> None:
        super().__init__()

        self.model_name = config.model_name
        self.model_info = config.model_info
        self.uni_info = model_infos['universal']
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_pretrain_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_info_tokenized = {}
        self.model_info_tokenized['prompt_prefix'] = self.tokenizer.decode(self.tokenizer.encode(self.model_info['prompt_prefix'], add_special_tokens = False))
        self.model_info_tokenized['prompt_suffix'] = self.tokenizer.decode(self.tokenizer.encode(self.model_info['prompt_suffix'], add_special_tokens = False))
        self.model_info_tokenized['response_prefix'] = self.tokenizer.decode(self.tokenizer.encode(self.model_info['response_prefix'], add_special_tokens = False))
        self.model_info_tokenized['response_suffix'] = self.tokenizer.decode(self.tokenizer.encode(self.model_info['response_suffix'], add_special_tokens = False))
        self.pretrained_model = None

    @property
    def device(self):
        return self.pretrained_model.device

    @classmethod
    def _split_kwargs(cls, kwargs):

        supported_kwargs = {}
        unsupported_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs
    
    def set_freeze(self, freeze):

        for p in self.parameters():
            p.requires_grad = not freeze
    
    def forward(
        self,
        **kwargs,
    ):
        return self.pretrained_model(**kwargs)
    
    def remove_special_tokens(self, text: str):

        for special_token in self.tokenizer.all_special_tokens:
            text = text.replace(special_token, '')
        return text
    
    def replace_instruct_prompts(
        self,
        text: str,
        to_uni: bool
    ):
        if to_uni:
            text = text.replace(self.model_info_tokenized['prompt_prefix'], self.uni_info['prompt_prefix'])
            text = text.replace(self.model_info_tokenized['response_prefix'], self.uni_info['response_prefix'])
            if len(self.model_info_tokenized['prompt_suffix']) > 1:
                text = text.replace(self.model_info_tokenized['prompt_suffix'], self.uni_info['prompt_suffix'])
            if len(self.model_info_tokenized['response_suffix']) > 1:
                text = text.replace(self.model_info_tokenized['response_suffix'], self.uni_info['response_suffix'])
        else:
            text = text.replace(self.uni_info['prompt_prefix'], self.model_info['prompt_prefix'])
            text = text.replace(self.uni_info['response_prefix'], self.model_info['response_prefix'])
            text = text.replace(self.uni_info['prompt_suffix'], self.model_info['prompt_suffix'])
            text = text.replace(self.uni_info['response_suffix'], self.model_info['response_suffix'])

        return text
    
    def encode_single(self, prompt: str, response: str):

        prompt = self.replace_instruct_prompts(prompt, to_uni = False)
        tokenizer_out = self.tokenizer(
            prompt,
            response,
            truncation = True,
            return_tensors = 'pt'
        ).to(self.device)

        return tokenizer_out
    
    def encode_batch(self, prompt_texts, response_texts, return_padding = False):
        
        if return_padding:
            prompt_texts = [
                self.replace_instruct_prompts(prompt_text, to_uni = False)
                for prompt_text in prompt_texts
            ]
            tokenizer_out = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs = list(zip(prompt_texts, response_texts)),
                padding = True,
                # max_length = self.max_len,
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

    def decode_single(self, input_ids, attention_mask, prompt_mask):

        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        prompt_mask = prompt_mask.squeeze()

        prompt_ids = input_ids[:torch.sum(prompt_mask)]
        response_ids = input_ids[torch.sum(prompt_mask): torch.sum(attention_mask)]

        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens = False)
        response = self.tokenizer.decode(response_ids, skip_special_tokens = True)
        prompt = self.replace_instruct_prompts(prompt, to_uni = True)
        # response = self.replace_instruct_prompts(response, to_uni = True)
        prompt = self.remove_special_tokens(prompt)

        return prompt, response
    
    def decode_batch(self, inputs_ids, attention_masks, prompt_masks):

        prompts = []
        responses = []
        for input_ids, mask, prompt_mask in zip(inputs_ids, attention_masks, prompt_masks):
            
            prompt, response = self.decode_single(input_ids, mask, prompt_mask)
            prompts.append(prompt)
            responses.append(response)
        
        return prompts, responses
    
    def state_dict(self, *args, **kwargs):

        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        return pretrained_model_state_dict

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]):

        self.pretrained_model.load_state_dict(state_dict, strict = False)

    def post_init(self):

        raise NotImplementedError
    
class BaseLM(Base_Warpper):

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = ()

    def __init__(
        self,
        config: LM_Config,
        **kwargs
    ):
        super().__init__(config)

        supported_kwargs, pretrained_kwargs = self._split_kwargs(kwargs)
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(config.model_pretrain_path, **pretrained_kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        input_keys = list(kwargs.keys())
        for input_key in input_keys:
            if input_key not in inspect.signature(self.pretrained_model.forward).parameters:
                kwargs.pop(input_key)

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        return (lm_logits, last_hidden_state, loss)
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)
    
    def post_init(self):
        pass
    
class BaseRM(Base_Warpper):

    def __init__(
        self,
        config: RM_Config,
        **kwargs
    ):
        super().__init__(config)

        supported_kwargs, pretrained_kwargs = self._split_kwargs(kwargs)
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(config.model_pretrain_path, **pretrained_kwargs)

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

        lm_out = self.pretrained_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            **kwargs
        )

        reward = lm_out.logits[0]
        return reward

    def post_init(self):
        pass

class ValueHead(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output

class BaseLMWithValueHeads(Base_Warpper):

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "n_v_head",
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(
        self,
        config: LM_Config,
        **kwargs
    ):
        super().__init__(config)

        supported_kwargs, pretrained_kwargs = self._split_kwargs(kwargs)
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(config.model_pretrain_path, **pretrained_kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")
        
        self.n_v_head = supported_kwargs.pop("n_v_head", 1)
        # self.use_ref_head = 
        self.v_heads = nn.ModuleList([
            ValueHead(
                self.pretrained_model.config,
                **supported_kwargs
            ) for i in range(self.n_v_head)
        ])

        self._init_weights(**supported_kwargs)
        self.post_init()

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            for v_head in self.v_heads:
                v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
                v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if last_hidden_state.device != self.v_heads[0].summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_heads[0].summary.weight.device)

        values = []
        for v_head in self.v_heads:
            value = v_head(last_hidden_state).squeeze(-1)
            values.append(value)
        values = torch.stack(values, dim = -1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, values)

    def generate(self, *args, **kwargs):

        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):

        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_heads.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict
    
    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]):

        self.pretrained_model.load_state_dict(state_dict, strict = False)
        self.v_heads.load_state_dict(state_dict, strict = False)

    def post_init(self):

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_heads = self.v_heads.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output
            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True

if __name__ == '__main__':

    # config = LM_Config()

    # print(AutoModelForCausalLMWithValueHead)
    model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
    # config.model_pretrain_path = model_path
    # full_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    full_model = BaseLMWithValueHeads.from_pretrained(model_path, n_v_head = 2)
    state_dict = full_model.state_dict()
    full_model.load_state_dict(state_dict)
    print(full_model)