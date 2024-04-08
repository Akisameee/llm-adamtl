import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from peft import get_peft_model

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2Model, AutoConfig, AutoTokenizer
from configs import SFT_Train_Config, LM_Config
from modules.utils import eval_decorator, top_k, top_p, gumbel_sample, masked_mean


class BaseLM(nn.Module):

    def __init__(self, config: LM_Config):
        super(BaseLM, self).__init__()
        
        self.lm = AutoModelForCausalLM.from_pretrained(config.model_pretrain_path)
        # if config.peft_cfg is not None:
        #     self.lm = get_peft_model(self.lm, config.peft_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_pretrain_path)
        self.generation_config = config.generation_config
        if self.generation_config.pad_token_id == None:
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        self.hidden_dim = 1024

        self.device = config.device
        # if self.device == 'cuda':
        #     self = nn.DataParallel(self, device_ids=config.device_ids)
        self.to(self.device)

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
        
    # @torch.no_grad()
    # @eval_decorator
    # def generate(
    #         self,
    #         seq_len,
    #         prompt = None,
    #         temperature = 1.,
    #         filter_logits_fn = top_k,
    #         filter_thres = 0.9,
    #         pad_value = 0.,
    #         eos_token = None,
    #         return_seq_without_prompt = True,
    #         **kwargs
    #     ):

    #     if eos_token is None:
    #         eos_token = self.tokenizer.eos_token_id

    #     if prompt is None:
    #         prompt = torch.randint(0, self.num_tokens, (1, 1))
    #         prompt = prompt.to(self.device)
    #         return_seq_without_prompt = False

    #     prompt, leading_dims = pack([prompt], '* n')

    #     n, out = prompt.shape[-1], prompt.clone()

    #     sample_num_times = max(1, seq_len - prompt.shape[-1])

    #     for _ in range(sample_num_times):
    #         logits, embeds, _ = self.forward(out, **kwargs)
    #         logits, embeds = logits[:, -1], embeds[:, -1]

    #         if filter_logits_fn is not None:
    #             logits = filter_logits_fn(logits, thres = filter_thres)

    #         sample = gumbel_sample(logits, temperature = temperature, dim = -1)
    #         out, _ = pack([out, sample], 'b *')

    #         if eos_token is not None:
    #             is_eos_tokens = (out == eos_token)

    #             if is_eos_tokens.any(dim = -1).all():
    #                 # mask out everything after the eos tokens
    #                 shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
    #                 mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
    #                 out = out.masked_fill(mask, pad_value)
    #                 break

    #     out, = unpack(out, leading_dims, '* n')

    #     if not return_seq_without_prompt:
    #         return out

    #     return out[..., n:]
    
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

    def __init__(self, config):
        super(RewardLM, self).__init__()

        lm_config = AutoConfig.from_pretrained(config.model_pretrain_path)
        # if lm_config.architectures[0] != 'GPT2ForSequenceClassification':
        #     raise NotImplementedError
        
        self.lm = AutoModelForSequenceClassification.from_pretrained(config.model_pretrain_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_pretrain_path)

        self.max_len = lm_config.max_position_embeddings
        self.device = config.device
        self.to(self.device)
    
    def set_freeze(self, freeze):

        for p in self.parameters():
            p.requires_grad = not freeze
        
    def encode_single(self, prompt_text, response_text):

        tokenizer_out = self.tokenizer(
            prompt_text,
            response_text,
            # padding = 'max_length',
            max_length = self.max_len,
            truncation=True,
            return_tensors = 'pt',
            return_token_type_ids = True
        ).to(self.device)

        return (
            tokenizer_out['input_ids'],
            tokenizer_out['attention_mask'],
            tokenizer_out['token_type_ids']
        )
    
    def encode_batch(self, prompt_texts, response_texts, return_padding = False):
        
        if return_padding:
            tokenizer_out = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs = list(zip(prompt_texts, response_texts)),
                padding = True,
                max_length = self.max_len,
                truncation = True,
                return_tensors = 'pt',
                return_token_type_ids = True
            ).to(self.device)
            return (
                tokenizer_out['input_ids'],
                tokenizer_out['attention_mask'],
                tokenizer_out['token_type_ids']
            )
        else:
            outs = []
            for prompt_text, response_text in zip(prompt_texts, response_texts):
                tokenizer_out = self.encode_single(prompt_text, response_text)
                outs.append(tokenizer_out)
            return (
                [out[0] for out in outs],
                [out[1] for out in outs],
                [out[2] for out in outs]
            )
    
    # def get_text_reward()

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            **kwargs
        ):

        lm_out = self.lm(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )

        reward = lm_out.logits.squeeze()
        # if len(reward.size()) < 2:
        #     reward = reward.unsqueeze(0)

        return reward
