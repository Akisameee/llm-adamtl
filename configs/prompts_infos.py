from transformers.generation.configuration_utils import GenerationConfig

dataset_infos = {

    'hh-rlhf': {
        'prompt_prefix': r'Human: ',
        'response_prefix': r'Assistant: '
    },

    'sharegpt': {
        'prompt_prefix': r'Human: ',
        'response_prefix': r'Assistant: '
    }
}

model_infos = {

    'universal': {
        'prompt_prefix': '<prompt_prefix>',
        'prompt_suffix': '<prompt_suffix>',
        'response_prefix': '<response_prefix>',
        'response_suffix': '<response_suffix>',
    },

    'LocutusqueXFelladrin-TinyMistral248M-Instruct': {
        'prompt_prefix': '<|USER|> ',
        'prompt_suffix': '',
        'response_prefix': '<|ASSISTANT|> ',
        'response_suffix': '',
        'eos_token_id': 32001,
        'generation_config': GenerationConfig(
            top_k = 50,
            top_p = 1.0,
            do_sample = True,
            eos_token_id = 32001,
            max_length = 512,
            max_new_tokens = 256
        ),
        'tokenizer_kwargs': dict(
            add_special_tokens = True
        )
    },

    'MindLLM-1b3-chat-zh-v2.0': {
        'prompt_prefix': '<|endoftext|><user>\n',
        'prompt_suffix': '',
        'response_prefix': '\n<assistant>\n',
        'response_suffix': '',
        'eos_token_id': 50256,
        'generation_config': GenerationConfig(
            top_k = 50,
            top_p = 1.0,
            do_sample = True,
            eos_token_id = 50256,
            num_beams = 2,
            repetition_penalty = 0.5,
            no_repeat_ngram_size = 5,
            pad_token_id = 50256,
            max_length = 512,
            max_new_tokens = 256
        ),
        'tokenizer_kwargs': dict(
            add_special_tokens = False
        )
    },

    'MindLLM': {
        'prompt_prefix': '[INST]',
        'prompt_suffix': '',
        'response_prefix': '[/INST]',
        'response_suffix': '',
        'eos_token_id': 50256,
        'generation_config': GenerationConfig(
            top_k = 50,
            top_p = 1.0,
            do_sample = True,
            eos_token_id = 50256,
            num_beams = 2,
            repetition_penalty = 0.5,
            no_repeat_ngram_size = 5,
            pad_token_id = 50256,
            max_length = 512,
            max_new_tokens = 256
        ),
        'tokenizer_kwargs': dict(
            add_special_tokens = True
        )
    },

    'Llama-2-7b-chat-hf': {
        'prompt_prefix': '<s>[INST] ',
        'prompt_suffix': '',
        'response_prefix': ' [/INST] ',
        'response_suffix': '</s>',
        # 'eos_token_id': 50256,
        'generation_config': GenerationConfig(
            top_k = 50,
            top_p = 1.0,
            do_sample = True,
            # eos_token_id = 50256,
            num_beams = 2,
            # repetition_penalty = 0.5,
            # no_repeat_ngram_size = 5,
            # pad_token_id = 50256,
            max_length = 512,
            max_new_tokens = 256
        ),
        'tokenizer_kwargs': dict(
            add_special_tokens = False
        )
    }
}

rm_infos = {

    'universal': {
        'prompt_prefix': '<prompt_prefix>',
        'prompt_suffix': '<prompt_suffix>',
        'response_prefix': '<response_prefix>',
        'response_suffix': '<response_suffix>',
    },

    'gpt2-large-helpful-reward_model': {
        'prompt_prefix': '\n\nHuman: ',
        'prompt_suffix': '',
        'response_prefix': '\n\nAssistant: ',
        'response_suffix': '',
    },

    'gpt2-large-harmless-reward_model': {
        'prompt_prefix': '\n\nHuman: ',
        'prompt_suffix': '',
        'response_prefix': '\n\nAssistant: ',
        'response_suffix': '',
    },
    
    'reward-model-deberta-v3-base': {
        'prompt_prefix': '',
        'prompt_suffix': '',
        'response_prefix': '',
        'response_suffix': '',
    }
}