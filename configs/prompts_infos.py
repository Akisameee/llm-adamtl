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
        'response_prefix': '<response_prefix>',
    },

    'LocutusqueXFelladrin-TinyMistral248M-Instruct': {
        'prompt_prefix': '<|USER|> ',
        'response_prefix': '<|ASSISTANT|> ',
        'eos_token_id': 32001,
        'generation_config': GenerationConfig(
            top_k = 50,
            top_p = 1.0,
            do_sample = True,
            eos_token_id = 32001,
        )
    },

    'MindLLM-1b3-chat-zh-v2.0': {
        'prompt_prefix': '<|endoftext|><user>\n',
        'response_prefix': '\n<assistant>\n',
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
            max_new_tokens = 256
        )
    }
}

rm_infos = {

    'universal': {
        'prompt_prefix': '<prompt_prefix>',
        'response_prefix': '<response_prefix>',
    },

    'gpt2-large-helpful-reward_model': {
        'prompt_prefix': '\n\nHuman: ',
        'response_prefix': '\n\nAssistant: '
    },
    
    'reward-model-deberta-v3-base': {
        'prompt_prefix': '',
        'response_prefix': ''
    }
}