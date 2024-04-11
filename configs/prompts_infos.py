import re

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
        'eos_token_id': 32001
    },

    'MindLLM': {
        'prompt_prefix': '<|endoftext|><user>\n',
        'response_prefix': '\n<assistant>\n',
        'eos_token_id': 50256
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