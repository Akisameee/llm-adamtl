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

    'LocutusqueXFelladrin-TinyMistral248M-Instruct': {
        'prompt_prefix': '<|USER|> ',
        'response_prefix': '<|ASSISTANT|> ',
        'eos_token_id': 32001
    },

    'MindLLM': {
        'prompt_prefix': '<user>\n',
        'response_prefix': '<assistant>\n',
        'eos_token_id': 50256
    }
}