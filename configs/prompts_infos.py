from transformers.generation.configuration_utils import GenerationConfig

dataset_infos = {

    'hh-rlhf': {
        'prompt_prefix': r'Human: ',
        'response_prefix': r'Assistant: ',
        'sub_classes': [
            'helpful-base',
            'harmless-base'
        ]
    },

    'sharegpt': {
        'prompt_prefix': r'Human: ',
        'response_prefix': r'Assistant: '
    },

    'infinity-instruct': {
        'prompt_prefix': '',
        'response_prefix': '',
        'sub_classes': [
            '7M_domains/code',
            '7M_domains/commonsense',
            '7M_domains/math',
            '7M_domains/subjective'
        ]
    },

    'sciq': {
        'prompt_prefix': '',
        'response_prefix': '',
        'sub_classes': [
            'biology',
            'physics',
            'chemistry',
            'geography'
        ]
    },

    'bigbench': {
        'prompt_prefix': '',
        'response_prefix': '',
        'sub_classes': [
            'ascii_word_recognition',
            'auto_categorization',
            'auto_debugging',
            'bridging_anaphora_resolution_barqa',
            'chess_state_tracking',
            'chinese_remainder_theorem',
            'codenames',
            'conlang_translation',
            'cryptonite',
            'disfl_qa',
            'few_shot_nlg',
            'gem',
            'gender_inclusive_sentences_german',
            'hindi_question_answering',
            'international_phonetic_alphabet_transliterate',
            'language_games',
            'linguistic_mappings',
            'linguistics_puzzles',
            'list_functions',
            'matrixshapes',
            'minute_mysteries_qa',
            'modified_arithmetic',
            'mult_data_wrangling',
            'object_counting',
            'operators',
            'paragraph_segmentation',
            'parsinlu_reading_comprehension',
            'physics_questions',
            'polish_sequence_labeling',
            'qa_wikidata',
            'repeat_copy_logic',
            'rephrase',
            'scientific_press_release',
            'semantic_parsing_in_context_sparc',
            'semantic_parsing_spider',
            'simp_turing_concept',
            'simple_arithmetic_json',
            'simple_text_editing',
            'sufficient_information',
            'tellmewhy',
            'tense',
            'topical_chat',
            'unnatural_in_context_learning',
            'word_sorting',
            'word_unscrambling'
        ]
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
        'prompt_prefix': '<user>\n',
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
            pad_token_id = 75169,
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
            # num_beams = 2,
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