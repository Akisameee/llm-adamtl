import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from modules.utils import get_model
from configs import Panacea_PPO_Config

generation_config = GenerationConfig(
    # top_k = 50,
    # top_p = 1.0,
    do_sample = True,
    eos_token_id = 50256,
    max_new_tokens=1024,
    num_beams=4,
    repetition_penalty=0.5,
    no_repeat_ngram_size=5,
    pad_token_id = 50256
)

# model_path = '/home/smliu/huggingface/openlm-research/open_llama_3b_v2'
model_path = '/home/share/models/huggingface/meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.max_length = 1024

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map = 'auto')
print(model.hf_device_map)

prompt = "<s>[INST] What is the largest animal? [/INST]"

tokenizer_out = tokenizer(prompt, return_tensors='pt', add_special_tokens = False)
# print(tokenizer.decode(tokenizer_out['input_ids'].squeeze()))
tokenizer_out = {k: v.to(model.device) for k, v in tokenizer_out.items()}
# generation_config.max_new_tokens = 1024
sequence = model.generate(tokenizer_out['input_ids'], max_new_tokens = 512)

output_text = tokenizer.decode(sequence.squeeze())
print(output_text)

# -------------------------------------------------------------------------
# tokenizer_path = '/home/share/ours_llm/blue_space'
# model_path = '/home/share/ours_llm/models/model3b/checkpoint-80000'
# # model_path = '/home/share/models/huggingface/openlm-research/open_llama_3b'

# # model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = False)
# tokenizer.max_length = 1024
# config = Panacea_PPO_Config()
# model_cfg = config.model_cfg
# model_cfg.model_pretrain_path = model_path

# model, _ = get_model(
#     config = config.model_cfg,
#     dispatch = False
# )
# model = model.to('cuda')

# # prompt = "<|endoftext|><user>\nI just came out of from jail, any suggestion of my future?\n<assistant>\n"
# prompt = "<s>[INST]I just came out of from jail, any suggestion of my future?[/INST]"
# tokenizer_out = tokenizer.encode_plus(prompt, return_tensors='pt')
# tokenizer_out = {k: v.to(model.device) for k, v in tokenizer_out.items()}
# generation_config.max_new_tokens = 1024
# # sequence = model.generate(**tokenizer_out,  **generation_config.to_dict())
# sequence = model.generate(**tokenizer_out, max_new_tokens=1024, do_sample = True)

# output_text = tokenizer.decode(sequence.squeeze())
# print(output_text)

# -------------------------------------------------------------------------

# model_path = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# # tokenizer.max_length = 1024
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# # p = '\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:'
# # e = 'Sorry, I don\'t understand.'
# # p = '\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:'
# # e = 'It\'s great to hear that you have been released from jail.'
# # e = 'Go back to jail you scum.'
# p = '\n\nHuman: Explain nuclear fusion like I am five \n\nAssistant:'
# e = 'Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants.'
# tokenizer_out = tokenizer(p, e, truncation=True, return_tensors='pt')
# with torch.no_grad():
#     score = model.forward(**tokenizer_out).logits[0]
# print(score)


