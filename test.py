import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig

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

# model_path = '/home/share/models/huggingface/openlm-research/open_llama_3b'
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False)
# # tokenizer.max_length = 1024

# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map = 'auto')

# prompt = "Q: What is the largest animal?\nA: "

# tokenizer_out = tokenizer.encode_plus(prompt, return_tensors='pt', add_special_tokens = False)
# print(tokenizer.decode(tokenizer_out['input_ids'].squeeze()))
# tokenizer_out = {k: v.to(model.device) for k, v in tokenizer_out.items()}
# # generation_config.max_new_tokens = 1024
# sequence = model.generate(**tokenizer_out, max_new_tokens = 512)

# output_text = tokenizer.decode(sequence.squeeze())
# print(output_text)

# -------------------------------------------------------------------------
# model_path = '/home/share/models/huggingface/bit-dny/MindLLM'
model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.max_length = 1024

model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto')

# prompt = "<|endoftext|><user>\n你知道电动车相对传统汽油车有哪些优点吗?\n<assistant>\n"

# generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# # output_text = generator(prompt, max_new_tokens=1024, do_sample=True, num_beams=4, repetition_penalty=0.5, no_repeat_ngram_size=5, return_full_text=False)
# output_text = generator(prompt, max_new_tokens=1024, do_sample=True, eos_token_id=50256, top_k=0, top_p=1)
# print(output_text[0]['generated_text'])


prompt = "<|endoftext|><user>\nI just came out of from jail, any suggestion of my future?\n<assistant>\n"
tokenizer_out = tokenizer.encode_plus(prompt, return_tensors='pt')
tokenizer_out = {k: v.to(model.device) for k, v in tokenizer_out.items()}
generation_config.max_new_tokens = 1024
# sequence = model.generate(**tokenizer_out,  **generation_config.to_dict())
sequence = model.generate(**tokenizer_out, max_new_tokens=1024, do_sample = True)

output_text = tokenizer.decode(sequence.squeeze())
print(output_text)

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


