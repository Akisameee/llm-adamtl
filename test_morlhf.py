import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from modules.utils import get_model
from modules.base import BaseLMWithValueHeads
from configs import Panacea_PPO_Config, RLHF_Config

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

model_path = '/home/smliu/huggingface/bit-dny/MindLLM-1b3-chat-zh-v2.0'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.max_length = 1024
# config = Panacea_PPO_Config()
config = Panacea_PPO_Config()
model_cfg = config.model_cfg
model_cfg.model_pretrain_path = model_path
model_cfg.model_class = BaseLMWithValueHeads
model, _ = get_model(
    config = config.model_cfg,
    dispatch = False
)

model = model.pretrained_model
path_to_checkpoint = './output/Panacea_train 2024-04-16 23-05-04/MindLLM-1b3-chat-zh-v2.0_0_32/checkpoint.pt'
state_dict = torch.load(path_to_checkpoint)
model.load_state_dict(state_dict, strict = False)
model = model.to('cuda')

prompt = "<|endoftext|><user>\nI just came out of from jail, any suggestion of my future?\n<assistant>\n"
tokenizer_out = tokenizer.encode_plus(prompt, return_tensors='pt')
tokenizer_out = {k: v.to(model.device) for k, v in tokenizer_out.items()}
generation_config.max_new_tokens = 1024
sequence = model.generate(**tokenizer_out,  **generation_config.to_dict())
# sequence = model.generate(**tokenizer_out, max_new_tokens=1024, do_sample = True)

output_text = tokenizer.decode(sequence.squeeze())
print(output_text)