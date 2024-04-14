from .base import *
from .peft import Peft_Config

@dataclass
class Model_Config(Base_Config):

    model_pretrain_path: str = None
    model_class: type = None
    peft_cfg: Peft_Config = None

@dataclass
class LM_Config(Model_Config):

    model_pretrain_path: str = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    peft_cfg: Peft_Config = None
    ckpt_path: str = os.path.join('.', 'ckpts')
    use_ckpt: bool = False
    ckpt_load_path: str = None
    device: str = 'cuda'
    max_len: int = 512
    generation_config: GenerationConfig = generation_config

@dataclass
class RM_Config(Model_Config):

    model_pretrain_path: str = os.path.join('/home', 'smliu', 'Pretrain_Models', 'reward-model-deberta-v3-base')
    peft_cfg: Peft_Config = None
    device: str = 'cuda'