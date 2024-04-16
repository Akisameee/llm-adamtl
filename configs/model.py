from configs.base import *
from configs.peft import Peft_Config

from configs.prompts_infos import model_infos, rm_infos

@dataclass
class Model_Config(Base_Config):

    model_pretrain_path: str = None
    model_class: type = None
    peft_cfg: Peft_Config = None

    @property
    def model_name(self):
        return os.path.split(self.model_pretrain_path)[-1]
    
    @property
    def model_info(self):
        raise NotImplementedError

@dataclass
class LM_Config(Model_Config):

    model_pretrain_path: str = None
    # model_class: type = None
    peft_cfg: Peft_Config = None
    ckpt_path: str = os.path.join('.', 'ckpts')
    use_ckpt: bool = False
    ckpt_load_path: str = None
    device: str = 'cuda'
    max_len: int = 512

    @property
    def model_info(self):
        return model_infos[self.model_name]

@dataclass
class RM_Config(Model_Config):

    model_pretrain_path: str = os.path.join('/home', 'smliu', 'Pretrain_Models', 'reward-model-deberta-v3-base')
    peft_cfg: Peft_Config = None
    device: str = 'cuda'

    @property
    def model_info(self):
        return rm_infos[self.model_name]