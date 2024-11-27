from .base import *

@dataclass
class Peft_Config(Base_Config):

    adapter_name: str = None
    target_modules: list[str] = None

@dataclass
class Lora_Config(Peft_Config):

    adapter_name: str = 'lora'
    r: int = 8
    lora_alpha: int = 1
    lora_dropout: float = 0.0

@dataclass
class Lora_Altered_Config(Lora_Config):

    adapter_name: str = 'lora-altered'
    pref_dim: int = 2
    pref_r: int = 1

@dataclass
class SVD_Lora_Config(Peft_Config):

    adapter_name: str = 'panacea'
    r: int = 8
    pref_r: int = 1
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    pref_dim: int = 2

    init_strategy: Optional[Literal['b_zero', 'diag_zero']] = None

@dataclass
class SVD_Lora_Altered_Config(SVD_Lora_Config):

    adapter_name: str = 'svd_lora_altered'