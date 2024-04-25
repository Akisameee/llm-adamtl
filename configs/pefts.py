from .base import *

@dataclass
class Peft_Config(Base_Config):

    adapter_name: str = None
    use_peft: bool = False
    target_modules: list[str] = None

@dataclass
class Lora_Config(Peft_Config):

    adapter_name: str = 'lora'
    r: int = 8
    lora_alpha: int = 1
    lora_dropout: float = 0.0

@dataclass
class Panacea_SVD_Config(Peft_Config):

    adapter_name: str = 'panacea'
    r: int = 8
    pref_r: int = 1
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    pref_dim: int = 2