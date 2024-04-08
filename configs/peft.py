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