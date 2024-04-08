from .base import *
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType

from .peft import Lora_Config
from .model import LM_Config, RM_Config

@dataclass
class PPO_Config(Base_Config):

    accelertor_cfg: Accelertor_Config = Accelertor_Config()
    model_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        # peft_config = LoraConfig(
        #     task_type = TaskType.CAUSAL_LM,
        #     target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        #     inference_mode = False,
        #     r = 4,
        #     lora_alpha = 32,
        #     lora_dropout = 0.1
        # )
        peft_cfg = Lora_Config(
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r = 4,
            lora_alpha = 32,
            lora_dropout = 0.1
        )
    )
    ref_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    )
    
    lr: float = 1e-4
    weight_decay: float = 5e-4

    pooled_values: bool = False
    max_norm: bool = None
    
    kl_ref_coef: float = 0.2
    kl_type: Optional[Literal['kl', 'abs', 'mse', 'full']] = 'kl'
    eps_clip: float = 0.2
    value_clip: float = 0.2
    beta_s: float = 0.01
    lam: float = 0.95
    gae_gamma: float = 1
    ratio_threshold: float = 10
    value_loss_coef: float = 0.1

    train_batch_size: int = 8
    n_update_epoch: int = 5

    output_dir: str = os.path.join('.', 'output')
