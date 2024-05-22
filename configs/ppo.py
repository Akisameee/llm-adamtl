from .base import *

from .pefts import Lora_Config
from .datasets_config import Instruct_Dataset_Config
from .model import LM_Config, RM_Config

@dataclass
class Manipulator_Config(Base_Config):

    weighted_loss_type: Optional[Literal['ls', 'sils', 'mo', 'mols']] = 'mols'
    svd_lora_type: Optional[Literal['adaptive']] = None
    svd_lora_random_init: bool = False
    svd_lora_split_percentage: float = None
    n_adapt_step: int = 128

@dataclass
class PPO_Config(Trainer_Config):

    accelertor_cfg: Accelertor_Config = Accelertor_Config()
    dateset_cfg: Instruct_Dataset_Config = None
    model_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        model_class = None,
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
    
    lr: float = 1e-5
    critic_lr: float = 1e-4
    weight_decay: float = 5e-4

    pooled_values: bool = False
    max_norm: float = None
    
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
    critic_pretrain_epoch: int = 10

    output_dir: str = os.path.join('.', 'output')

@dataclass
class MOPPO_Config(PPO_Config):

    manipulator_cfg: Manipulator_Config = None
    