from .base import *
# from peft.tuners.lora import LoraConfig
# from peft.utils.peft_types import TaskType
# from trl import AutoModelForCausalLMWithValueHead

from .ppo import PPO_Config
from .datasets_config import Instruct_Dataset_Config
from .pefts import Lora_Config, Panacea_SVD_Config
from .model import LM_Config, RM_Config

@dataclass
class Panacea_PPO_Config(PPO_Config):

    task_name: str = 'Panacea_train'
    accelertor_cfg: Accelertor_Config = Accelertor_Config(
        gradient_accumulation_steps = 2
    )
    dateset_cfg: Instruct_Dataset_Config = Instruct_Dataset_Config(
        data_path = os.path.join('/home', 'smliu', 'datasets', 'instruct', 'sharegpt'),
        tokenizer_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        padding_side = 'left',
        max_len = 512,
        tokenize_type = 'prompt_not_pad'
    )
    model_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        model_class = None,
        peft_cfg = Panacea_SVD_Config(
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r = 8,
            lora_alpha = 32,
            lora_dropout = 0.1,
            pref_dim = 2
        )
    )
    ref_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    )
    reward_cfgs: list[RM_Config] = field(default_factory=list)
    reward_names: list[str] = field(default_factory=list)

    retokenization: bool = True
    
    lr: float = 1e-4
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
    scalariztion_type: Optional[Literal['ls', 'tche']] = 'ls'

    n_episode: int = 1
    sample_batch_size: int = 1
    n_sample_reuse: int = 1
    n_update_timestep: int = 8
    train_batch_size: int = 2
    n_update_epoch: int = 5

    n_save_time: int = 3
    output_dir: str = os.path.join('.', 'output')
