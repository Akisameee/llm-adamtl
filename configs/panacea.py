from .base import *
# from peft.tuners.lora import LoraConfig
# from peft.utils.peft_types import TaskType
# from trl import AutoModelForCausalLMWithValueHead

from .ppo import PPO_Config, MOPPO_Config
from .datasets_config import Instruct_Dataset_Config
from .pefts import Lora_Config, Panacea_SVD_Config
from .model import LM_Config, RM_Config

@dataclass
class Panacea_PPO_Config(MOPPO_Config):

    task_name: str = 'Panacea_train'
    accelertor_cfg: Accelertor_Config = Accelertor_Config(
        gradient_accumulation_steps = 16
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
            use_peft = True,
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            r = 7,
            pref_r = 1,
            lora_alpha = 32,
            lora_dropout = 0.1,
            pref_dim = 2
        )
    )
    ref_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    )
    reward_cfg_0: RM_Config = None
    reward_name_0: str = None
    reward_cfg_1: RM_Config = None
    reward_name_1: str = None
    reward_cfg_2: RM_Config = None
    reward_name_2: str = None
    
    lr: float = 5e-5
    critic_lr: float = 1e-4
    weight_decay: float = 5e-4

    pooled_values: bool = False
    max_norm: float = None
    
    kl_ref_coef: float = 0.5
    kl_type: Optional[Literal['kl', 'abs', 'mse', 'full']] = 'kl'
    eps_clip: float = 0.2
    value_clip: float = 0.2
    beta_s: float = 0.01
    lam: float = 0.95
    gae_gamma: float = 1
    ratio_threshold: float = 10
    value_loss_coef: float = 0.1
    reward_scalariztion_type: Optional[Literal['ls', 'tche']] = None
    loss_manipulator_type: Optional[Literal['ls', 'sils', 'mo', 'mols']] = 'mols'

    n_episode: int = 1
    sample_batch_size: int = 1
    n_sample_reuse: int = 1
    n_update_timestep: int = 64
    train_batch_size: int = 1
    n_update_epoch: int = 1

    n_save_step: int = 3 # n_eval_time
    n_eval_epoch: int = 11
    n_eval_sample: int = 100

    output_dir: str = os.path.join('.', 'output')
