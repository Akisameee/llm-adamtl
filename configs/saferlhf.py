from .base import *

from .ppo import PPO_Config, SafePPO_Config, Manipulator_Config
from .datasets_config import Instruct_Dataset_Config
from .pefts import Lora_Config, SVD_Lora_Config
from .model import LM_Config, RM_Config

@dataclass
class Safe_RLHF_Config(SafePPO_Config):

    task_name: str = 'SafeRLHF_train'
    accelertor_cfg: Accelertor_Config = Accelertor_Config(
        gradient_accumulation_steps = 8
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
        peft_cfg = Lora_Config(
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            r = 7,
            lora_alpha = 32,
            lora_dropout = 0.1
        )
    )
    ref_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    )
    reward_cfg: RM_Config = RM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-helpful-reward_model')
    )
    cost_cfg: RM_Config = RM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'huggingface', 'Ray2333', 'gpt2-large-harmless-reward_model')
    )

    lr: float = 1e-4
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
    cost_coef_lr: float = 0.04

    n_episode: int = 1
    sample_batch_size: int = 1
    n_sample_reuse: int = 1
    n_update_timestep: int = 64
    train_batch_size: int = 2
    n_update_epoch: int = 1

    n_save_step: int = 3 # n_eval_time
    n_eval_sample: int = 100

    output_dir: str = os.path.join('.', 'output')
