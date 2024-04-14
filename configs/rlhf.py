from .base import *
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType
from trl import AutoModelForCausalLMWithValueHead

from .ppo import PPO_Config
from .peft import Lora_Config
from .model import LM_Config, RM_Config

@dataclass
class RLHF_Config(PPO_Config):

    task_name: str = 'RLHF_train'
    accelertor_cfg: Accelertor_Config = Accelertor_Config()
    dateset_cfg: Instruct_Dataset_Config = Instruct_Dataset_Config(
        model_name = 'casuallm',
        data_path = os.path.join('/home', 'smliu', 'datasets', 'instruct', 'sharegpt'),
        tokenizer_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        padding_side = 'left',
        max_len = 512,
        tokenize_type = 'prompt_not_pad',
        remove_chinese = False
    )
    model_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        model_class = AutoModelForCausalLMWithValueHead,
        peft_cfg = Lora_Config(
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r = 8,
            lora_alpha = 32,
            lora_dropout = 0.1
        )
    )
    ref_cfg: LM_Config = LM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    )
    reward_cfg: RM_Config = RM_Config(
        model_pretrain_path = os.path.join('/home', 'smliu', 'huggingface', 'OpenAssistant', 'reward-model-deberta-v3-base')
    )

    retokenization: bool = True
    
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

    n_episode: int = 5
    sample_batch_size: int = 1
    n_sample_reuse: int = 2
    n_update_timestep: int = 8
    train_batch_size: int = 4
    n_update_epoch: int = 5

    output_dir: str = os.path.join('.', 'output')
    ckpts_dir: str = os.path.join('/home', 'smliu', 'ckpts')
