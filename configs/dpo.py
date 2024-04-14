from .base import *
from peft.tuners.lora import LoraConfig
from peft.utils.peft_types import TaskType

from .peft import Lora_Config
from .model import LM_Config, RM_Config

@dataclass
class DPO_Config(Trainer_Config):

    task_name: str = 'DPO_train'
    model_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct')
    dataset_path = os.path.join('/home', 'smliu', 'datasets', 'hf', 'hh-rlhf')

    accelertor_cfg: Accelertor_Config = Accelertor_Config()
    model_cfg = LM_Config(
        model_pretrain_path = model_path,
        peft_cfg = Lora_Config(
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            r = 4,
            lora_alpha = 32,
            lora_dropout = 0.1
        )
    )
    ref_cfg = LM_Config(
        model_pretrain_path = model_path
    )

    dateset_cfg = HumanFeedback_Dataset_Config(
        data_path = dataset_path,
        sub_data_path = ['helpful-base'],
        name = os.path.split(dataset_path)[-1],
        model_name = os.path.split(model_path)[-1],
        tokenizer_pretrain_path = model_path,
        label_pad_token_id = -100,
        truncation_side = 'left'
    )
    
    lr: float = 1e-4
    weight_decay: float = 5e-4
    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Optional[Literal['sigmoid', 'hinge', 'ipo', 'kto_pair']] = 'sigmoid'

    n_epoch: int = 5
    n_eval_step: int = 200
    train_batch_size: int = 2
    eval_batch_size: int = 4

    output_dir: str = os.path.join('.', 'output')
