from .base import *

from .pefts import Lora_Config
from .ppo import Manipulator_Config
from .datasets_config import Instruct_Dataset_Config
from .model import LM_Config, RM_Config

@dataclass
class Instruct_MTL_Config(Trainer_Config):

    accelertor_cfg: Accelertor_Config = Accelertor_Config(
        gradient_accumulation_steps = 8
    )
    base_dateset_cfg: Instruct_Dataset_Config = Instruct_Dataset_Config(
        tokenizer_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
        padding_side = 'left',
        max_len = 512,
        tokenize_type = 'prompt_response'
    )
    dataset_data_paths: list[str] = None
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

    manipulator_cfg: Manipulator_Config = Manipulator_Config(
        weighted_loss_type = 'mols',
        svd_lora_type = None,
        svd_lora_random_init = None,
        n_adapt_step = 128
    )
    
    lr: float = 1e-5
    weight_decay: float = 5e-4

    n_episode: int = 1
    train_batch_size: int = 8

    output_dir: str = os.path.join('.', 'output')

    def get_dataset_cfgs(self):

        dataset_configs = []
        for data_path in self.dataset_data_paths:
            dataset_config = deepcopy(self.base_dateset_cfg)
            dataset_config.data_path = data_path
            dataset_configs.append(dataset_config)
                                   
        return dataset_configs