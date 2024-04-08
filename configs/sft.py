from .base import *

@dataclass
class SFT_Train_Config(object):

    def __init__(self) -> None:

        self.instruct_dataset_config = Instruct_Dataset_Config(
            model_name = 'gpt2-tiny',
            train_data_path = os.path.join('/home', 'smliu', 'datasets', 'sft', 'train_sft.csv'),
            val_data_path = os.path.join('/home', 'smliu', 'datasets', 'sft', 'dev_sft.csv'),
            tokenizer_pretrain_path = os.path.join('/home', 'smliu', 'Pretrain_Models', 'LocutusqueXFelladrin-TinyMistral248M-Instruct'),
            prompt_only = False
        )
        self.ckpt_path = os.path.join('.', 'ckpts')
        self.use_ckpt = False
        self.ckpt_load_path = None
        self.device = 'cuda'
        self.max_len = 1024
        self.epoch = 5
        self.train_batch_size = 2
        self.val_batch_size = 4
        self.lr = 1e-4
        self.weight_decay = 5e-4