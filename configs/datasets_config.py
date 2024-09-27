from .base import *

from .prompts_infos import model_infos, rm_infos, dataset_infos

@dataclass
class Dataset_Config(Base_Config):

    data_path: str = None
    sub_data_path: list = None
    tokenizer_pretrain_path: str = None
    max_len: int = 512
    remove_chinese: bool = True

    @property
    def model_name(self):
        return os.path.split(self.tokenizer_pretrain_path)[-1]

    @property
    def name(self):
        if 'Infinity-Instruct' in self.data_path.split('/'):
            # path_split = os.path.split(self.data_path)
            # return '/'.join(path_split[path_split.index('Infinity_Instruct'):])
            return 'infinity-instruct'
        else:
            return os.path.split(self.data_path)[-1]
    
    @property
    def model_info(self):
        return model_infos[self.model_name]
    
    @property
    def dataset_info(self):
        return dataset_infos[self.name]
    

@dataclass
class Instruct_Dataset_Config(Dataset_Config):

    data_path: str = None
    sub_data_path: list = None
    padding_side: Optional[Literal['left', 'right']] = 'left'
    prompt_only: bool = False
    tokenize_type: Optional[Literal['prompt_pad', 'prompt_not_pad', 'prompt_response']] = 'prompt_pad'

@dataclass
class HumanFeedback_Dataset_Config(Dataset_Config):

    data_path: str = None
    sub_data_path: list = None
    pad_token_id: int = None
    label_pad_token_id: int = -100
    truncation_side: Optional[Literal['left', 'right']] = 'left'