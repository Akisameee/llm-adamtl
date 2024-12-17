import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from accelerate.utils import gather, reduce
from rouge_score.rouge_scorer import RougeScorer

from base_evaluator import Base_Evaluator
from datas import Instruct_MTL_Dataset, Instruct_MTL_Train_Generator
from configs import Lora_Config, Lora_Altered_Config, SVD_Lora_Altered_Config, Instruct_MTL_Config, dataset_infos
# from modules.lms import BaseLM, RewardLM
from modules.base import BaseLMWithValueHeads
from modules.manipulators import MOE_Manipulator_Altered, Base_MTL_Manipulator
from modules.pefts import set_all_adapters, Lora_Linear_Altered, SVD_Lora_Linear_Altered
from modules.utils import shift, log_prob, merge_dict, get_model

TEST = 0

class MTL_Evaluator(Base_Evaluator):

    def __init__(
        self,
        config: Instruct_MTL_Config,
        res_dir: str = None
    ) -> None:
        
        config.task_name = 'MultiTask_eval'
        if res_dir is not None:
            if not os.path.exists(res_dir):
                res_dir = os.path.join(config.output_dir, res_dir)
            if not os.path.split(res_dir)[-1].startswith(config.model_cfg.model_name):
                ckpt_paths = []
                for ckpt_path in os.listdir(res_dir):
                    if ckpt_path.startswith(config.model_cfg.model_name):
                        ckpt_paths.append(ckpt_path)

                ckpt_path = max(ckpt_paths, key = lambda x: eval(x.split('_')[-1]))
                res_dir = os.path.join(res_dir, ckpt_path)
            else:
                ckpt_path = None
        
        super().__init__(
            config = config,
            accelerator_cfg = config.accelertor_cfg,
            model_cfg = config.model_cfg,
            res_dir = res_dir
        )

        # TODO: model_cfg is a model then
        if res_dir is not None:
            self.load(
                self.model,
                ckpt_path = os.path.join(res_dir, 'checkpoint.pt')
            )
        
        self.n_task = len(config.dataset_data_paths)

    def set_pref_vec(
        self,
        pref_vec
    ):
        for module in self.model.modules():
            if hasattr(module, 'set_pref_vec'):
                module.set_pref_vec(pref_vec)
    
    def get_rouge_scores(self, targets, preds):

        scorer = RougeScorer(['rougeL'], use_stemmer = True)
        scores = []
        for target, pred in zip(targets, preds):
            if isinstance(target, str):
                target = [target]
            scores.append(
                scorer.score_multi(target, pred)['rougeL'].fmeasure
            )
        return scores

    @torch.no_grad()
    def evaluate_mtl(
        self,
        val_generators: list,
        val_batch_size: int = 1,
        enable_adapters: bool = True
    ):
        tokenizer: PreTrainedTokenizer = val_generators[0].dataset.tokenizer
        set_all_adapters(
            model = self.model,
            enable = enable_adapters
        )

        pref_dim = self.n_task
        self.set_pref_vec(torch.ones(pref_dim).float())
        
        for task_idx, generator in enumerate(val_generators):
            dataloader = DataLoader(
                dataset = generator,
                batch_size = val_batch_size,
                shuffle = False,
                collate_fn = generator.create_mtl_collator(),
                drop_last = False
            )
            dataloader = self.accelerator.prepare(dataloader)
            max_timestep = len(dataloader) * val_batch_size
            timestep = 0
            tqdm_bar = tqdm(
                total = max_timestep // val_batch_size,
                disable = not self.accelerator.is_main_process
            )

            pref_vec = torch.zeros(pref_dim).float()
            pref_vec[task_idx] = 1
            self.set_pref_vec(pref_vec)
            
            all_targets = []
            all_preds = []
            for batch in dataloader:
                target_texts = batch['target_texts']
                timestep += len(target_texts)
                response_ids = self.accelerator.unwrap_model(self.model).generate(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    **self.generation_kwargs
                )
                response_ids = response_ids[..., batch['input_ids'].size(-1):]
                response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens = True)
                # self.logger.info(
                #     f'target_texts: ' + '\n'.join(target_texts) + '\n\n' + \
                #         'response_texts: ' + '\n'.join(response_texts)
                # )
                all_targets += target_texts
                all_preds += response_texts
                tqdm_bar.update(1)

            scores = self.get_rouge_scores(
                targets = all_targets,
                preds = all_preds
            )
            scores_mean = torch.FloatTensor(scores).mean()
            scores_mean = self.accelerator.reduce(scores_mean.to(self.accelerator.device), reduction = 'mean').item()
            self.logger.step(
                episode = 0,
                timestep = timestep,
                stat_dict = {
                    'Task Name': generator.dataset.subset_names[task_idx],
                    'Rouge-L Score': scores_mean
                },
                eval_step = 0
            )

        self.accelerator.wait_for_everyone()
        self.logger.info(f'{self.task_name} complete.')
        self.logger.save_res()

if __name__ == '__main__':

    res_dir = 'bigbench-mindllm1b3-ada48-2'

    config = Instruct_MTL_Config()
    # config.model_cfg.peft_cfg = Lora_Config()
    config.model_cfg.peft_cfg = Lora_Altered_Config()
    config.from_json(
        os.path.join(
            config.output_dir,
            res_dir,
            'train_config.json'
        )
    )

    dataset = Instruct_MTL_Dataset(configs = config.get_dataset_cfgs())
    dataset.load(mode = 'val', max_sample = 4 if TEST else None, pre_tokenize = False)

    evaluator = MTL_Evaluator(
        config = config,
        res_dir = res_dir
    )
    evaluator.evaluate_mtl(
        val_generators = dataset.get_generator(),
        val_batch_size = 1,
        enable_adapters = True
    )