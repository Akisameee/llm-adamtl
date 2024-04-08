import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import DPO_Config, parse_args_into_dataclasses
from data import HumanFeedback_Dataset
from modules.dpo import DPO_Trainer

def main():

    config = DPO_Config()
    config = parse_args_into_dataclasses(
        dataclass = DPO_Config
    )

    dataset = HumanFeedback_Dataset(config.dateset_cfg, config.dateset_cfg.data_path)
    train_ds, eval_ds = dataset.get_generator()

    model = AutoModelForCausalLM.from_pretrained(config.model_cfg.model_pretrain_path)
    tokenizer = AutoTokenizer.from_pretrained(config.dateset_cfg.tokenizer_pretrain_path)

    dpo_trainer = DPO_Trainer(
        config = config,
        model = model
    )

    dpo_trainer.train(
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        n_epoch = config.n_epoch,
        n_eval_step = config.n_eval_step,
        train_batch_size = config.train_batch_size,
        eval_batch_size = config.eval_batch_size
    )

if __name__ == '__main__':

    main()