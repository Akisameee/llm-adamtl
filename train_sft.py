import torch
import time

from datas.instruct_mtl_dataset import Instruct_Pref_Dataset
from configs import SFT_Train_Config
from modules.lms import BaseLM
from torch.utils.data import DataLoader

class Timer():

    def __init__(self):
        
        self.start_time_stamp = 0

    def start(self):

        self.start_time_stamp = time.time()

    def end(self):

        assert self.start_time_stamp != 0
        end_time_stamp = time.time()
        time_cost = end_time_stamp - self.start_time_stamp
        self.start_time_stamp = 0

        return time_cost

class Trainer():
    
    def __init__(self, model, optimizer, config) -> None:
        
        self.device = config.device
        self.ckpt_path = config.ckpt_path
        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.timer = Timer()

        self.train_loss_avg = []
        self.val_loss_avg = []

        self.min_loss = 9999

    def train_epoch(self, loader, epoch):

        model.train()
        loss_sum = 0
        self.timer.start()
        for data in loader:

            input_ids = data['input_ids'].squeeze().to(self.device)
            token_type_ids = data['token_type_ids'].squeeze().to(self.device)
            attention_mask = data['attention_mask'].squeeze().to(self.device)
            labels = data['labels'].squeeze().to(self.device)
            batch_size = input_ids.size(0)

            lm_out = self.model(
                input_ids = input_ids,
                token_type_ids = token_type_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            # logits = lm_out['logits']
            loss = lm_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.item()

        time_cost = self.timer.end()
        loss_avg = loss_sum / len(loader.dataset)
        print(f'Epoch:{epoch}, Train_Loss:{loss_avg:.4g}, Time_Cost:{time_cost:.4g}s')

        self.train_loss_avg.append(loss_avg)

    def val_epoch(self, loader):

        model.eval()
        loss_sum = 0
        n_correct = 0
        self.timer.start()
        with torch.no_grad():
            for data in loader:

                input_ids = data['input_ids'].squeeze().to(self.device)
                token_type_ids = data['token_type_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)
                batch_size = input_ids.size(0)

                lm_out = self.model(
                    input_ids = input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )
                # logits = lm_out['logits']
                loss = lm_out['loss']

                loss_sum += loss.item()

        time_cost = self.timer.end()
        loss_avg = loss_sum / len(loader.dataset)
        print(f'Val_Loss:{loss_avg:.4g}, Time_Cost:{time_cost:.4g}s')

        self.val_loss_avg.append(loss_avg)

        if loss_avg > self.min_loss:
            self.min_loss = loss_avg
            torch.save(model.state_dict(), self.ckpt_path)


if __name__ == '__main__':
    
    config = SFT_Train_Config()
    dataset = Instruct_Pref_Dataset(config.instruct_dataset_config)

    train_generator, val_generator = dataset.get_generator()
    train_loader = DataLoader(train_generator, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_generator, batch_size=config.val_batch_size, shuffle=True)

    model = BaseLM(config)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config
    )

    for epoch in range(config.epoch):
        trainer.train_epoch(train_loader, epoch)
        trainer.val_epoch(val_loader)