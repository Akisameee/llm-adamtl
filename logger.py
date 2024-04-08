import time
import datetime
import os
import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
    
def get_logger(name='logger', file_name='./log.txt'):
    
    # console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_name, mode="a", encoding="utf-8")

    console_fmt = "(%(levelname)s) %(asctime)s - %(name)s:\n%(message)s"
    file_fmt = "(%(levelname)s) %(asctime)s  - %(name)s:\n%(message)s"

    console_formatter = logging.Formatter(fmt = console_fmt)
    file_formatter = logging.Formatter(fmt = file_fmt)

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.INFO)
    tqdm_handler.setFormatter(fmt = console_formatter)
    file_handler.setFormatter(fmt = file_formatter)

    logging.basicConfig(level='INFO', handlers=[tqdm_handler, file_handler])
    logger = logging.getLogger(name)

    return logger

class TqdmLoggingHandler(logging.StreamHandler):

    def __init__(self, level = logging.NOTSET):
        logging.StreamHandler.__init__(self)

    def emit(self, record):

        msg = self.format(record)
        tqdm.write(msg)
        self.flush()
    
class Logger():

    def __init__(self, output_dir, task_name, disable = False) -> None:
        
        self.disable = disable
        if self.disable:
            return
        self.output_dir = output_dir
        current_time = datetime.datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H-%M-%S')
        self.dir = os.path.join(output_dir, time_str)
        os.mkdir(self.dir)

        self.logger = get_logger(f'{task_name}_logger', os.path.join(self.dir, f'{task_name}_log.txt'))
        self.historys = None

    def step(self, episode, timestep, stat_dict = None):

        if self.disable:
            return
        
        log_str = f'Episode: {episode} Timestep: {timestep}\n'
        log_str += ' '.join([f'{key}: {value:.4g}'for key, value in stat_dict.items()]) if stat_dict is not None else ''
        
        self.info(log_str)

        columns = ['Episode', 'Timestep'] + list(stat_dict.keys())
        history = stat_dict if stat_dict is not None else {}
        history['Episode'] = episode
        history['Timestep'] = timestep
        if self.historys is None:
            self.historys = pd.DataFrame(columns=columns)
        self.historys.loc[len(self.historys)] = history
        # print(self.historys)

    def info(self, log_str):

        if self.disable:
            return
        
        self.logger.info(log_str)

    def save_res(self):

        if self.disable:
            return
        
        # print(self.historys)
        self.historys.to_csv(os.path.join(self.dir, 'train_result.csv'))
        for col_name in self.historys.columns:
            if col_name in ['Episode', 'Timestep']:
                continue
            else:
                figure = sns.lineplot(
                    data = self.historys,
                    x = 'Timestep',
                    y = col_name
                )
                figure.get_figure().savefig(os.path.join(self.dir, col_name), dpi=400)
                plt.close()


if __name__ == '__main__':

    logger = Logger('output', 'test')
    for epoch in tqdm(range(10)):
        time.sleep(1)
        logger.info(f'test msg:{epoch}')
