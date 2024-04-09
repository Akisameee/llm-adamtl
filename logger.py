import time
import datetime
import os
import logging
import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import spatial
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

def check_line_is_pareto_front(
    point_line
):
    A, B = point_line[0], point_line[1]
    k = (A[1] - B[1]) / (A[0] - B[0])
    b = A[1] - k * A[0]

    if b / k < 0 and b > 0:
        return True
    else:
        return False
    
def check_plane_is_pareto_front(
    tri_plane
):  
    A, B, C = tri_plane[0], tri_plane[1], tri_plane[2]
    AB = [B[0]-A[0], B[1]-A[1], B[2]-A[2]]
    AC = [C[0]-A[0], C[1]-A[1], C[2]-A[2]]
    
    N = np.cross(AB, AC)
    d = (-N[0]*A[0]) - (N[1]*A[1]) - (N[2]*A[2])

    if d / N[0] < 0 and d / N[1] < 0 and d / N[2] < 0:
        return True
    else:
        return False

def draw_pareto_fronts(
    dataframe: pd.DataFrame,
    axes_names: tuple = ('x', 'y', 'z'),
    save_path: str = ''
):
    if len(axes_names) == 2:

        fig = plt.figure()
        axes = fig.add_subplot(111)

        x = dataframe[axes_names[0]]
        y = dataframe[axes_names[1]]

        vertices = np.array([x, y]).T
        hull = spatial.ConvexHull(vertices)
        faces = hull.simplices
        
        axes.scatter(x, y)
        for face in faces:
            if check_line_is_pareto_front([
                vertices[face[0]],
                vertices[face[1]]
            ]):
                axes.plot(
                    (x[face[0]], x[face[1]]),
                    (y[face[0]], y[face[1]]),
                    color = 'C0',
                    alpha = 0.75
                )

        axes.set_xlabel(axes_names[0])
        axes.set_ylabel(axes_names[1])

    elif len(axes_names) == 3:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')

        x = dataframe[axes_names[0]]
        y = dataframe[axes_names[1]]
        z = dataframe[axes_names[2]]

        vertices = np.array([x, y, z]).T
        hull = spatial.ConvexHull(vertices)
        faces = hull.simplices
        
        axes.scatter(x, y, z)
        line_vertices = []
        for face in faces:
            line = [
                vertices[face[0]],
                vertices[face[1]],
                vertices[face[2]]
            ]
            if check_plane_is_pareto_front(line):
                line_vertices.append(np.array(line))
        axes.add_collection3d(Poly3DCollection(line_vertices, alpha = 0.2))

        axes.set_xlabel(axes_names[0])
        axes.set_ylabel(axes_names[1])
        axes.set_zlabel(axes_names[2])
    
    else:
        raise ValueError('dim > 3')

    plt.savefig(save_path, dpi = 400)
    plt.close()


if __name__ == '__main__':

    logger = Logger('output', 'test')
    for i in range(100):
        logger.step(
            episode = 0,
            timestep = i,
            stat_dict = {
                'reward_a': random.uniform(-5, 5),
                'reward_b': random.uniform(-5, 5),
                'reward_c': random.uniform(-5, 5)
            }
        )
    draw_pareto_fronts(
        dataframe = logger.historys,
        axes_names = ('reward_a', 'reward_b', 'reward_c'),
        save_path = os.path.join(logger.dir, 'pareto_fronts')
    )
    
