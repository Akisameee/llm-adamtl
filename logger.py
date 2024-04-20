import time
import datetime
import os
import logging
import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy import spatial
from tqdm.auto import tqdm

arrow_kwargs = {
    'color': 'C0',
    'linewidth': 0.05,
    # 'headwidth': 0.1,
    # 'headlength': 0.1
}
    
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
        self.output_dir = output_dir
        current_time = datetime.datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H-%M-%S')
        self.dir_name = f'{task_name} {time_str}'
        self.dir = os.path.join(output_dir, self.dir_name)
        self.historys = None
        if self.disable:
            return
        
        os.mkdir(self.dir)
        self.logger = get_logger(f'{task_name}_logger', os.path.join(self.dir, f'{task_name}.log'))
        
    def step(self, episode, timestep, stat_dict = None):

        if self.disable:
            return
        
        log_str = f'Episode: {episode} Timestep: {timestep}\n'
        stat_strs = []
        for key, value in stat_dict.items():
            if isinstance(value, (int, float)):
                stat_strs.append(f'{key}: {value:.4g}')
            else:
                if isinstance(value, torch.Tensor):
                    value = value.detach().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                value = [f'{num:.3f}' for num in value]
                stat_strs.append(f'{key}: {value}')
        
        log_str += ' '.join(stat_strs)
        self.info(log_str)

        columns = ['Episode', 'Timestep'] + list(stat_dict.keys())
        history = stat_dict if stat_dict is not None else {}
        history['Episode'] = episode
        history['Timestep'] = timestep
        if self.historys is None:
            self.historys = pd.DataFrame(columns=columns)
        self.historys.loc[len(self.historys)] = history

    def info(self, log_str):

        if self.disable:
            return
        self.logger.info(log_str)

    def warning(self, log_str):

        if self.disable:
            return
        self.logger.warning(log_str)

    def save_res(
        self,
        save_dir: str = None
    ):
        if self.disable:
            return
        
        save_dir = self.dir if save_dir is None else save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.historys.to_csv(os.path.join(save_dir, 'train_result.csv'))
        for col_name in self.historys.columns:
            if col_name in ['Episode', 'Timestep']:
                continue
            elif not np.issubdtype(self.historys[col_name].dtype, np.number):
                continue
            else:
                figure = sns.lineplot(
                    data = self.historys,
                    x = 'Timestep',
                    y = col_name
                )
                figure.get_figure().savefig(
                    os.path.join(save_dir, col_name),
                    dpi = 400
                )
                plt.close()

    def save_pareto_front(
        self,
        axes_names: tuple,
        vecs_name: str = None,
        save_dir: str = None
    ):
        if self.disable:
            return

        save_dir = self.dir if save_dir is None else save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        draw_pareto_fronts(
            dataframe = self.historys,
            axes_names = axes_names,
            vecs_name = vecs_name,
            save_path = os.path.join(save_dir, '_'.join(axes_names) + '_pareto_front') 
        )

def check_line_is_pareto_front(
    point_line,
    points
):
    A, B = point_line[0], point_line[1]
    k = (A[1] - B[1]) / (A[0] - B[0])
    b = A[1] - k * A[0]

    if k < 0:
        for point in points:
            if point[1] - 1e-4 > k * point[0] + b:
                return False
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
    
def unit_vec(vec):

    length = np.sqrt(sum([x**2 for x in vec]))
    return np.array(vec) / length

def draw_pareto_fronts(
    dataframe: pd.DataFrame,
    axes_names: tuple = ('x', 'y', 'z'),
    vecs_name: str = None,
    save_path: str = ''
):
    if vecs_name:
        pref_vecs = dataframe[vecs_name]
    else:
        pref_vecs = None
    
    alphas = np.linspace(0.1, 1, len(dataframe))
    
    if len(axes_names) == 2:
        fig = plt.figure()
        axes = fig.add_subplot(111)

        x = dataframe[axes_names[0]]
        y = dataframe[axes_names[1]]

        vertices = np.array([x, y]).T
        hull = spatial.ConvexHull(vertices)
        lines = hull.simplices

        scaling = min([np.ptp(x.values), np.ptp(y.values)]) / np.log(len(dataframe))

        axes.scatter(x, y, alpha = alphas, s = 25 * scaling)

        if pref_vecs is not None:
            for i in range(len(x)):
                pref_vec = unit_vec(pref_vecs[i]) * 0.1 * scaling
                axes.arrow(
                    x[i], y[i],
                    pref_vec[0], pref_vec[1],
                    alpha = alphas[i], color = 'C0',
                    width = 0.01 * scaling, head_width = 0.02 * scaling, head_length = 0.02 * scaling
                )
        
        hull_points = [(x[p], y[p]) for p in np.unique(lines)]
        print(np.unique(lines))
        for line in lines:
            if check_line_is_pareto_front(
                [
                    vertices[line[0]],
                    vertices[line[1]]
                ], points = hull_points
            ):
                axes.plot(
                    (x[line[0]], x[line[1]]),
                    (y[line[0]], y[line[1]]),
                    color = 'C0', alpha = 0.75
                )

        axes.set_xlabel(axes_names[0])
        axes.set_ylabel(axes_names[1])

    elif len(axes_names) == 3:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        x = dataframe[axes_names[0]]
        y = dataframe[axes_names[1]]
        z = dataframe[axes_names[2]]
        axes.scatter(x, y, z, alpha = alphas)
        
        if pref_vecs is not None:
            for i in range(len(x)):
                pref_vec = unit_vec(pref_vecs[i]) / 2
                axes.quiver(
                    x[i], y[i], z[i],
                    pref_vec[0], pref_vec[1], pref_vec[2],
                    alpha = alphas[i],
                    color = 'C0'
                )
                print(alphas[i])
        
        vertices = np.array([x, y, z]).T
        hull = spatial.ConvexHull(vertices)
        lines = hull.simplices
        line_vertices = []
        for line in lines:
            line = [
                vertices[line[0]],
                vertices[line[1]],
                vertices[line[2]]
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
    pref_dim = 2
    for i in range(2000):
        pref_vec = torch.rand(pref_dim)
        pref_vec = pref_vec / torch.sum(pref_vec)
        vec_len = random.uniform(0, 2)
        vec_angle = (torch.rand(pref_dim) - 0.5).numpy().tolist()
        logger.step(
            episode = 0,
            timestep = i,
            stat_dict = {
                'reward_a': vec_angle[0] * vec_len,
                'reward_b': vec_angle[1] * vec_len,
                # 'reward_c': random.uniform(-5, 5),
                'pref_vec': pref_vec
            }
        )
    logger.save_res()
    logger.save_pareto_front(
        axes_names = ('reward_a', 'reward_b'),
        # axes_names = ('reward_a', 'reward_b', 'reward_c'),
        vecs_name = 'pref_vec'
    )
    
