import time
import datetime
import os
import json
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

from modules.pefts import get_adapter_iter
    
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
        self.train_historys = None
        self.eval_historys = []
        if self.disable:
            return
        
        os.mkdir(self.dir)
        self.logger = get_logger(f'{task_name}_logger', os.path.join(self.dir, f'{task_name}.log'))
        
    def step(self, episode, timestep, stat_dict = None, eval_step = None):

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
        if eval_step is None:
            if self.train_historys is None:
                self.train_historys = pd.DataFrame(columns = columns)
            self.train_historys.loc[len(self.train_historys)] = history
        else:
            assert eval_step <= len(self.eval_historys) and eval_step >=0
            if len(self.eval_historys) == eval_step:
                self.eval_historys.append(pd.DataFrame(columns = columns))
            self.eval_historys[eval_step].loc[len(self.eval_historys[eval_step])] = history

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

        if self.train_historys is not None:
            self.train_historys.to_csv(os.path.join(save_dir, 'train_result.csv'))
            for col_name in self.train_historys.columns:
                if col_name in ['Episode', 'Timestep']:
                    continue
                elif not np.issubdtype(self.train_historys[col_name].dtype, np.number):
                    continue
                else:
                    figure = sns.lineplot(
                        data = self.train_historys,
                        x = 'Timestep',
                        y = col_name
                    )
                    figure.get_figure().savefig(
                        os.path.join(save_dir, col_name),
                        dpi = 400
                    )
                    plt.close()
        else:
            if len(self.eval_historys) > 0:
                for eval_step, eval_history in enumerate(self.eval_historys):
                    eval_history.to_csv(os.path.join(save_dir, f'{eval_step}_eval_result.csv'))

    def save_pareto_front_train(
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
        
        train_plot_args = get_train_plot_args(
            dataframe = self.train_historys,
            axes_names = axes_names,
            vecs_name = vecs_name
        )
        axes = plot_pareto_fronts_2d(**train_plot_args)

        eval_alphas_factor = np.linspace(0.1, 1, len(self.eval_historys))
        for eval_step, eval_history in enumerate(self.eval_historys):
            eval_plot_args = get_eval_plot_args(
                dataframe = eval_history,
                axes_names = axes_names,
                vecs_name = vecs_name
            )
            eval_plot_args['p_alphas'] *= eval_alphas_factor[eval_step]
            eval_plot_args['l_alphas'] *= eval_alphas_factor[eval_step]
            axes = plot_pareto_fronts_2d(**eval_plot_args, prev_axes = axes)

        save_path = os.path.join(save_dir, '_'.join(axes_names) + '_pareto_front') 
        plt.savefig(save_path, dpi = 400)
        plt.close()
        

    def save_pareto_front_test(
        self,
        axes_names: tuple,
        vecs_name: str = None,
        eval_step: int = 0,
        save_dir: str = None
    ):
        if self.disable:
            return

        save_dir = self.dir if save_dir is None else save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        eval_plot_args = get_eval_plot_args(
            dataframe = self.eval_historys[eval_step],
            axes_names = axes_names,
            vecs_name = vecs_name
        )
        plot_pareto_fronts_2d(**eval_plot_args)

        save_path = os.path.join(save_dir, '_'.join(axes_names) + '_pareto_front') 
        plt.savefig(save_path, dpi = 400)
        plt.close()

    def save_tensors(
        self,
        tensors: list[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
        name: str,
        save_dir: str = None
    ):
        if self.disable:
            return
        
        save_dir = self.dir if save_dir is None else save_dir
        if isinstance(tensors, list):
            if isinstance(tensors[0], torch.Tensor):
                tensors = torch.stack(tensors, dim = 0)
            elif isinstance(tensors[0], np.ndarray):
                tensors = np.stack(tensors, dim = 0)
            else:
                raise TypeError(f'Invalid tensors type {type(tensors)}.')
        if isinstance(tensors, torch.Tensor):
            tensors = tensors.cpu().numpy()
        
        save_path = os.path.join(save_dir, name)
        np.save(save_path, tensors)

    def save_conflict_scores(
        self,
        model
    ):
        save_path = os.path.join(self.dir, 'conflict_scores')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for adapter in get_adapter_iter(model):
            if not hasattr(adapter, 'records'):
                return
            tensor_names = [k for k, v in adapter.records.items() if len(v) > 0]
            break

        for tensor_name in tensor_names:
            self.save_tensors(
                [torch.stack(svd_layer.records[tensor_name]) for svd_layer in get_adapter_iter(model)],
                name = tensor_name,
                save_dir = save_path
            )

        module_names = {idx: m[0] for idx, m in enumerate(get_adapter_iter(model, return_name = True))}
        with open(os.path.join(save_path, 'module_names.json'), 'w') as file:
            json.dump(module_names, file)
        

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

def plot_pareto_fronts_2d(
    x, y,
    arrows,
    lines,
    axes_names,
    p_alphas,
    l_alphas,
    scaling,
    color = 'C0',
    sp_points = None,
    prev_axes = None
):
    if prev_axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    else:
        axes = prev_axes

    axes.scatter(x, y, alpha = p_alphas, s = 25 * scaling, color = color)
    if sp_points is not None:
        for sp_point in sp_points:
            axes.scatter(sp_point[0], sp_point[1], color = 'C1')

    if arrows is not None:
        for i in range(len(x)):
            pref_vec = unit_vec(arrows[i]) * 0.1 * scaling
            axes.arrow(
                x[i], y[i],
                pref_vec[0], pref_vec[1],
                alpha = p_alphas[i], color = color,
                width = 0.01 * scaling, head_width = 0.02 * scaling, head_length = 0.02 * scaling
            )
    for line, l_alpha in zip(lines, l_alphas):
        axes.plot(
            (x[line[0]], x[line[1]]),
            (y[line[0]], y[line[1]]),
            color = color, alpha = 0.75 * l_alpha
        )

    axes.set_xlabel(axes_names[0])
    axes.set_ylabel(axes_names[1])

    return axes

def get_train_plot_args(
    dataframe: pd.DataFrame,
    axes_names: tuple = ('x', 'y', 'z'),
    vecs_name: str = None,
):
    if vecs_name:
        pref_vecs = dataframe[vecs_name]
    else:
        pref_vecs = None
    
    p_alphas = np.linspace(0.1, 1, len(dataframe))
    
    if len(axes_names) == 2:

        x = dataframe[axes_names[0]]
        y = dataframe[axes_names[1]]

        vertices = np.array([x, y]).T
        hull = spatial.ConvexHull(vertices)
        lines = hull.simplices

        scaling = min([np.ptp(x.values), np.ptp(y.values)]) / np.log(len(dataframe))

        hull_points = [(x[p], y[p]) for p in np.unique(lines)]
        pareto_front = []
        for line in lines:
            if check_line_is_pareto_front(
                [
                    vertices[line[0]],
                    vertices[line[1]]
                ], points = hull_points
            ):
                pareto_front.append(line)
        
        l_alphas = np.ones(len(pareto_front))
        plot_args = dict(
            x = x,
            y = y,
            arrows = pref_vecs,
            lines = pareto_front,
            axes_names = axes_names,
            p_alphas = p_alphas,
            l_alphas = l_alphas,
            scaling = scaling,
            color = 'C0'
        )

    elif len(axes_names) == 3:
        raise NotImplementedError
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
    
    return plot_args

def get_eval_plot_args(
    dataframe: pd.DataFrame,
    axes_names: tuple = ('x', 'y', 'z'),
    vecs_name: str = None,
):
    p_alphas = np.ones(len(dataframe))
    
    if len(axes_names) == 2:
        
        if vecs_name is not None:
            if isinstance(dataframe[vecs_name][0], (np.ndarray, torch.Tensor)):
                dataframe[vecs_name] = dataframe[vecs_name].apply(lambda x: x.tolist())

            sp_idx = dataframe[dataframe[vecs_name].apply(lambda x: x == [0.0, 0.0])].index
            if len(sp_idx) == 1:
                sp_x, sp_y = dataframe.loc[sp_idx][axes_names[0]], dataframe.loc[sp_idx][axes_names[1]]
                sp_points = [(sp_x, sp_y)]
            else:
                sp_points = None
            dataframe.drop(sp_idx, inplace = True)

            dataframe['sort_val'] = dataframe[vecs_name].apply(lambda x: x[0])
            dataframe = dataframe.sort_values(by = 'sort_val')
            pref_vecs = dataframe[vecs_name]
        else: pref_vecs = None

        x = dataframe[axes_names[0]]
        y = dataframe[axes_names[1]]

        scaling = min([np.ptp(x.values), np.ptp(y.values)]) / np.log(len(dataframe))

        if pref_vecs is not None:
            pareto_front = [(i, i + 1) for i in range(0, len(dataframe) - 1)]
            l_alphas = np.ones(len(pareto_front))
        else:
            pareto_front = None
            l_alphas = None

        plot_args = dict(
            x = x,
            y = y,
            arrows = pref_vecs,
            lines = pareto_front,
            axes_names = axes_names,
            p_alphas = p_alphas,
            l_alphas = l_alphas,
            scaling = scaling,
            sp_points = sp_points,
            color = 'C2'
        )

    else:
        raise NotImplementedError

    return plot_args


if __name__ == '__main__':

    logger = Logger('output', 'test')
    pref_dim = 2
    n_eval_timestep = 250
    
    def sim_train():
        for i in range(1000):
            pref_vec = torch.rand(pref_dim)
            pref_vec = pref_vec / torch.sum(pref_vec)
            vec_len = random.uniform(0, 2)
            vec_angle = (torch.rand(pref_dim) - 0.5).tolist()
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
            if (i + 1) % n_eval_timestep == 0:
                sim_eval((i + 1) // n_eval_timestep - 1)
    
    def sim_eval(eval_step):
        for i in range(11):
            pref_vec = torch.rand(pref_dim)
            pref_vec = pref_vec / torch.sum(pref_vec)
            vec_len = random.uniform(0, 2)
            vec_angle = (torch.rand(pref_dim) - 0.5).tolist()
            logger.step(
                episode = 0,
                timestep = i,
                stat_dict = {
                    'reward_a': vec_angle[0] * vec_len,
                    'reward_b': vec_angle[1] * vec_len,
                    # 'reward_c': random.uniform(-5, 5),
                    'pref_vec': pref_vec
                },
                eval_step = eval_step
            )
        if eval_step == 0:
            logger.step(
                episode = 0,
                timestep = i + 1,
                stat_dict = {
                    'reward_a': 0.5,
                    'reward_b': 0.3,
                    # 'reward_c': random.uniform(-5, 5),
                    'pref_vec': torch.FloatTensor([0, 0])
                },
                eval_step = eval_step
            )
    
    sim_train()
    logger.save_res()
    logger.save_pareto_front_train(
        axes_names = ('reward_a', 'reward_b'),
        # axes_names = ('reward_a', 'reward_b', 'reward_c'),
        vecs_name = 'pref_vec'
    )
    
