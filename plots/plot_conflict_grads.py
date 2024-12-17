import numpy as np
import os
import json
import matplotlib.pyplot as plt

tab10_colors = plt.colormaps.get_cmap('tab10')

def signed_log(array: np.ndarray):

    return np.sign(array) * np.log(np.abs(array) + 1)

def plot_scores_line(
    ax,
    x,
    ys: np.ndarray,
    t_idx
):
    # ys = np.log(ys)
    y_mean = ys.sum(axis = -1)
    y_std = ys.std(axis = -1)
    y_max = ys.max(axis = -1)
    y_min = ys.min(axis = -1)
    # y_max = signed_log(y_max)
    # y_min = signed_log(y_min)
    # ax.plot(x, y_mean, alpha = 0.2)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Scores')
    # ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha = 0.1)
    ax.fill_between(
        x,
        y_min,
        y_max,
        alpha = 0.05
    )

def plot_scores_bar(
    ax,
    x,
    ys: np.ndarray,
    t_idx
):
    y_mean = ys.sum(axis = -1)
    y_std = ys.std(axis = -1)
    ax.bar(x, y_mean, alpha = 0.1, color = 'gray')
    ax.errorbar(x, y_mean, yerr = y_std, fmt = 'o', alpha = 0.05, color = 'gray')

def plot_scores_scatter(
    ax,
    x,
    ys: np.ndarray,
    t_idx: int
):
    # ys = ys.reshape(ys.)
    for timestep in range(ys.shape[-1]):
        ax.scatter(x, ys[:, timestep], alpha = 0.05, color = tab10_colors(t_idx + 1))

# def plot_scores_violin(
#     ax,
#     x,
#     ys: np.ndarray,
#     t_idx
# ):
#     data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4]  

#     ax = fig.add_axes([0,0,1,1])  
#     ax.violinplot(data_to_plot)

if __name__ == '__main__':

    dir_path = './output/bigbench-mindllm1b3-ada48_test-2'
    # dir_path = './output/completed/Panacea_train 71 random'
    data_path = os.path.join(dir_path, 'conflict_scores')
    res_path = os.path.join(dir_path, 'conflict_scores_plot')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    with open(os.path.join(data_path, 'module_names.json'), 'r') as f:
        module_names = json.load(f)
        module_names = {
            int(k): v.replace('.', '_') for k, v in module_names.items()
        }
    
    # cf_scores = np.load(os.path.join(data_path, 'cf_scores.npy'))
    cf_scores = np.load(os.path.join(data_path, 'cos_sims.npy'))
    n_layer, n_timestep, n_r, n_task = cf_scores.shape

    timesteps = np.arange(1, n_timestep + 1)
    

    for layer_type in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
        l_idxs = [i for i, name in module_names.items() if name.endswith(layer_type)]
        x_layers = np.arange(1, len(l_idxs) + 1)
        y_scores = np.array([cf_scores[l_idx] for l_idx in l_idxs])
        fig, ax = plt.subplots()
        for t_idx in range(n_task):
            plot_scores_line(
                ax = ax,
                x = x_layers,
                ys = y_scores[:, :, :, t_idx].mean(axis = -1),
                t_idx = t_idx
            )
        ax.set_title(f'{layer_type} Scores')
        fig.tight_layout()
        fig.savefig(os.path.join(res_path, f'{layer_type}_cf_scores.png'), dpi = 400)

    # for layer_type in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
    #     l_idxs = [i for i, name in module_names.items() if name.endswith(layer_type)]
    #     x_layers = np.arange(1, len(l_idxs) + 1)
    #     y_scores = np.array([cf_scores[l_idx] for l_idx in l_idxs])