import numpy as np
import os
import json
import matplotlib.pyplot as plt

def plot_lines(
    axe,
    x, ys,
    labels,
    colors = None
):
    if colors is None:
        for y, label in zip(ys, labels):
            axe.plot(x, y, label = label, alpha = 0.75)
    else:
        for y, label, color in zip(ys, labels, colors):
            for (s, e, c) in color:
                axe.plot(x[s: e], y[s: e], label = label, color = c, alpha = 0.5)
    return axe

if __name__ == '__main__':

    dir_path = './output/Panacea_train 71-128-diagzero'
    # dir_path = './output/completed/Panacea_train 71 random'
    data_path = os.path.join(dir_path, 'conflict_scores')
    res_path = os.path.join(dir_path, 'conflict_scores_plot')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    diags = np.load(os.path.join(data_path, 'diags.npy'))
    n_svd_lora = diags.shape[0]
    n_timestep = diags.shape[1]
    n_rs = diags.shape[2]

    timesteps = np.arange(1, n_timestep + 1)[np.newaxis, :, np.newaxis].repeat(n_svd_lora, axis = 0).repeat(n_rs, axis = 2)
    conflict_cos_sims = np.load(os.path.join(data_path, 'conflict_cos_sims.npy'))
    conflict_cos_sims = np.cumsum(conflict_cos_sims, axis = 1) / timesteps
    grad_conflict_scores = np.load(os.path.join(data_path, 'grad_conflict_scores.npy'))
    grad_conflict_scores = np.cumsum(grad_conflict_scores, axis = 1) / timesteps
    if os.path.exists(os.path.join(data_path, 'split_flags.npy')):
        split_flags = np.load(os.path.join(data_path, 'split_flags.npy'))
    else:
        split_flags = None
    if os.path.exists(os.path.join(data_path, 'module_names.json')):
        with open(os.path.join(data_path, 'module_names.json'), 'r') as f:
            module_names = json.load(f)
            module_names = {
                int(k): v.replace('.', '_') for k, v in module_names.items()
            }
    else:
        module_names = None
    
    # barplot
    plt.figure(figsize = (10, 5))
    n_bin = 10
    bar_width = 0.35
    x_ranges = np.linspace(-1.0, 1.0, num = n_bin + 1)
    tick_label = [f'({x_ranges[idx + 1]:.1f}, {x_ranges[idx]:.1f})' for idx in reversed(range(len(x_ranges) - 1))]
    x_idx = np.arange(n_bin)
    if split_flags is not None:
        split_distribution = []
        unsplit_distribution = []
        split_cos_sims = conflict_cos_sims[:, -1, :][split_flags[:, -1, :] == 1]
        unsplit_cos_sims = conflict_cos_sims[:, -1, :][split_flags[:, -1, :] == 0]
        for idx in reversed(range(len(x_ranges) - 1)):
            split_distribution.append(
                np.sum((split_cos_sims <= x_ranges[idx + 1]) & (split_cos_sims > x_ranges[idx]))
            )
            unsplit_distribution.append(
                np.sum((unsplit_cos_sims <= x_ranges[idx + 1]) & (unsplit_cos_sims > x_ranges[idx]))
            )
        plt.bar(
            x_idx, unsplit_distribution,
            # align = 'center',
            color = 'C0',
            tick_label = tick_label,
            label = 'unsplit'
        )
        plt.bar(
            x_idx, split_distribution,
            # align = 'center',
            bottom = unsplit_distribution,
            color = 'C1',
            label = 'split'
        )
        plt.legend()
    else:
        distribution = []
        cos_sims = conflict_cos_sims[:, -1, :].reshape(-1)
        for idx in reversed(range(len(x_ranges) - 1)):
            distribution.append(
                np.sum((cos_sims <= x_ranges[idx + 1]) & (cos_sims > x_ranges[idx]))
            )
        plt.bar(
            x_idx, distribution,
            # align = 'center',
            color = 'C0',
            tick_label = tick_label,
            # label = 'unsplit'
        )
            
    plt.tight_layout()
    save_path = os.path.join(res_path, f'conflict_cos_sims_distribution')
    plt.savefig(save_path, dpi = 400)
    plt.close()

    # lineplot
    for idx in range(n_svd_lora):
        if split_flags is None:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for subgragh_idx, (name, tensor, axe) in enumerate(zip(
            ['diags', 'conflict_cos_sims', 'grad_conflict_scores'],
            [diags, conflict_cos_sims, grad_conflict_scores],
            axes if split_flags is None else axes[0]
        )):
            axe = plot_lines(
                axe = axe,
                x = np.arange(n_timestep),
                ys = [tensor[idx, :, r_idx] for r_idx in range(tensor.shape[2])],
                labels = [f'r_idx={r_idx}' for r_idx in range(tensor.shape[2])]
            )
            axe.set_xlabel('timestep')
            axe.set_ylabel(name)
            
            if name == 'diags':
                axe.legend()
        # plt.legend()

        if split_flags is not None:
            split_flag = split_flags[idx]
            colors = []
            for r_idx in range(n_rs):
                color = []
                flag = split_flag[: , r_idx]
                start = 0
                change_points = np.where(np.diff(flag) != 0)[0] + 1
                change_points = np.concatenate(([0], change_points, [n_timestep]))
                for i in range(len(change_points) - 1):
                    start = change_points[i]
                    end = change_points[i + 1]
                    color.append((start if start == 0 else start - 1, end, f'C{int(flag[start])}'))
                colors.append(color)
                            
            for subgragh_idx, (name, tensor, axe) in enumerate(zip(
                ['diags', 'conflict_cos_sims', 'grad_conflict_scores'],
                [diags, conflict_cos_sims, grad_conflict_scores],
                axes[1]
            )):
                axe = plot_lines(
                    axe = axe,
                    x = np.arange(n_timestep),
                    ys = [tensor[idx, :, r_idx] for r_idx in range(tensor.shape[2])],
                    labels = [f'r_idx={r_idx}' for r_idx in range(tensor.shape[2])],
                    colors = colors
                )
                axe.set_xlabel('timestep')
                axe.set_ylabel(name)
            # plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(res_path, f'{module_names[idx]}_scores')
        plt.savefig(save_path, dpi = 400)
        plt.close()