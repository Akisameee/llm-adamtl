import sys
sys.path.insert(0, '/home/smliu/RLHF')
import os
import re
from logger import plot_pareto_fronts_2d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


if __name__ == '__main__':
    
    output_dir = './output/'
    dir_paths = [
        os.path.join(output_dir, dir_path) for dir_path in os.listdir(output_dir) \
        if os.path.exists(os.path.join(output_dir, dir_path, 'Panacea_train.log')) and '2024-' not in dir_path
    ]

    helpful_pattern = r'helpful: ([\-]*\d+\.\d+)'
    harmless_pattern = r'harmless: ([\-]*\d+\.\d+)'
    pref_vec_pattern = r"pref_vec: \['([\-]*\d+\.\d+)', '([\-]*\d+\.\d+)'\]"

    all_eval_res = {}
    for dir_path in dir_paths:
        with open(os.path.join(dir_path, 'Panacea_train.log'), 'r') as f:
            eval_ress = []
            is_eval = False
            eval_res = []
            for line in f.readlines():
                if line.startswith('Evaluation step'):
                    if not is_eval:
                        is_eval = True
                    else:
                        is_eval = False
                        eval_ress.append(eval_res)
                        eval_res = []
                if not is_eval:
                    continue
                else:
                    if re.search(helpful_pattern, line):
                        helpful_match = re.search(helpful_pattern, line)
                        harmless_match = re.search(harmless_pattern, line)
                        pref_vec_match = re.search(pref_vec_pattern, line)
                        helpful = float(helpful_match.group(1))
                        harmless = float(harmless_match.group(1))
                        pref_vec = [float(pref_vec_match.group(1)), float(pref_vec_match.group(2))]
                        eval_res.append((helpful, harmless, pref_vec))

        all_eval_res[dir_path] = eval_ress

    print(all_eval_res)
    axes = None
    axes_names = ('helpful', 'harmless')
    legend_handles = []
    for idx, (dir_path, eval_ress) in enumerate(all_eval_res.items()):
        eval_res = eval_ress[-1]
        x = [res[0] for res in eval_res]
        y = [res[1] for res in eval_res]
        pref_vecs = [res[2] for res in eval_res]
        lines = [[i, i + 1] for i in range(len(x) - 1)]
        axes = plot_pareto_fronts_2d(
            x = x,
            y = y,
            lines = lines,
            axes_names = axes_names,
            p_alphas = [1] * len(x),
            l_alphas = [1] * len(lines),
            scaling = 1,
            color = f'C{idx}',
            arrows = pref_vecs,
            prev_axes = axes
        )
        legend_handle = mlines.Line2D(
            [], [],
            color = f'C{idx}',
            marker = '_',
            markersize = 15,
            label = os.path.split(dir_path)[-1]
        )
        legend_handles.append(legend_handle)

    plt.legend(handles = legend_handles)
    save_path = os.path.join(dir_paths[0], 'all_eval_res')
    print(f'saved result at {dir_paths[0]}')
    plt.savefig(save_path, dpi = 400)
    plt.close()
                        