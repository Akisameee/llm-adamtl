import matplotlib.pyplot as plt  
import numpy as np
import os
import pandas as pd  

def find_eval_res(res_dir):

    for name in os.listdir(res_dir):
        if name.endswith('eval_result.csv'):
            df = pd.read_csv(os.path.join(res_dir, name))  
            scores = df['Rouge-L Score'].to_list()
            task_names = df['Task Name'].to_list()
            return task_names, scores
        
    for ckpt_path in os.listdir(res_dir):
        ckpt_path = os.path.join(res_dir, ckpt_path)
        if os.path.isdir(ckpt_path):
            res = find_eval_res(ckpt_path)
            if res is not None:
                return res
        
    return None

def plot_all_res(
    tasks,
    scores,
    legend_names
):
    x = np.arange(len(tasks))

    plt.figure(figsize = (10, 7))

    colors = plt.colormaps.get_cmap('tab10')

    for i in range(len(tasks)):

        idx_scores = [(idx, score[i]) for idx, score in enumerate(scores)]
        idx_scores.sort(key = lambda x: x[1], reverse = True)
        
        for idx, score in idx_scores:
            plt.bar(x[i], score, color = colors(idx))

    plt.xticks(x, tasks, rotation = 60, ha = 'right')
    plt.legend(legend_names)

    plt.ylabel('Scores')
    plt.title('Multitask Results')
    plt.tight_layout()

    plt.savefig('./plots/res/all_eval_res.png')

def plot_base_comp(
    base_scores,
    base_name,
    scores,
    name
):
    x = np.arange(len(tasks))

    plt.figure(figsize = (10, 7))

    colors = plt.colormaps.get_cmap('tab10')

    for i in range(len(tasks)):
        
        score = scores[i] - base_scores[i]
        plt.bar(
            x[i],
            score,
            color = colors(0 if score >=0 else 1)
        )

    plt.xticks(x, tasks, rotation = 60, ha = 'right')

    plt.ylabel('Scores')
    plt.title(f'{name} - {base_name}')
    plt.tight_layout()

    plt.savefig(f'./plots/res/base_comp_{name}.png')

base_res_dir = './output/bigbench-mindllm1b3'
res_dirs = [
    './output/bigbench-mindllm1b3-mix48',
    './output/bigbench-mindllm1b3-lora48',
    './output/bigbench-mindllm1b3-loramix48',
    './output/bigbench-mindllm1b3-mgda48',
    './output/bigbench-mindllm1b3-pcgrad48',
    './output/bigbench-mindllm1b3-cagrad48',
    './output/bigbench-mindllm1b3-famo48',
    './output/bigbench-mindllm1b3-ada48-1',
    './output/bigbench-mindllm1b3-ada48-2'
]

tasks, base_scores = find_eval_res(base_res_dir)
scores = []
for res_dir in res_dirs:
    task_names, score = find_eval_res(res_dir)
    assert tasks == task_names
    print(f'Result Name: {os.path.split(res_dir)[-1]}, Avg Score: {sum(score) / len(score)}')
    scores.append(score)

base_name = os.path.split(base_res_dir)[-1]
legend_names = []
for res_dir in res_dirs:
    legend_names.append(os.path.split(res_dir)[-1])

plot_all_res(
    tasks = tasks,
    scores = scores,
    legend_names = legend_names
)

for name, score in zip(legend_names, scores):
    plot_base_comp(
        base_scores = base_scores,
        base_name = base_name,
        scores = score,
        name = name
    )