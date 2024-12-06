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

res_dirs = [
    './output/bigbench-mindllm1b3',
    './output/bigbench-mix48',
    './output/bigbench-loramix48'
]

scores = []
tasks = None
for res_dir in res_dirs:
    task_names, score = find_eval_res(res_dir)
    if tasks is None:
        tasks = task_names
    else:
        assert tasks == task_names
    print(f'Result Name: {os.path.split(res_dir)[-1]}, Avg Score: {sum(score) / len(score)}')
    scores.append(score)

x = np.arange(len(tasks))

plt.figure(figsize = (10, 7))

colors = plt.colormaps.get_cmap('tab10')

for i in range(len(tasks)):

    idx_scores = [(idx, score[i]) for idx, score in enumerate(scores)]
    idx_scores.sort(key = lambda x: x[1], reverse = True)
    
    for idx, score in idx_scores:
        plt.bar(x[i], score, color = colors(idx))

plt.xticks(x, tasks, rotation = 60, ha = 'right')

legend_names = []
for res_dir in res_dirs:
    legend_names.append(os.path.split(res_dir)[-1])
plt.legend(legend_names)

plt.ylabel('Scores')
plt.title('Multitask Results')
plt.tight_layout()

plt.savefig('./plots/res.png')