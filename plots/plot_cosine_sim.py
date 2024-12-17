import numpy as np
import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dir_path = './output/bigbench-mindllm1b3-loramix48'
    # dir_path = './output/completed/Panacea_train 71 random'
    data_path = os.path.join(dir_path, 'conflict_scores')
    res_path = os.path.join(dir_path, 'conflict_scores_plot')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    
    conflict_cos_sims = np.load(os.path.join(data_path, 'conflict_cos_sims.npy'))
    
    if os.path.exists(os.path.join(data_path, 'module_names.json')):
        with open(os.path.join(data_path, 'module_names.json'), 'r') as f:
            module_names = json.load(f)
            module_names = {
                int(k): v.replace('.', '_') for k, v in module_names.items()
            }
    else:
        module_names = None