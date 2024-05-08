import numpy as np
import random

from configs.pefts import SVD_Lora_Config
from modules.pefts import SVD_Lora_Linear

def get_random_split(n_svd_lora: int, n_split: int, max_r: int):
    
    assert max_r * n_svd_lora >= n_split

    random_split = [0] * n_svd_lora
    for _ in range(n_split):
        idx = random.randint(0, n_svd_lora - 1)
        while random_split[idx] >= max_r:
            if idx < n_svd_lora: idx += 1
            else: idx = 0
        random_split[idx] += 1
    random.shuffle(random_split)

    return random_split

def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == "norm":
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == "loss":
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == "loss+":
        for t in grads:
            gn[t] = losses[t] * np.sqrt(
                np.sum([gr.pow(2).sum().data[0] for gr in grads[t]])
            )
    elif normalization_type == "none":
        for t in grads:
            gn[t] = 1.0
    else:
        print("ERROR: Invalid Normalization Type")
    return gn