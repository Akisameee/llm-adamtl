import torch
import numpy as np
print(torch.cuda.device_count())

points = np.linspace(0, 40242, 3, endpoint=True)
points = np.round(points, 0).astype(int)
print(points)