import torch
import numpy as np
print(torch.cuda.device_count())

# points = np.linspace(0, 40242, 3, endpoint=True)
# points = np.round(points, 0).astype(int)
# print(points)
# n_epoch = 11
# x = torch.range(0, 1, step = 1 / (n_epoch - 1))
# print(x)
# x = torch.linspace(0, 1, steps = n_epoch)
# print(x)

x = torch.tensor([1,6,3,2,5,7,0,4])

y = x.argsort()
print(y)
print(x.index_select(0, y))