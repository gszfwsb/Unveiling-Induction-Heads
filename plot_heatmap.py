import os
import random

import numpy as np
import torch
from torch.backends import cudnn
from typing import Callable, List, Literal, Tuple
from tools import draw_heatmap
import matplotlib.pyplot as plt


n_epoch = 100
batch_size = 1024

save_file_path = f'/cpfs01/user/luanqi.p/wangshaobo/ICL/results/Task1/{n_epoch}_{batch_size}_0.1_0.1/'


A_path1 = f"{save_file_path}/A1.pt"
A_path2 = f"{save_file_path}/A2.pt"
W_path = f"{save_file_path}/WO_{n_epoch-1}.pt"






A1 = torch.load(A_path1)[0]
A2 = torch.load(A_path2)[0]
WO = torch.load(W_path)

print(WO)

heatmap_path1 = f"{save_file_path}/heatmap_A1_{n_epoch-1}.png"
heatmap_path2 = f"{save_file_path}/heatmap_A2_{n_epoch-1}.png"
heatmap_W = f"{save_file_path}/heatmap_WO_{n_epoch-1}.png"



draw_heatmap(A1, heatmap_path1)
draw_heatmap(A2, heatmap_path2)
draw_heatmap(WO, heatmap_W, vmin=-0.1,vmax=0.1)