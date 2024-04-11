import os
import random

import numpy as np
import torch
from torch.backends import cudnn
from typing import Callable, List, Literal, Tuple

import matplotlib.pyplot as plt

def set_seed(seed=3407):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.enabled = False  # type: ignore
    return


def makedirs(path):
    os.makedirs(path, exist_ok=True)
    return


def save_dataset(x, y,file_path):
    torch.save((x, y), file_path)
    
    
def load_dataset(file_path):
    return torch.load(file_path)

def draw_heatmap(data, heatmap_path, vmin=-.5, vmax=.5):
    # Create a heatmap using matplotlib and your desired colormap
    # print(data.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.colorbar()
    # Save the heatmap to a file
    plt.savefig(heatmap_path)
    # Close plt figure to free memory
    plt.close()
    return heatmap_path

def visualize(model, save_file_path, epoch=-1):
    for layer in range(len(model.layers)):
        A_l = model.layers[layer].A
        for head in range(len(A_l)):
            A_l_i = A_l[head].cpu().detach().numpy()
            if epoch == -1:
                heatmap_path = f"{save_file_path}/heatmap_A_{layer+1}_{head+1}.png"
            else:
                heatmap_path = f"{save_file_path}/heatmap_A_{layer+1}_{head+1}_{epoch}.png"
            draw_heatmap(A_l_i, heatmap_path, vmin=-.2, vmax=1)
    if epoch == -1:
        heatmap_W = f"{save_file_path}/heatmap_WO.png"
    else:
        heatmap_W = f"{save_file_path}/heatmap_WO_{epoch}.png"
    draw_heatmap(model.Wo.cpu().detach().numpy(), heatmap_W, vmin=-.4,vmax=.4)

def save(model, save_file_path, epoch=-1):
    for layer in range(len(model.layers)):
        A_l = model.layers[layer].A
        for head in range(len(A_l)):
            A_l_i = A_l[head].cpu().detach().numpy()
            if epoch == -1:
                A_path = f"{save_file_path}/A_{layer+1}_{head+1}.pt"
            else:
                A_path = f"{save_file_path}/A_{layer+1}_{head+1}_{epoch}.pt"
            torch.save(A_l_i,A_path)
    if epoch == -1:
        W_path = f"{save_file_path}/WO.pt"
    else:
        W_path = f"{save_file_path}/WO_{epoch}.pt"
    torch.save(model.Wo.cpu().detach().numpy(), W_path)