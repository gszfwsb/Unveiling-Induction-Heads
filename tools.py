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


def save_dataset(x, y, file_path):
    torch.save((x, y), file_path)
    
    
def load_dataset(file_path):
    return torch.load(file_path)

def draw_heatmap(data, heatmap_path, vmin=-.5, vmax=.5):
    # Create a heatmap using matplotlib and your desired colormap
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
    if epoch == -1:
        heatmap_path1 = f"{save_file_path}/heatmap_A1.png"
        heatmap_path2 = f"{save_file_path}/heatmap_A2.png"
        heatmap_W = f"{save_file_path}/heatmap_WO.png"
    else:
        heatmap_path1 = f"{save_file_path}/heatmap_A1_{epoch}.png"
        heatmap_path2 = f"{save_file_path}/heatmap_A2_{epoch}.png"
        heatmap_W = f"{save_file_path}/heatmap_WO_{epoch}.png"
    draw_heatmap(model.layers[0].A.cpu().detach().numpy()[0], heatmap_path1,vmin=-.2,vmax=1)
    draw_heatmap(model.layers[1].A.cpu().detach().numpy()[0], heatmap_path2,vmin=-.2,vmax=1)
    draw_heatmap(model.output_layer.weight.data.cpu().detach().numpy(), heatmap_W,vmin=-.4,vmax=.4)

def save(model, save_file_path, epoch=-1):
    if epoch == -1:
        torch.save(model.layers[0].A.data.cpu().detach(),f'{save_file_path}/A1.pt')
        torch.save(model.layers[1].A.data.cpu().detach(),f'{save_file_path}/A2.pt')
        torch.save(model.output_layer.weight.data.cpu().detach().numpy(),f'{save_file_path}/WO.pt')
    else:
        torch.save(model.layers[0].A.data.cpu().detach(),f'{save_file_path}/A1_{epoch}.pt')
        torch.save(model.layers[1].A.data.cpu().detach(),f'{save_file_path}/A2_{epoch}.pt')
        torch.save(model.output_layer.weight.data.cpu().detach().numpy(),f'{save_file_path}/WO_{epoch}.pt')