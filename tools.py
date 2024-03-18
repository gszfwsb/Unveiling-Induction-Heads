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

def draw_heatmap(data, heatmap_path, vmin=0, vmax=1):
    # Create a heatmap using matplotlib and your desired colormap
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar()
    # Save the heatmap to a file
    plt.tight_layout()
    plt.savefig(heatmap_path)
    # Close plt figure to free memory
    plt.close()
    return heatmap_path