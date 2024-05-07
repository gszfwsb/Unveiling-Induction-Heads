import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import itertools
# Define the custom color map
H = 3

degrees = torch.tensor(list(itertools.product(range(2), repeat=H+1))) # [max_individual_degree ** num_components, num_components]


x_tilde_pos = torch.where(degrees[:,0] == 0)[0]

print(degrees)

print(x_tilde_pos)