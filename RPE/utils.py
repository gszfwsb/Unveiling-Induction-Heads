import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import wandb
from PIL import Image
import torch.nn.functional as F

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

def normalize_data(data):
    # Normalize data to the range [-1, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

def draw_heatmap(data, heatmap_path, vmin=-.5, vmax=.5, normalize=False):
    # Create a heatmap using matplotlib and your desired colormap
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    if normalize:
        data = normalize_data(data)
    # print(data.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='inferno', vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.colorbar()
    # Save the heatmap to a file
    plt.savefig(heatmap_path)
    # Close plt figure to free memory
    plt.close()

def draw_curves(train_data, val_data, val_acc, save_file_path, phase=1,enable_wandb=False):
    curve_path = f"{save_file_path}/phase{phase}_curve.png"
    plt.figure(figsize=(15, 6))
    x = list(range(len(train_data)))
    plt.subplot(131)
    plt.plot(x,train_data,label='train',color='green')
    plt.title('train loss')
    plt.subplot(132)
    plt.plot(x,val_data,label='val',color='blue')
    plt.title('val loss')
    plt.subplot(133)
    plt.plot(x,val_acc,label='acc',color='orange')
    plt.title('val acc')
    plt.tight_layout()
    # Save the curve to a file
    plt.savefig(curve_path)
    # Close plt figure to free memory
    plt.close('all')
    if enable_wandb:
        image = Image.open(curve_path)
        wandb.log({"Learning Dynamics": wandb.Image(image)})

def draw_a_curve(a_list, save_file_path, phase=1,enable_wandb=False):
    curve_path = f"{save_file_path}/phase{phase}_a_curve.png"
    plt.figure(figsize=(6, 6))
    x = list(range(len(a_list)))
    plt.plot(x,a_list)
    plt.title('a')
    plt.savefig(curve_path)
    plt.close('all')
    if enable_wandb:
        image = Image.open(curve_path)
        wandb.log({"a": wandb.Image(image)})

def create_matrix_W_h(W, H, n, T, h):
    """
    Create and return the softmax-normalized matrix W_h based on the input numpy matrix W_np.
    
    Args:
    W_np (numpy.ndarray): Input numpy array of shape [S, T], where each column represents values for the diagonals of W_h.
    T (int): The size of the desired output matrix W_h, which will be (T+1) x (T+1).
    
    Returns:
    numpy.ndarray: Output matrix W_h after applying softmax, of shape [T+1, T+1].
    """
    # Convert numpy matrix to torch.Tensor
    W = torch.tensor(W, dtype=torch.float32)
    
    # Create an initial matrix of shape [T+1, T+1] filled with negative infinity
    W_h = torch.full((T+1, T+1), float('-inf'), dtype=torch.float32, device=W.device)
    # Fill the diagonals
    for j in range(H):
        torch.diagonal(W_h, -j-1-h).fill_(W[j+h])
    # Apply softmax along each row
    W_h = F.softmax(W_h, dim=1)
    W_h[torch.isnan(W_h)] = 0
    # Convert the torch.Tensor back to numpy.ndarray before returning
    W_h_np = W_h.detach().cpu().numpy()
    return W_h_np


def visualize_W(W, H, T, n, save_file_path, epoch=-1, phase=1,enable_wandb=False):
    W_path = f"{save_file_path}/phase{phase}_W_{epoch}.png"
    W_thres = max(W.max(),abs(W.min()))
    draw_heatmap(W, W_path, vmin=-W_thres,vmax=W_thres)
    if enable_wandb:
        image = Image.open(W_path)
        wandb.log({"W": wandb.Image(image)})
    for h in range(W.shape[1]):
        W_h = create_matrix_W_h(W[:,h], H, n, T, h)
        W_h_path = f"{save_file_path}/phase{phase}_W_head{h}_{epoch}.png"
        draw_heatmap(W_h, W_h_path, vmin=0, vmax=W_h.max())
        if enable_wandb:
            image = Image.open(W_h_path)
            wandb.log({f"W_{h}": wandb.Image(image)})


def draw_C_alpha_curve(C_alpha_list, save_file_path, phase=1,enable_wandb=False):
    curve_path = f"{save_file_path}/phase{phase}_C_alpha_curve.png"
    plt.figure(figsize=(6, 6))
    C_alpha_list = np.array(C_alpha_list)
    x = list(range(len(C_alpha_list)))
    for h in range(len(C_alpha_list[0])):
        plt.plot(x, C_alpha_list[:, h], label=f'C_{h}')
        plt.legend()
    plt.title('C_alpha_curve')
    plt.savefig(curve_path)
    plt.close('all')
    if enable_wandb:
        image = Image.open(curve_path)
        wandb.log({"C_alpha_curve": wandb.Image(image)})

def visualize_C_alpha_grad(grad, save_file_path, epoch=-1, phase=1,enable_wandb=False):
    C_alpha_grad_path = f"{save_file_path}/phase{phase}_C_alpha_grad_{epoch}.png"
    _, ax = plt.subplots(figsize=(10, 6))
    # Plot the bar of C_alpha
    ax.bar(np.arange(len(grad)), grad)
    ax.set_title('C_alpha Grad Bar')
    ax.set_xlabel('Index')
    ax.set_ylabel('Abs C_alpha Grad')
    plt.savefig(C_alpha_grad_path)
    plt.close('all')

    if enable_wandb:
        image = Image.open(C_alpha_grad_path)
        wandb.log({"C_alpha Grad Bar": wandb.Image(image)})

def visualize_C_alpha(C_alpha, dominating_C_alpha_value, dominating_C_alpha_index, save_file_path, epoch=-1, phase=1,enable_wandb=False):
    C_alpha_path = f"{save_file_path}/phase{phase}_C_alpha_{epoch}.png"
    curve_path = f"{save_file_path}/phase{phase}_C_dominance_curve.png"
    C_alpha_sqaure = C_alpha ** 2
    _, ax = plt.subplots(figsize=(10, 6))
    # Plot the bar of C_alpha
    ax.bar(np.arange(len(C_alpha_sqaure)), C_alpha_sqaure)
    ax.set_title('C_alpha Bar')
    ax.set_xlabel('Index')
    ax.set_ylabel('C_alpha')
    plt.savefig(C_alpha_path)
    plt.close('all')

    if enable_wandb:
        image = Image.open(C_alpha_path)
        wandb.log({"C_alpha Bar": wandb.Image(image)})

    if len(dominating_C_alpha_index) > 0:
        plt.figure(figsize=(15, 6))
        plt.subplot(121)
        x = list(range(len(dominating_C_alpha_value)))
        plt.plot(x,dominating_C_alpha_value)
        plt.title('Dominating C_alpha Square Ratio')
        plt.subplot(122)
        plt.plot(x,dominating_C_alpha_index)
        plt.title('Dominating C_alpha Square Index')
        plt.tight_layout()
        plt.savefig(curve_path)
        plt.close('all')

        if enable_wandb:
            image = Image.open(curve_path)
            wandb.log({"Dominating C_alpha Index": wandb.Image(image)})

def check_dominate_C(C_alpha_list):
    C_alpha_list = torch.from_numpy(C_alpha_list)
    # Calculate the sum of the squares of C_alpha_list
    sum_of_squares = torch.sum(C_alpha_list ** 2)
    # Check for dominance for each c_alpha
    dominance = (C_alpha_list ** 2) / sum_of_squares
    # Find which c_alpha is dominating
    max_index = torch.argmax(dominance)
    # Check if c_alpha dominates or not
    is_dominating = dominance[max_index] >= 0.4  # Threshold of 0.5 is arbitrary, adjust as needed
    # Optionally, check if it grows exponentially faster
    # This would require historical data to compare the growth rate
    dominance_value = dominance[max_index]
    return is_dominating, max_index, dominance_value

