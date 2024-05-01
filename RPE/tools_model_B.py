import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import wandb


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

def draw_curves(train_data, val_data, val_acc, save_file_path, phase=1):
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


def draw_a_curve(a_list, save_file_path, phase=1):
    curve_path = f"{save_file_path}/phase{phase}_a_curve.png"
    plt.figure(figsize=(6, 6))
    x = list(range(len(a_list)))
    plt.plot(x,a_list)
    plt.title('a')
    plt.savefig(curve_path)
    plt.close('all')


def visualize_W(W, save_file_path, epoch=-1, phase=1):
    W_path = f"{save_file_path}/phase{phase}_W_{epoch}.png"
    W_thres = max(W.max(),abs(W.min()))
    draw_heatmap(W, W_path, vmin=-W_thres,vmax=W_thres)

def visualize_C_alpha(C_alpha, dominating_C_alpha_value, dominating_C_alpha_index, save_file_path, epoch=-1, phase=1):
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

    if len(dominating_C_alpha_index) > 0:
        plt.figure(figsize=(15, 6))
        plt.subplot(121)
        x = list(range(len(dominating_C_alpha_value)))
        plt.plot(x,dominating_C_alpha_value)
        plt.title('Dominating C_alpha Value')
        plt.subplot(122)
        plt.plot(x,dominating_C_alpha_index)
        plt.title('Dominating C_alpha Index')
        plt.tight_layout()
        plt.savefig(curve_path)
        plt.close('all')


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

