import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import wandb
from PIL import Image
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec


colors = ["green", "lime", "white", "pink", "deeppink"]  # Corrected color name
CMAP = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion



def draw_heatmap(data, heatmap_path, vmin=-.5, vmax=.5):
    # Create a heatmap using matplotlib and your desired colormap
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    # Create a figure with a gridspec that defines a 1x2 grid
    fig = plt.figure(figsize=(5.2, 5))  # Adjusted figure size for better control
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 0.2], wspace=0.05)  # wspace controls the space between the image and colorbar

    # Add an image plot to the first cell of the gridspec
    ax1 = fig.add_subplot(gs[0])
    img = ax1.imshow(data, cmap=CMAP, vmin=vmin, vmax=vmax)  # Use a default colormap here

    # Add a colorbar to the second cell of the gridspec
    ax2 = fig.add_subplot(gs[1])
    plt.colorbar(img, cax=ax2)
    # plt.tight_layout()
    # Save the heatmap to a file
    plt.savefig(heatmap_path)
    # Close plt figure to free memory
    plt.close()

def draw_curves(train_data, val_data, curve_path):
    plt.figure(figsize=(15, 6))
    x = list(range(len(train_data)))
    plt.subplot(121)
    plt.plot(x,train_data,label='train',color='green')
    plt.title('train loss')
    plt.subplot(122)
    plt.plot(x,val_data,label='val',color='blue')
    plt.title('val loss')
    plt.subplot(133)
    plt.tight_layout()
    # Save the curve to a file
    plt.savefig(curve_path)
    # Close plt figure to free memory
    plt.close('all')


def draw_a_curve(a_list, curve_path):
    plt.figure(figsize=(6, 6))
    x = list(range(len(a_list)))
    plt.plot(x,a_list)
    plt.title('a')
    plt.savefig(curve_path)
    plt.close('all')


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
    W_h_raw = W_h.clone()
    W_h = F.softmax(W_h, dim=1)
    W_h[torch.isnan(W_h)] = 0
    # Convert the torch.Tensor back to numpy.ndarray before returning
    W_h_np = W_h.detach().cpu().numpy()
    W_h_raw = W_h_raw.detach().cpu().numpy()
    return W_h_raw, W_h_np


def visualize_W(W, H, T, n, save_file_path, epoch=-1, phase=1,sub_size=10):
    W_path = f"{save_file_path}/phase{phase}_W_{epoch}.png"
    W_thres = max(W.max(),abs(W.min()))
    draw_heatmap(W, W_path, vmin=-W_thres,vmax=W_thres)
    for h in range(W.shape[1]):
        W_h_raw, W_h = create_matrix_W_h(W[:,h], H, n, T, h)
        W_h_path = f"{save_file_path}/phase{phase}_W_head{h}_{epoch}.png"
        W_h_raw_path = f"{save_file_path}/phase{phase}_W_head{h}_before_{epoch}.png"
        draw_heatmap(W_h, W_h_path, vmin=0, vmax=W_h.max())
        draw_heatmap(W_h, W_h_raw_path, vmin=W_h_raw.min(), vmax=W_h_raw.max())
        W_h_raw, W_h = W_h_raw[:sub_size,:sub_size], W_h[:sub_size,:sub_size]
        W_h_path = f"{save_file_path}/phase{phase}_W_head{h}_after_{sub_size}_{epoch}.png"
        W_h_raw_path = f"{save_file_path}/phase{phase}_W_head{h}_before_{sub_size}_{epoch}.png"
        draw_heatmap(W_h, W_h_path, vmin=0, vmax=W_h.max())
        draw_heatmap(W_h, W_h_raw_path, vmin=W_h_raw.min(), vmax=W_h_raw.max())

def draw_C_alpha_curve(C_alpha_list, x_label, curve_path):
    scale = len(x_label) / 12
    plt.figure(figsize=(15*scale, 6*scale))
    plt.subplot(121)
    C_alpha_list = np.array(C_alpha_list)
    for h in range(len(C_alpha_list[0])):
        plt.plot(list(range(len(C_alpha_list))), C_alpha_list[:, h], label=f'{x_label[h]}')
        plt.legend()
    plt.title('C_alpha')
    plt.subplot(122)
    C_alpha_list_ratio = C_alpha_list**2 / np.linalg.norm(C_alpha_list, axis=-1, keepdims=True)**2
    for h in range(len(C_alpha_list_ratio[0])):
        plt.plot(list(range(len(C_alpha_list_ratio))), C_alpha_list_ratio[:, h], label=f'{x_label[h]}')
        plt.legend()
    plt.title('C_alpha^2 ratio')
    plt.savefig(curve_path)
    plt.close('all')

def draw_bar(data, x_label, bar_path):
    scale = len(x_label) / 12
    _, ax = plt.subplots(figsize=(10*scale, 6*scale))
    # Plot the bar of C_alpha
    ax.bar(x_label, data)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    plt.savefig(bar_path)
    plt.close('all')



def ind2code(ind, S, n):
    """Decode the index into a base-S list of length n"""
    assert ind < S**n
    code = [0 for _ in range(n)]
    for j in range(n):
        code[n - j - 1] = ind % S
        ind //= S
    return code
def code2ind(code, S):
    """Encode the base-S list into an index"""
    n = len(code)
    return sum([code[j] * S**(n-j-1) for j in range(n)])
def next_state(parent, son, S, n):
    """Return the next state of the Markov chain given the current state i and the next symbol j
    
    Args:
    parent: Union[int, list], the parent context
    son: int, the next symbol
    S: int, the number of symbols in the alphabet
    n: int, the length of parent context

    Returns:
    state: Union[int, list], the next state of the Markov chain, the form is determined by the input parent
    """
    if isinstance(parent, int):
        return parent // S + son * S**(n-1)
    else:
        return [son] + parent[0:-1]
    

def get_stationary(pi, S, n, max_iter=100, seed_index=0, output=False):
    """Get the stationary distribution of the Markov chain
    
    Args:
    pi: torch.tensor, the transition matrix
    S: int, the number of symbols in the alphabet
    n: int, the length of the context
    max_iter: int, the maximum number of iterations
    seed_index: int, the index of the seed distribution

    Returns:
    x: torch.tensor, the stationary distribution
    """
    # initialize x
    x = torch.rand(S**n, 1)
    x /= x.sum()

    # get the joint distribution of the parent and the son
    for i in range(max_iter):
        y = x * pi
        # transpose and reshape
        y = y.transpose(0, 1).reshape(-1).reshape(-1, S).sum(axis=-1, keepdims=True)

        # take the TV distance
        d = torch.abs(x - y).sum()
        # print(f'Iteration {i+1}, TV distance: {d}')

        # update x
        x = y
    if output:
        print(f'Final TV distance after {i+1} iterations: {d}')
    return x, d



def plot_hist(y, S, n, hist_path, title='Stationary distribution'):
    """Plot the stationary distribution"""
    plt.bar(range(S**n), y.reshape(-1))
    plt.xlabel('State')
    plt.title(title)
    # add the state's code to the x-axis
    plt.xticks(range(S**n), [f'{i}:{ind2code(i, S, n)}' for i in range(S**n)], rotation=-90)
    plt.savefig(hist_path)

# get the stationary distribution for window size 1
def get_stationary_single_symbol(mu, n):
    S = int(np.exp(np.log(len(mu.squeeze())) / n))
    return mu.reshape(S, -1).sum(axis=-1, keepdims=False)
    
def get_stationary_multi_support(mu_prod_pi, support, S, n):
    """Get the stationary distribution for multiple parents
    
    Args:
    mu: torch.tensor, the stationary distribution
    pi: torch.tensor, the transition matrix
    support: Union[list, int], the list of support of the parents or the binary representation of the support
    
    Returns:
    torch.tensor, the stationary distribution"""

    if tuple(mu_prod_pi.shape) == (S**n, S):
        mu_prod_pi = mu_prod_pi.transpose(0, 1).reshape(-1, S)
    mu_extended = mu_prod_pi.view(tuple(
            [S for _ in range(n+1)]
        ))
    
    if isinstance(support, int):
        support = ind2code(support, 2, n)
    assert isinstance(support, list)
    # include the current state
    support_extended = torch.tensor([1] + support, dtype=torch.bool)
    
    # marginalize out the unsupported parents
    # check if all the parents are supported
    if all(support_extended):
        return mu_extended
    else:
        return torch.sum(mu_extended, dim=tuple([i for i in range(n+1) if not support_extended[i]]), keepdim=True)
        
# # calculate the chi-square mutual information
# def chi_square_mutual_info(parent, pi, mu):
#     S = pi.shape[1]
#     # n = log(S, pi.shape[0])
#     n = int(np.log(pi.shape[0]) / np.log(S))

#     # get the stationary distribution for one symbol
#     mu_single = get_stationary_single_symbol(mu, n)
    
#     if parent == -1:
#         # the average chi-square mutual information
#         return mu.reshape(1, -1) @ ((pi ** 2 / mu_single.reshape(1, -1)).sum(axis=-1, keepdims=True) - 1)
#     elif parent == -2:
#         # the inner product of the squared stationary distribution and the mutual information
#         return mu.reshape(1, -1)**2 @ ((pi ** 2 / mu_single.reshape(1, -1)).sum(axis=-1, keepdims=True) - 1)
#     else: 
#         if isinstance(parent, list):
#             parent = code2ind(parent, S)
#         p = pi[parent]
#         # the chi-square mutual information between mu_single and p
#         return (p ** 2 / mu_single).sum() - 1
#     # take the average over all the parents
    
def chi_square_mutual_info(joint_dist, power=1):
    """Calculate the chi-square mutual information
    
    Args:
    joint_dist: torch.tensor, the joint distribution of the parent and the son
    
    Returns:
    torch.tensor, the chi-square mutual information"""
    marginal_dist = joint_dist.sum(dim=tuple(range(1, joint_dist.ndim)), keepdim=True)
    joint_dist_parent = joint_dist.sum(dim=0, keepdim=True)
    return ((((joint_dist / joint_dist_parent) ** 2 / marginal_dist).sum(dim=0, keepdim=True) - 1) * joint_dist_parent**power).sum()

def chi_square_mutual_info_support(support, mu_prod_pi, S, n, power=1):
    mu_extended = get_stationary_multi_support(mu_prod_pi, support, S, n)
    return chi_square_mutual_info(mu_extended, power)