import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from model_B import TwoLayerTransformer
from dataset import MarkovDataset, NGramDataset
from tools import *
import argparse
import wandb
import os

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
    plt.close()

def draw_a_curve(a_list, save_file_path, phase=1):
    curve_path = f"{save_file_path}/phase{phase}_a_curve.png"
    plt.figure(figsize=(6, 6))
    x = list(range(len(a_list)))
    plt.plot(x,a_list)
    plt.title('a')
    plt.savefig(curve_path)
    plt.close()


def visualize_params(W, C_alpha, save_file_path, epoch=-1, phase=1):
    W_path = f"{save_file_path}/phase{phase}_W_{epoch}.png"
    C_alpha_path = f"{save_file_path}/phase{phase}_C_alpha_{epoch}.png"
    W_thres = max(W.max(),abs(W.min()))
    draw_heatmap(W, W_path, vmin=-W_thres,vmax=W_thres)
    C_alpha_thres = max(C_alpha.max(),abs(C_alpha.min()))
    draw_heatmap(C_alpha, C_alpha_path, vmin=-C_alpha_thres,vmax=C_alpha_thres)



parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=3)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--window-length',type=int, default=8)
parser.add_argument('--n-heads',type=int, default=5)
parser.add_argument('--lr1',type=float, default=0.8)
parser.add_argument('--lr2',type=float, default=0.8)
parser.add_argument('--batch-size',type=int, default=100000)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--n-sample',type=int,default=10000)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--enable-wandb',type=bool,default=False)
parser.add_argument('--dataset',type=str,default='NGram')
parser.add_argument('--optim',type=str,default='adam')
parser.add_argument('--w-plus',type=float,default=0.05)
parser.add_argument('--w-minus',type=float,default=0.01)
parser.add_argument('--a',type=float,default=0.01)
parser.add_argument('--c-alpha',type=float,default=1)
parser.add_argument('--alpha',type=float,default=0.3)
parser.add_argument('--n-epochs',type=int,default=1000)
parser.add_argument('--n-gram',type=int,default=3)


args = parser.parse_args()

set_seed(args.seed)
device = args.device
# model setting
S = args.vocab_size
L = args.seq_length
H = args.n_heads
M = args.window_length
w_plus = args.w_plus
w_minus = args.w_minus
a_init = args.a
# training setting
bs = args.batch_size
n_sample = args.n_sample
lr1 = args.lr1
lr2 = args.lr2
dataset = args.dataset
optim_method = args.optim
n_epochs = args.n_epochs
alpha = args.alpha
ignore_idx = -100 
n = args.n_gram
c_alpha_init = args.c_alpha

if not args.enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'

# wandb init
wandb.init(project='In-Context-Learning-0427model', 
           entity='shaobowang', 
           name=f'Independently_{dataset}_parent{n}_bs{bs}_L{L}_S{S}_{optim_method}_lr1{lr1}_lr2{lr2}',
           config=vars(args)
)

# Define the file paths
root_path = './data'
save_file_path = f'results/{dataset}/Independently_parent{n}_n{n_sample}_L{L}_S{S}_H{H}_M{M}_lr1{lr1}_lr2{lr2}_opt{optim_method}_w+{w_plus}_w-{w_minus}_c_alpha_init{c_alpha_init}_a_init{a_init}_alpha{alpha}_n-epochs{n_epochs}'
makedirs(save_file_path)

# Generate the TwoLayerCausalTransformer
model = TwoLayerTransformer(S, L, H, M, w_plus, w_minus, a_init, c_alpha_init)
model.to(device)

criterion = population_loss(ignore_idx)
 
# define optimizers and schedulars
if optim_method == 'sgd':
    optimizer_1 = optim.SGD([model.a, model.C_alpha_list], lr=lr1)
    optimizer_2 =  optim.SGD([model.W], lr=lr2)
elif optim_method == 'adam':
    optimizer_1 = optim.Adam([model.a, model.C_alpha_list], lr=lr1)
    optimizer_2 = optim.Adam([model.W], lr=lr2)
else:
    raise NotImplementedError(f'{optim_method} not supported!')


data_path = f'./data/{dataset}'
makedirs(data_path)

n_train, n_val = int(n_sample * 0.9), int(n_sample * 0.1)

# Save the datasets

if dataset == 'Markov':
    if os.path.exists(f'{data_path}/{n_sample}_train_set.pt'):
        train_dataset = torch.load(f'{data_path}/{n_sample}_train_set.pt')
        val_dataset = torch.load(f'{data_path}/{n_sample}_val_set.pt')
    else:  
        dataset = MarkovDataset(S, L, alpha, n_sample)
        # Split into train and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        torch.save(train_dataset, f'{data_path}/{n_sample}_train_set.pt')
        torch.save(val_dataset, f'{data_path}/{n_sample}_val_set.pt')
else:
    if os.path.exists(f'{data_path}/{n}_{n_sample}_train_set.pt'):
        train_dataset = torch.load(f'{data_path}/{n}_{n_sample}_train_set.pt')
        val_dataset = torch.load(f'{data_path}/{n}_{n_sample}_val_set.pt')
    else:
        dataset = NGramDataset(S, L, n, alpha, n_sample)
        # Split into train and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        torch.save(train_dataset, f'{data_path}/{n}_{n_sample}_train_set.pt')
        torch.save(val_dataset, f'{data_path}/{n}_{n_sample}_val_set.pt')
  
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)





eval_freq = 100


# test before
C_alpha_list = model.C_alpha_list.clone().cpu().detach().numpy()
W = model.W.clone().cpu().detach().numpy()
visualize_params(W, C_alpha_list, save_file_path, 'init', phase=1)



train_loss_list, val_loss_list, val_acc_list = [], [], []
a_list = []
pbar = tqdm(range(500),ncols=100,mininterval=1)

for epoch in pbar:
    model.train()
    train_loss, eval_loss = 0, 0
    for x, y in train_loader:
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        optimizer_1.zero_grad()
        logits = model(x) # [bs, S]
        loss = criterion(logits, y)
        loss.backward()
        optimizer_1.step()
        # Update the learning rate
        # scheduler.step()
        pbar.set_description(f'Train loss:{loss.item():.10f}')
        wandb.log({'Train loss':loss.item()})    
        
        train_loss += loss.item()
    
    train_loss_list.append(train_loss / n_train)

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            # assert not (torch.isnan(x).any() or torch.isnan(x).any())
            x, y = x.to(device), y.to(device)
            logits = model(x) # [bs, 1, S]
            loss = criterion(logits, y)
            eval_loss += loss.item()
             # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)  # Get the index of the max log-probability
            total_correct += (predicted.squeeze() == y).sum().item()
            # scheduler.step()
            pbar.set_description(f'Val loss:{loss.item():.10f}')
            wandb.log({'Val loss':loss.item()})
        val_acc_list.append(total_correct / n_val)           
        val_loss_list.append(eval_loss / n_val)
        a_list.append(model.a.cpu().detach().numpy())
    
    if epoch % eval_freq == 0:
        W = model.W.clone().cpu().detach().numpy()
        C_alpha_list = model.C_alpha_list.clone().cpu().detach().numpy()
        visualize_params(W, C_alpha_list, save_file_path, epoch, phase=1)
        draw_curves(train_loss_list, val_loss_list, val_acc_list, save_file_path, phase=1)
        draw_a_curve(a_list, save_file_path, phase=1)
        



# train W second
train_loss_list, val_loss_list, val_acc_list = [], [], []
a_list = []

pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)

for epoch in pbar:
    model.train()
    train_loss, eval_loss = 0, 0
    for x, y in train_loader:
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        optimizer_2.zero_grad()
        logits = model(x) # [bs, S]
        loss = criterion(logits, y)
        loss.backward()
        optimizer_2.step()
        # Update the learning rate
        # scheduler.step()
        pbar.set_description(f'Train loss:{loss.item():.10f}')
        wandb.log({'Train loss':loss.item()})    
        train_loss += loss.item()
    train_loss_list.append(train_loss / n_train)

            
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            # assert not (torch.isnan(x).any() or torch.isnan(x).any())
            x, y = x.to(device), y.to(device)
            logits = model(x) # [bs, 1, S]
            loss = criterion(logits, y)
            eval_loss += loss.item()
            # Update the learning rate
             # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)  # Get the index of the max log-probability
            total_correct += (predicted.squeeze() == y).sum().item()
            # scheduler.step()
            pbar.set_description(f'Val loss:{loss.item():.10f}')
            wandb.log({'Val loss':loss.item()})
        val_acc_list.append(total_correct / n_val)           
        val_loss_list.append(eval_loss / n_val)
        a_list.append(model.a.cpu().detach().numpy())
        
    if epoch % eval_freq == 0:
        W = model.W.clone().cpu().detach().numpy()
        C_alpha_list = model.C_alpha_list.clone().cpu().detach().numpy()
        visualize_params(W, C_alpha_list, save_file_path, epoch, phase=2)
        draw_curves(train_loss_list, val_loss_list, val_acc_list, save_file_path, phase=2)
        draw_a_curve(a_list, save_file_path, phase=2)

W = model.W.clone().cpu().detach().numpy()
C_alpha_list = model.C_alpha_list.clone().cpu().detach().numpy()
visualize_params(W, C_alpha_list, save_file_path, 'end', phase=2)
draw_curves(train_loss_list, val_loss_list, val_acc_list, save_file_path, phase=2)
draw_a_curve(a_list, save_file_path, phase=2)

# Finish the wandb run
wandb.finish()