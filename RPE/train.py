import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from model import TwoLayerTransformer
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

def draw_curves(train_data, val_data, val_acc, save_file_path, data_type='train'):
    curve_path = f"{save_file_path}/curve.png"
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

def visualize_params(W, A, save_file_path, epoch=-1):
    W_path = f"{save_file_path}/W_{epoch}.png"
    A_path = f"{save_file_path}/A_{epoch}.png"
    W_thres = max(W.max(),abs(W.min()))
    draw_heatmap(W, W_path, vmin=-W_thres,vmax=W_thres)
    A_thres = max(W.max(),abs(W.min()))
    draw_heatmap(A, A_path, vmin=-A_thres,vmax=A_thres)



parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--window-length',type=int, default=8)
parser.add_argument('--n-heads',type=int, default=5)
parser.add_argument('--lr',type=float, default=0.8)
parser.add_argument('--batch-size',type=int, default=100000)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--n-sample',type=int,default=100000)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--enable-wandb',type=bool,default=False)
parser.add_argument('--dataset',type=str,default='NGram')
parser.add_argument('--optim',type=str,default='adam')
parser.add_argument('--w-plus',type=float,default=0.05)
parser.add_argument('--w-minus',type=float,default=0.01)
parser.add_argument('--a',type=float,default=0.01)
parser.add_argument('--alpha',type=float,default=0.3)
parser.add_argument('--n-epochs',type=int,default=1000)
parser.add_argument('--n-gram',type=int,default=2)


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
a = args.a
# training setting
bs = args.batch_size
n_sample = args.n_sample
lr = args.lr
dataset = args.dataset
optim_method = args.optim
n_epochs = args.n_epochs
alpha = args.alpha
ignore_idx = -100 
n = args.n_gram

if not args.enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'

# wandb init
wandb.init(project='In-Context-Learning-0409model', 
           entity='shaobowang', 
           name=f'{dataset}_bs{bs}_L{L}_S{S}_{optim_method}_lr{lr}',
           config=vars(args)
)

# Define the file paths
root_path = './data'
save_file_path = f'results/{dataset}/n{n_sample}_L{L}_S{S}_H{H}_M{M}_lr{lr}_opt{optim_method}_w+{w_plus}_w-{w_minus}_a{a}-alpha{alpha}'
makedirs(save_file_path)

# Generate the TwoLayerCausalTransformer
model = TwoLayerTransformer(S, L, H, M, w_plus, w_minus, a)
model.to(device)

criterion = population_loss(ignore_idx)
 
# define optimizers and schedulars
if optim_method == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0, momentum=0)
elif optim_method == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError(f'{optim_method} not supported!')


data_path = './data'
makedirs(data_path)

n_train, n_val = int(n_sample * 0.9), int(n_sample * 0.1)

# Save the datasets
if os.path.exists(f'{data_path}/{dataset}/{n_sample}_train_set.pt'):
    train_dataset = torch.load(f'{data_path}/{dataset}/{n_sample}_train_set.pt')
    val_dataset = torch.load(f'{data_path}/{dataset}/{n_sample}_val_set.pt')
else:  
    if dataset == 'Markov':
        dataset = MarkovDataset(S, L, alpha, n_sample)
    else:
        dataset = NGramDataset(S, L, n, alpha, n_sample)
    # Split into train and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    torch.save(train_dataset, f'{data_path}/{dataset}/{n_sample}_train_set.pt')
    torch.save(val_dataset, f'{data_path}/{dataset}/{n_sample}_val_set.pt')
  
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)



pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)

eval_freq = 100

train_loss_list, val_loss_list, val_acc_list = [], [], []

# test before
A = model.A.clone().cpu().detach().numpy()
W = model.W.clone().cpu().detach().numpy()
visualize_params(W, A, save_file_path, 'init')

for epoch in pbar:
    model.train()
    train_loss, eval_loss = 0, 0
    for x, y in train_loader:
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x) # [bs, S]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
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

    
    if epoch % eval_freq == 0:
        A = model.A.clone().cpu().detach().numpy()
        W = model.W.clone().cpu().detach().numpy()
        visualize_params(W, A, save_file_path, epoch)
        draw_curves(train_loss_list, val_loss_list, val_acc_list, save_file_path)


visualize_params(W, A, save_file_path, 'end')
draw_curves(train_loss_list, val_loss_list, val_acc_list, save_file_path)

# Finish the wandb run
wandb.finish()