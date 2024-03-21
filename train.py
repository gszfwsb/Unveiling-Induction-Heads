import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from cat import DisentangledTransformer
from MarkovDataset import MarkovDataset
from tools import *
import argparse
import wandb
import os



def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--n-layers',type=int, default=2)
parser.add_argument('--lr',type=float, default=100)
parser.add_argument('--n-heads',type=list,default=[1,1])
parser.add_argument('--d-out',type=int, default=10)
parser.add_argument('--batch-size',type=int, default=1024)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--ignore-idx',type=int, default=-100)
parser.add_argument('--n-sample',type=int,default=2**27)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--enable-wandb',type=bool,default=False)

args = parser.parse_args()

set_seed(args.seed)
device = args.device
S = args.vocab_size  # Define your vocab size here (size of alphabet)
T = args.seq_length
d_out = args.d_out
n_layers = args.n_layers
n_heads = args.n_heads
n_sample = args.n_sample
bs = args.batch_size
alpha = args.alpha  # Dirichlet parameter
ignore_idx = args.ignore_idx
n_sample = args.n_sample
lr = args.lr

if not args.enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'

# wandb init
wandb.init(project='In-Context-Learning', 
           entity='shaobowang', 
           name=f'Task1_once_bs{bs}_a{alpha}',
           config=vars(args)
        )

# Define the file paths
# root_path = '/cpfs01/user/luanqi.p/wangshaobo/data'
root_path = './data'
dataset_file_path = f'{root_path}/Task1_data_seed{args.seed}_n{n_sample}_alpha{alpha}.pt'  # Specify your path here
save_file_path = f'results/Task1_once/{bs}_{lr}_{alpha}'
makedirs(save_file_path)

# Generate the DisentangledTransformer
model = DisentangledTransformer(S, n_heads, n_layers, T, d_out)
model.to(device)

# Generate the population loss
criterion = population_loss(args.ignore_idx)

# define optimizers and schedulars
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**17)

dataset = MarkovDataset(S, T, alpha, n_sample)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

# visualize before train
visualize(model, save_file_path, 'init')
save(model,save_file_path,'init')

pbar = tqdm(dataloader,ncols=100,mininterval=1)
step = 0
global_step = 0

for x, y in pbar:
    # assert not (torch.isnan(x).any() or torch.isnan(x).any())
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(x) # [bs, T, S]
    logits[:,:T-1,:] = ignore_idx # set to ignore index, only T is valid
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    # Update the learning rate
    scheduler.step()
    pbar.set_description(f'loss:{loss.item():.10f}')
    
    step += 1
    global_step += bs
    
    # Log the loss and heatmap of A1 after every update
    if step % 10 == 0:   
        visualize(model, save_file_path, global_step)
    if step % 100 == 0:   
        save(model,save_file_path,global_step)

# visualize at the end
visualize(model, save_file_path)
save(model,save_file_path)

# Finish the wandb run
wandb.finish()