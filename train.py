import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from cat import DisentangledTransformer
from tools import *
import argparse
import wandb
import os
from causal_data import GraphCausalModel
from MarkovDataset_perm import MarkovDataset_perm

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--n-layers',type=int, default=2)
parser.add_argument('--lr',type=float, default=0.5)
parser.add_argument('--n-heads',type=int, nargs='+',default=[2,1])
parser.add_argument('--batch-size',type=int, default=2**13)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--ignore-idx',type=int, default=-100)
parser.add_argument('--n-sample',type=int,default=2**20)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--enable-wandb',type=bool,default=False)
parser.add_argument('--data-type',type=str,default='Markov chain')
parser.add_argument('--optim',type=str,default='sgd')
parser.add_argument('--init',type=str,default='random')


args = parser.parse_args()

set_seed(args.seed)
device = args.device
S = args.vocab_size  # Define your vocab size here (size of alphabet)
T = args.seq_length
n_layers = args.n_layers
n_heads = args.n_heads
n_sample = args.n_sample
bs = args.batch_size
ignore_idx = args.ignore_idx
n_sample = args.n_sample
lr = args.lr
data_type = args.data_type
optim_method = args.optim
init = args.init
if not args.enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'
if data_type == 'Two grams':
    data_number = 2
else:
    data_number = 1
# wandb init
wandb.init(project='In-Context-Learning', 
           entity='shaobowang', 
           name=f'Task{data_number}_random_bs{bs}_T{T}_S{S}_{optim_method}_lr{lr}_{init}',
           config=vars(args)
        )

# Define the file paths
# root_path = '/cpfs01/user/luanqi.p/wangshaobo/data'
root_path = './data'
save_file_path = f'results/Task{data_number}_random_{data_type}_perm/{bs}_{lr}_T{T}_S{S}_opt{optim_method}_{init}'
makedirs(save_file_path)

# Generate the DisentangledTransformer
model = DisentangledTransformer(S, n_heads, n_layers, T, S)
model.to(device)

# set markov converged matrices
if data_type == 'Markov chain':
    if init == 'realistic':
        A1 = torch.zeros((S+T,S+T)).to(device)
        A1[S+1:-1, S:-2] = torch.eye(T-2).to(device)
        A2 = torch.zeros((2*(S+T)), 2*(S+T)).to(device)
        A2[:S, S+T:S+T+S] = torch.eye(S).to(device)
        Wo = torch.zeros((S, 4*(S+T))).to(device)
        Wo[:, 2*(S+T):2*(S+T)+S] = torch.eye(S).to(device)

        model.layers[0].A = torch.nn.Parameter(A1.unsqueeze(0))
        model.layers[1].A = torch.nn.Parameter(A2.unsqueeze(0))
        model.Wo = torch.nn.Parameter(Wo)
    elif init == 'paper':
        nn.init.zeros_(model.layers[0].A[0])
        nn.init.zeros_(model.layers[1].A[0])
        nn.init.zeros_(model.Wo)
    elif init == 'random':
        nn.init.normal_(model.layers[0].A[0], std=0.001)
        nn.init.normal_(model.layers[1].A[0], std=0.001)
        nn.init.normal_(model.Wo, std=0.001)

elif data_type == 'Two grams':
    if init == 'realistic':
        A11 = torch.zeros((S+T,S+T)).to(device)
        A11[S:, S:] = torch.diag(torch.ones(T-1), diagonal=-1).to(device)
        A12 = torch.zeros((S+T,S+T)).to(device)
        A12[S:, S:] = torch.diag(torch.ones(T-2), diagonal=-2).to(device)
        A2 = torch.zeros((3*(S+T)), 3*(S+T)).to(device)
        A2[:S, 1*(S+T):1*(S+T)+S] = torch.eye(S).to(device)
        A2[:S, 2*(S+T):2*(S+T)+S] = torch.eye(S).to(device)
        Wo = torch.zeros((S, 6*(S+T))).to(device)
        Wo[:, 3*(S+T):3*(S+T)+S] = torch.eye(S).to(device)

        model.layers[0].A = torch.nn.Parameter(torch.stack([A11,A12],dim=0))
        model.layers[1].A = torch.nn.Parameter(A2.unsqueeze(0))
        model.Wo = torch.nn.Parameter(Wo)
    elif init == 'paper':
        nn.init.normal_(model.layers[0].A[0],mean=0,std=0.01)
        nn.init.normal_(model.layers[0].A[1],mean=0,std=0.01)
        nn.init.normal_(model.layers[1].A[0],mean=0,std=0.01)
        nn.init.zeros_(model.Wo)

# Generate the population loss
criterion = population_loss(args.ignore_idx)
 
# define optimizers and schedulars
if optim_method == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0, momentum=0)
elif optim_method == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError(f'{optim_method} not supported!')


dataset_file_path = f'{root_path}/Task{data_number}_new_data_seed{args.seed}_n{n_sample}.pt' 

dataset = MarkovDataset_perm(S, T, n_sample)
dataloader = DataLoader(dataset, batch_size=bs)


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
    # scheduler.step()
    pbar.set_description(f'loss:{loss.item():.10f}')
    wandb.log({'loss':loss.item()})    
    step += 1
    global_step += bs

    # Log the loss and heatmap of A1 after every update
    if step % 100 == 0:   
        visualize(model, save_file_path, step+1)
        
    if step % 100 == 0:   
        save(model,save_file_path,step+1)


# visualize at the end
visualize(model, save_file_path)
save(model,save_file_path)

# Finish the wandb run
wandb.finish()