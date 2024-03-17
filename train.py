import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
from cat import DisentangledTransformer
from task import generate_sequence_with_causal_structure

from tools import *
import argparse
import wandb
import os



def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

def train(model,inputs,targets,criterion, args, optimizers, schedulers):
    n_stage = len(optimizers)
    # Stage i: Train only the ith layer's A parameter
    for stage in range(n_stage):
        for t in range(args.time[stage]):
            optimizers[stage].zero_grad()
            logits = model(inputs) # [bs, T, S]
            logits[:,:T-1,:] = ignore_idx # set to ignore index, only T is valid
            loss = criterion(logits, targets)
            loss.backward()
            optimizers[stage].step()
            # Update the learning rate
            schedulers[stage].step()

    return loss

def get_dataset(S, T, alpha, bs):
    x, y, pi, mu_pi = generate_sequence_with_causal_structure(S, T, alpha, bs)
    x = F.one_hot(x, num_classes=S).float()  # (bs, T, S) S word emb indices 
    y = F.one_hot(y, num_classes=S)  # (bs, S) S word emb indices
    return x, y

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--n-layers',type=int, default=2)
parser.add_argument('--time',type=list, default=[10, 10])
parser.add_argument('--lr',type=list, default=[1,1])
parser.add_argument('--n-heads',type=list,default=[1,1])
parser.add_argument('--d-out',type=int, default=10)
parser.add_argument('--batch-size',type=int, default=8192)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--beta',type=float, default=0.1)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--ignore-idx',type=int, default=-100)
parser.add_argument('--n-epoch',type=int,default=500)
parser.add_argument('--n-sample',type=int,default=100000)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--enable-wandb',type=bool,default=True)

args = parser.parse_args()

set_seed(args.seed)
device = args.device
S = args.vocab_size  # Define your vocab size here (size of alphabet)
T = args.seq_length
d_out = args.d_out
n_layers = args.n_layers
n_heads = args.n_heads
n_epoch = args.n_epoch
n_sample = args.n_sample
bs = args.batch_size
alpha = args.alpha  # Dirichlet parameter
beta =args.beta
ignore_idx = args.ignore_idx
n_sample = args.n_sample

if not args.enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'

# wandb init
wandb.init(project='In-Context-Learning', 
           entity='shaobowang', 
           name=f'Task1_epoch{n_epoch}',
           config=vars(args)
        )


# Define the file paths
root_path = '/data/wangshaobo/data'
dataset_file_path = f'{root_path}/Task1_data_seed{args.seed}_n{n_sample}_alpha{alpha}.pt'  # Specify your path here
save_file_path = f'results/Task1/{n_epoch}_{bs}_{alpha}_{beta}'
makedirs(save_file_path)

# Generate the DisentangledTransformer
model = DisentangledTransformer(S, n_heads, n_layers, T, d_out)
model.to(device)

# reinit the params
with torch.no_grad():
    model.layers[0].A.fill_(0.0)
    A2 = torch.zeros(2*(S+T), 2*(S+T))
    A2[:S, S+T:S+T+S] = torch.eye(S)
    model.layers[1].A[0] = torch.nn.Parameter(A2)

# Generate the population loss
criterion = population_loss(args.ignore_idx)

# Generate the sequence with causal structure
# Check if the dataset is already cached
if not os.path.isfile(dataset_file_path):
    print('generate and save the dataset')
    # If not, generate and save the dataset
    X, Y = get_dataset(S, T, alpha, n_sample) # [n_sample, T, S], [n_sample, S]
    save_dataset(X, Y, dataset_file_path)
else:
    # If it is, load the dataset from the cache
    X, Y = load_dataset(dataset_file_path)
    print('already cache, load from disk!')

# define optimizers and schedulars
optimizers = []
schedulers = []
for stage in range(n_layers):
    optimizer = optim.SGD([model.layers[stage].A], lr=args.lr[stage])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**17)
    optimizers.append(optimizer)
    schedulers.append(scheduler)

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

pbar = tqdm(list(range(n_epoch)),mininterval=1,ncols=100)
for epoch in pbar:
    loss_total = 0.
    size = 0
    for i, (x,y) in enumerate(dataloader):
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        loss = train(model, x, y, criterion, args, optimizers, schedulers)
        loss_total += loss.item()
        size += x.size(0)
    loss_total /= size
    pbar.set_description(f'loss:{loss_total:.10f}')
    

    wandb.log({
        'Loss': loss_total,
        # 'A1': wandb.Image(heatmap_path1),
        # 'A2': wandb.Image(heatmap_path2)
    })
    
    # Log the loss and heatmap of A1 after every update
    if epoch % 10 == 0:   
        heatmap_path1 = f"{save_file_path}/heatmap_A1_{epoch}.png"
        heatmap_path2 = f"{save_file_path}/heatmap_A2_{epoch}.png"
        draw_heatmap(model.layers[0].A.cpu().detach().numpy()[0], heatmap_path1)
        draw_heatmap(model.layers[1].A.cpu().detach().numpy()[0], heatmap_path2)
    if epoch % 50 == 0:   
        torch.save(model.layers[0].A.data.cpu().detach(),f'{save_file_path}/A1_{epoch}.pt')
        torch.save(model.layers[1].A.data.cpu().detach(),f'{save_file_path}/A2_{epoch}.pt')
    

torch.save(model.layers[0].A.data.cpu().detach(),f'{save_file_path}/A1.pt')
torch.save(model.layers[1].A.data.cpu().detach(),f'{save_file_path}/A2.pt')

# Finish the wandb run
wandb.finish()