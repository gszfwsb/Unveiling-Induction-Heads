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

def get_dataset(S, T, alpha, bs):
    x, y, pi, mu_pi = generate_sequence_with_causal_structure(S, T, alpha, bs)
    x = F.one_hot(x, num_classes=S).float()  # (bs, T, S) S word emb indices 
    y = F.one_hot(y, num_classes=S)  # (bs, S) S word emb indices
    return x, y

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--n-layers',type=int, default=2)
parser.add_argument('--lr',type=float, default=100)
parser.add_argument('--n-heads',type=list,default=[1,1])
parser.add_argument('--d-out',type=int, default=10)
parser.add_argument('--batch-size',type=int, default=8192)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--ignore-idx',type=int, default=-100)
parser.add_argument('--n-epoch',type=int,default=500)
parser.add_argument('--n-sample',type=int,default=100000)
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
n_epoch = args.n_epoch
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
           name=f'Task1_once_epoch{n_epoch}_bs{bs}_a{alpha}',
           config=vars(args)
        )


# Define the file paths
root_path = '/cpfs01/user/luanqi.p/wangshaobo/data'
# root_path = '/data/wangshaobo/data'
dataset_file_path = f'{root_path}/Task1_data_seed{args.seed}_n{n_sample}_alpha{alpha}.pt'  # Specify your path here
save_file_path = f'results/Task1_once/{bs}_{lr}_{alpha}'
makedirs(save_file_path)

# Generate the DisentangledTransformer
model = DisentangledTransformer(S, n_heads, n_layers, T, d_out)
model.to(device)

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
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**17)

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)


# visualize before train
visualize(model, save_file_path, 0)
save(model,save_file_path,0)


pbar = tqdm(list(range(n_epoch)),mininterval=1,ncols=100)
for epoch in pbar:
    loss_total = 0.
    size = 0
    for i, (x,y) in enumerate(dataloader):
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
        visualize(model, save_file_path, epoch)
    if epoch % 50 == 0:   
        save(model,save_file_path,epoch)

# visualize at the end
visualize(model, save_file_path)
save(model,save_file_path)

# Finish the wandb run
wandb.finish()