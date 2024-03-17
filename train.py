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

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

def train(model,inputs,targets,criterion, args,save_file_path,epoch):
    # Stage 1: Train only the first layer's A parameter
    optimizer = optim.SGD([model.layers[0].A], lr=args.lr[0])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.time[0])


    for t in range(args.time[0]):
        optimizer.zero_grad()
        logits = model(inputs) # [bs, T, S]
        logits[:,:T-1,:] = ignore_idx # set to ignore index, only T is valid
        loss1 = criterion(logits, targets)
        loss1.backward()
        optimizer.step()
        # Update the learning rate
        scheduler.step()

    # Stage 2: Train only the second layer's A parameter
    optimizer = optim.SGD([model.layers[1].A], lr=args.lr[1])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.time[1])
    
    for t in range(args.time[1]):
        optimizer.zero_grad()
        logits = model(inputs) # [bs, T, S]
        logits[:,:T-1,:] = ignore_idx # set to ignore index, only T is valid
        loss2 = criterion(logits, targets)
        loss2.backward()
        optimizer.step()
        # Update the learning rate
        scheduler.step()
    
    # Output the trained parameters
    # A1 = model.layers[0].A.data
    # A2 = model.layers[1].A.data
    return loss1, loss2

def get_dataset(S, T, alpha, bs):
    x, y, pi, mu_pi = generate_sequence_with_causal_structure(S, T, alpha, bs)
    x = F.one_hot(x, num_classes=S).float()  # (bs, T, S) S word emb indices 
    y = F.one_hot(y, num_classes=S)  # (bs, S) S word emb indices
    return x, y

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--n-layers',type=int, default=2)
parser.add_argument('--time',type=list, default=[100, 100])
parser.add_argument('--lr',type=list, default=[1, 1])
parser.add_argument('--n-heads',type=list,default=[1,1])
parser.add_argument('--d-out',type=int, default=10)
parser.add_argument('--batch-size',type=int, default=1000)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--ignore-idx',type=int, default=-100)
parser.add_argument('--n-epoch',type=int,default=2**17)
parser.add_argument('--n-sample',type=int,default=10000)
parser.add_argument('--device',type=str, default='cuda:1')

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

# wandb init
wandb.init(project='In-Context-Learning', entity='shaobowang', name=f'Task1_epoch{n_epoch}')


# Define the file paths
dataset_file_path = f'/data/wangshaobo/data/Task1_data_seed{args.seed}_n{n_sample}_alpha{alpha}.pt'  # Specify your path here
save_file_path = f'results/Task1/{str(n_epoch).zfill(5)}'
makedirs(save_file_path)

# Generate the DisentangledTransformer
model = DisentangledTransformer(S, n_heads, n_layers, T, d_out)
model.to(device)

# Generate the population loss
criterion = population_loss(args.ignore_idx)

# Generate the sequence with causal structure
# Check if the dataset is already cached
if not os.path.isfile(dataset_file_path):
    # If not, generate and save the dataset
    X, Y = get_dataset(S, T, alpha, n_sample) # [n_sample, T, S], [n_sample, S]
    save_dataset(X, Y, dataset_file_path)
else:
    # If it is, load the dataset from the cache
    X, Y = load_dataset(dataset_file_path)
    print('already cache, load from disk!')

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

pbar = tqdm(list(range(n_epoch)),mininterval=1,ncols=100)
for epoch in pbar:
    loss1_total, loss2_total = 0, 0
    size = 0
    for i, (x,y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        loss1, loss2 = train(model, x, y, criterion, args, save_file_path,epoch)
        loss1_total += loss1.item()
        loss2_total += loss2.item()
        size += x.size(0)
    loss1_total /= size
    loss2_total /= size
    pbar.set_description(f'l1:{loss1_total},l2:{loss2_total}')
    
    wandb.log({
        'Stage 1 Loss': loss1_total,
        'Stage 2 Loss': loss2_total,
    })  
    # Log the loss and heatmap of A1 after every update
    if epoch in [2**i for i in range(20)]:
        heatmap_path = f"{save_file_path}/heatmap_A1_{epoch}.png"
        draw_heatmap(model.layers[0].A.cpu().detach().numpy()[0], heatmap_path)
        wandb.log({
            'A1': wandb.Image(heatmap_path)
        })
        
        heatmap_path = f"{save_file_path}/heatmap_A2_{epoch}.png"
        draw_heatmap(model.layers[1].A.cpu().detach().numpy()[0], heatmap_path)
        wandb.log({
            'A2': wandb.Image(heatmap_path)
        })
   
    if epoch in [2**i for i in range(20)]:
        torch.save(model.layers[0].A.data.cpu().detach(),f'{save_file_path}/A1_{epoch}.pt')
        torch.save(model.layers[1].A.data.cpu().detach(),f'{save_file_path}/A2_{epoch}.pt')
    

torch.save(model.layers[0].A.data.cpu().detach(),f'{save_file_path}/A1.pt')
torch.save(model.layers[1].A.data.cpu().detach(),f'{save_file_path}/A2.pt')

# Finish the wandb run
wandb.finish()