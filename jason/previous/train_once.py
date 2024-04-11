import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
from cat import DisentangledTransformer
from task_markov_fix import generate_sequence_with_causal_structure

from tools import *
import argparse
import wandb
import os



def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

def get_dataset(S, T, alpha, size):
    x, y, pi, mu_pi = generate_sequence_with_causal_structure(S, T, alpha, size)
    x = F.one_hot(x, num_classes=S).float()  # (bs, T, S) S word emb indices 
    y = F.one_hot(y, num_classes=S)  # (bs, S) S word emb indices
    return x, y, pi, mu_pi

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
parser.add_argument('--n-epoch',type=int,default=500)
parser.add_argument('--n-sample',type=int,default=2**17)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--enable-wandb',type=bool,default=False)

args = parser.parse_args()

# set_seed(args.seed)
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
# root_path = '/cpfs01/user/luanqi.p/wangshaobo/data'
root_path = '/data/wangshaobo/data'
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
# if True:
    print('generate and save the dataset')
    # If not, generate and save the dataset
    X, Y, pi, mu_pi = get_dataset(S, T, alpha, n_sample) # [n_sample, T, S], [n_sample, S]
    save_dataset(X, Y, pi, mu_pi, dataset_file_path)
else:
    # If it is, load the dataset from the cache
    X, Y, pi, mu_pi = load_dataset(dataset_file_path)
    print('already cache, load from disk!')

# print(pi, mu_pi)
# draw_heatmap(pi, f"{save_file_path}/pi.png",vmin=0,vmax=S)
# print(mu_pi)
# draw_heatmap(mu_pi, f"{save_file_path}/mu_pi.png",vmin=0,vmax=S)


# define optimizers and schedulars
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0, momentum=0)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**17//bs)

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)


# visualize before train
visualize(model, save_file_path, 'init')
save(model,save_file_path,'init')


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
    if epoch % 25 == 0:   
        visualize(model, save_file_path, epoch)
    if epoch % 100 == 0:   
        save(model,save_file_path,epoch)

# visualize at the end
visualize(model, save_file_path)
save(model,save_file_path)

# Finish the wandb run
wandb.finish()