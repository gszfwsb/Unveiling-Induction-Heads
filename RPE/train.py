import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MarkovDataset, NGramDataset
from tools import makedirs, set_seed
import argparse
import os
import numpy as np
from tools_model_B import *
import wandb
from model_simplified import ToyModel

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=3)
parser.add_argument('--seq-length',type=int, default=100)
parser.add_argument('--n-heads',type=int, default=3)
parser.add_argument('--lr',type=float, default=1e5)
parser.add_argument('--batch-size',type=int, default=100000)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--n-sample',type=int,default=10000)
parser.add_argument('--device',type=str, default='cuda:4')
parser.add_argument('--dataset',type=str,default='NGram')
parser.add_argument('--optim',type=str,default='sgd')
parser.add_argument('--a',type=float,default=0.01)
parser.add_argument('--c-alpha',type=float,default=1)
parser.add_argument('--alpha',type=float,default=0.1)
parser.add_argument('--n-epochs',type=int,default=100)
parser.add_argument('--n-gram',type=int,default=3)
parser.add_argument('--enable-wandb',type=bool,default=True)
parser.add_argument('--learn-a',type=bool,default=False)


args = parser.parse_args()


enable_wandb = args.enable_wandb
if not enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'

set_seed(args.seed)
device = args.device
# model setting
S = args.vocab_size
L = args.seq_length
H = args.n_heads
a_init = args.a
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
c_alpha_init = args.c_alpha
learn_a = args.learn_a

# Define the file paths
method_args = f'Simplified_parent{n}_n{n_sample}_L{L}_S{S}_H{H}_lr{lr}_opt{optim_method}_c_alpha_init{c_alpha_init}_a_init{a_init}_alpha{alpha}_n-epochs{n_epochs}'
method_args += '_learn-a' if learn_a else '_fix-a'
root_path = './data'
save_file_path = f'results/{dataset}/{method_args}'
makedirs(save_file_path)

# Generate the TwoLayerCausalTransformer
if learn_a:
    model = ToyModel(H, S, a_init, c_alpha_init, a='learnable')
else:
    model = ToyModel(H, S, a_init, c_alpha_init, a='fixed')
model.to(device)

criterion = population_loss(ignore_idx)
 
# define optimizers and schedulars
if optim_method == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif optim_method == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    raise NotImplementedError(f'{optim_method} not supported!')


# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//10, gamma=0.5, last_epoch=-1)

data_path = f'./data/{dataset}/vocab{S}-seq{L}-alpha{alpha}' if dataset == 'Markov' else f'./data/{dataset}/vocab{S}-seq{L}-n{n}-alpha{alpha}'
makedirs(data_path)

n_train, n_val = int(n_sample * 0.9), int(n_sample * 0.1)

# Save the datasets
if os.path.exists(f'{data_path}/train_set.pt'):
    train_dataset = torch.load(f'{data_path}/train_set.pt')
    val_dataset = torch.load(f'{data_path}/val_set.pt')
else:
    if dataset == 'Markov':
        dataset = MarkovDataset(S, L, alpha, n_sample)
    else:
        dataset = NGramDataset(S, L, n, alpha, n_sample)
    # Split into train and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    torch.save(train_dataset, f'{data_path}/train_set.pt')
    torch.save(val_dataset, f'{data_path}/val_set.pt')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

eval_freq = min(n_epochs // 10, 100)



# wandb init
wandb.init(project='ICL', 
           entity='Transformer-n-grams', 
           name=f'{method_args}',
           config=vars(args)
)

# test before
C_alpha_list = model.Encoder.C_alpha_list.data.clone().cpu().detach().numpy()[0]
visualize_C_alpha(C_alpha_list, [], [], save_file_path, 'init', phase=1, enable_wandb=enable_wandb)


train_loss_list, val_loss_list, val_acc_list = [], [], []
if learn_a:
    a_list = []
    a_list.append(model.Encoder.a.item())
dominating_C_alpha_index, dominating_C_alpha_value = [], []
pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)

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
        # lr_scheduler.step()

        pbar.set_description(f'Train loss:{loss.item():.10f}')
        
        train_loss += loss.item()
        if epoch % eval_freq == 0:
            C_alpha_grad = model.Encoder.C_alpha_list.grad.data.clone().detach().cpu().numpy()[0]
            C_alpha_grad = np.abs(C_alpha_grad)
            visualize_C_alpha_grad(C_alpha_grad,  save_file_path, epoch, phase=1,enable_wandb=enable_wandb)

    train_loss_list.append(train_loss / n_train)

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            # assert not (torch.isnan(x).any() or torch.isnan(x).any())
            x, y = x.to(device), y.to(device)
            logits = model(x) # [bs, S]
            loss = criterion(logits, y)
            eval_loss += loss.item()
             # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)  # Get the index of the max log-probability
            total_correct += (predicted.squeeze() == y).sum().item()
            # scheduler.step()
            pbar.set_description(f'Val loss:{loss.item():.10f}')
        val_acc_list.append(total_correct / n_val)           
        val_loss_list.append(eval_loss / n_val)
        if learn_a:
            a_list.append(model.Encoder.a.item())
        C_alpha_list = model.Encoder.C_alpha_list.data.cpu().detach().numpy()[0]
        _, max_index, dominance_value = check_dominate_C(C_alpha_list)
        dominating_C_alpha_index.append(max_index)
        dominating_C_alpha_value.append(dominance_value)
    if epoch % eval_freq == 0:
        C_alpha_list = model.Encoder.C_alpha_list.data.clone().cpu().detach().numpy()[0]
        visualize_C_alpha(C_alpha_list,  dominating_C_alpha_value, dominating_C_alpha_index, save_file_path, epoch, phase=1,enable_wandb=enable_wandb)
        draw_curves(train_loss_list, val_loss_list, val_acc_list, save_file_path, phase=1,enable_wandb=enable_wandb)
        if learn_a:
            draw_a_curve(a_list, save_file_path, phase=1,enable_wandb=enable_wandb)
