import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
sys.path.append(os.path.abspath('../'))
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MarkovDataset, NGramDataset
from tools import makedirs, set_seed
import argparse
import numpy as np
from RPE.utils import *
import wandb
from RPE.model import TwoLayerTransformer


parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=3)
parser.add_argument('--seq-length',type=int, default=100)
parser.add_argument('--n-heads',type=int, default=3)
parser.add_argument('--lr1',type=float, default=1)
parser.add_argument('--lr2',type=float, default=1)
parser.add_argument('--lr3',type=float, default=1)
parser.add_argument('--batch-size',type=int, default=100000)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--n-sample',type=int,default=10000)
parser.add_argument('--device',type=str, default='cuda:3')
parser.add_argument('--dataset',type=str,default='NGram')
parser.add_argument('--w-plus',type=float,default=1)
parser.add_argument('--w-minus',type=float,default=0.01)
parser.add_argument('--optim',type=str,default='sgd')
parser.add_argument('--a',type=float,default=0.01)
parser.add_argument('--c-alpha',type=float,default=1)
parser.add_argument('--alpha',type=float,default=0.1)
parser.add_argument('--n-epochs',type=int,default=10000)
parser.add_argument('--n-gram',type=int,default=3)
parser.add_argument('--low-degree',type=int,default=-1)
parser.add_argument('--train-cmd', action='append', type=str)
parser.add_argument('--enable-wandb',type=bool,default=False)


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
lr1 = args.lr1
lr2= args.lr2
lr3=args.lr3
dataset = args.dataset
optim_method = args.optim
n_epochs = args.n_epochs
alpha = args.alpha
ignore_idx = -100 
n = args.n_gram
c_alpha_init = args.c_alpha
w_plus = args.w_plus
w_minus = args.w_minus
low_degree = args.low_degree
train_cmd = args.train_cmd
# Define the file paths
method_args = f'Formal_{train_cmd}_parent{n-1}_n{n_sample}_L{L}_S{S}_H{H}_lr1{lr1}_lr2{lr2}_lr3{lr3}_opt{optim_method}_w+{w_plus}_w-{w_minus}_D{low_degree}_c_alpha_init{c_alpha_init}_a_init{a_init}_alpha{alpha}_n-epochs{n_epochs}'
root_path = './data'
save_file_path = f'results/{dataset}/{method_args}'
makedirs(save_file_path)

# Generate the TwoLayerCausalTransformer
if low_degree != -1:
    model = TwoLayerTransformer(S, L, H, w_plus, w_minus, a_init, c_alpha_init, n-1, low_degree)
else:
    model = TwoLayerTransformer(S, L, H, w_plus, w_minus, a_init, c_alpha_init, n-1)
model.to(device)




criterion = population_loss(ignore_idx)



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

eval_freq = min(n_epochs//10, 500)


degrees = model.layer2.degrees.data.clone().cpu().detach().numpy()
remain_pos = np.where(degrees[:, 0]==0)[0]
degrees = degrees[remain_pos] # only for those exclude X_tilde
alphas =  [''.join(row.astype(int).astype(str)) for row in degrees]
print(alphas)

def plot_begin(model, remain_pos, alphas, H, L, n, save_file_path):
    C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
    C_alpha_list = C_alpha_list[remain_pos]
    bar_path = f"{save_file_path}/phase1_C_alpha_init.png"
    draw_bar(C_alpha_list, alphas, bar_path)
    W = model.layer1.W.clone().cpu().detach().numpy()
    visualize_W(W, H, L, n-1, save_file_path, 'init', phase=1)


def generate_train_flags(train_cmd):
    train_flags = []
    for param in train_cmd:
        flags = [False, False, False]
        for char in param:
            if char == 'C':
                flags[0] = True
            elif char == 'W':
                flags[1] = True
            elif char == 'a':
                flags[2] = True
        if flags == [False, False, False]:
            continue
        train_flags.append(flags)
    return train_flags

def plot_end(model, remain_pos, alphas, W, H, L, n, save_file_path, phase=2):
    C_alpha_list = model.layer2.C_alpha_list.clone().cpu().detach().numpy()[0]
    C_alpha_list = C_alpha_list[remain_pos]
    W = model.layer1.W.clone().cpu().detach().numpy()
    visualize_W(W, H, L, n-1, save_file_path, 'end', phase=phase)
    bar_path = f"{save_file_path}/phase{phase}_C_alpha_end.png"
    draw_bar(C_alpha_list, alphas, bar_path)
    curve_path = f"{save_file_path}/phase{phase}_curve.png"
    draw_curves(train_loss_list, val_loss_list, curve_path)
    curve_path = f"{save_file_path}/phase{phase}_a_curve.png"
    draw_a_curve(a_list, curve_path)
    curve_path = f"{save_file_path}/phase{phase}_C_alpha_curve.png"
    draw_C_alpha_curve(C_list, alphas, curve_path)

def train(model, 
        train_loader, 
        val_loader, 
        n_train, 
        n_val,
        remain_pos,
        criterion, 
        alphas,
        lr1,
        lr2,
        lr3,
        phase,
        save_file_path,
        train_C=False, 
        train_W=False, 
        train_a=False,
        eval_freq=500,
        ):
    train_loss_list, val_loss_list = [], []
    pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)

    param_groups = []
    if train_C:
        param_groups.append({'params': model.layer2.C_alpha_list, 'lr': lr1})
        C_list = []
        C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
        C_alpha_list = C_alpha_list[remain_pos]
        C_list.append(C_alpha_list)
    if train_W:
        param_groups.append({'params': model.layer1.W, 'lr': lr2})
    if train_a:
        param_groups.append({'params': model.layer2.a, 'lr': lr3})
        a_list = []
        a_list.append(model.layer2.a.item())
    optimizer = optim.SGD(param_groups, momentum=0, weight_decay=0)
    for epoch in pbar:
        model.train()
        train_loss, eval_loss = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x) # [bs, S]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Train loss:{loss.item():.10f}')
            train_loss += loss.item()
            if epoch % eval_freq == 0:
                if train_C:
                    C_alpha_grad = model.layer2.C_alpha_list.grad.data.clone().detach().cpu().numpy()[0]
                    C_alpha_grad = C_alpha_grad[remain_pos]
                    bar_path = f"{save_file_path}/phase{phase}_grad_C_alpha_{epoch}.png"
                    draw_bar(C_alpha_grad, alphas, bar_path)
        train_loss_list.append(train_loss / n_train)
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x) # [bs, S]
                loss = criterion(logits, y)
                eval_loss += loss.item()
                pbar.set_description(f'Val loss:{loss.item():.10f}')
            val_loss_list.append(eval_loss / n_val)
            if train_a:
                a_list.append(model.layer2.a.item())
            if train_C:
                C_alpha_list = model.layer2.C_alpha_list.data.cpu().detach().numpy()[0]
                C_alpha_list = C_alpha_list[remain_pos]
                C_list.append(C_alpha_list)
        if epoch % eval_freq == 0:
            curve_path = f"{save_file_path}/phase{phase}_curve.png"
            draw_curves(train_loss_list, val_loss_list, curve_path)
            if train_C:
                C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
                C_alpha_list = C_alpha_list[remain_pos]
                curve_path = f"{save_file_path}/phase{phase}_C_alpha_curve.png"
                draw_C_alpha_curve(C_list, alphas, curve_path)
                bar_path = f"{save_file_path}/phase{phase}_C_alpha_{epoch}.png"
                draw_bar(C_alpha_list, alphas, bar_path)
            if train_W:
                W = model.layer1.W.clone().cpu().detach().numpy()
                visualize_W(W, H, L, n-1, save_file_path, epoch, phase=phase)
            if train_a:
                curve_path = f"{save_file_path}/phase{phase}_a_curve.png"
                draw_a_curve(a_list, curve_path)




plot_begin(model, remain_pos, alphas, H, L, n, save_file_path)


train_flags = generate_train_flags(train_cmd)

print(train_flags)

for phase, train_flag in enumerate(train_flags):
    train_C, train_W, train_a = train_flag[0], train_flag[1], train_flag[2]
    train(model, 
        train_loader, 
        val_loader, 
        n_train,
        n_val, 
        remain_pos,
        criterion, 
        alphas,
        lr1,
        lr2,
        lr3,
        phase+1,
        save_file_path,
        train_C=train_C, 
        train_W=train_W, 
        train_a=train_a,
        eval_freq=500,
    )



plot_end(model, remain_pos, alphas, H, L, n, save_file_path, phase=3)