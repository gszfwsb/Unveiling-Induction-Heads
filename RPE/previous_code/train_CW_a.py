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


parser = argparse.ArgumentParser('train 2-layer disentangled Transformer with a fixed a first')
parser.add_argument('--vocab-size',type=int,default=3)
parser.add_argument('--seq-length',type=int, default=100)
parser.add_argument('--n-heads',type=int, default=3)
parser.add_argument('--lr1',type=float, default=1e5)
parser.add_argument('--lr2',type=float, default=1e5)
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
# Define the file paths
method_args = f'Formal_fix_a_first_parent{n-1}_n{n_sample}_L{L}_S{S}_H{H}_lr1{lr1}_lr2{lr2}_opt{optim_method}_w+{w_plus}_w-{w_minus}_D{low_degree}_c_alpha_init{c_alpha_init}_a_init{a_init}_alpha{alpha}_n-epochs{n_epochs}'
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

eval_freq = min(n_epochs//10, 500)


 
# define optimizers and schedulars
if optim_method == 'sgd':
    optimizer1 = optim.SGD([model.layer1.W,model.layer2.C_alpha_list], lr=lr1, momentum=0, weight_decay=0)
    optimizer2 = optim.SGD([model.layer2.a], lr=lr2,  momentum=0, weight_decay=0)
elif optim_method == 'adam':
    optimizer1 = optim.Adam([model.layer1.W,model.layer2.C_alpha_list], lr=lr1)
    optimizer2 = optim.Adam([model.layer2.a], lr=lr2)
else:
    raise NotImplementedError(f'{optim_method} not supported!')

# wandb init
wandb.init(project='ICL', 
           entity='Transformer-n-grams', 
           name=f'{method_args}',
           config=vars(args)
)



degrees = model.layer2.degrees.data.clone().cpu().detach().numpy()
remain_pos = np.where(degrees[:, 0]==0)[0]
degrees = degrees[remain_pos] # only for those exclude X_tilde
alphas =  [''.join(row.astype(int).astype(str)) for row in degrees]

C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
C_alpha_list = C_alpha_list[remain_pos]

bar_path = f"{save_file_path}/phase1_C_alpha_init.png"
draw_bar(C_alpha_list, alphas, bar_path)
W = model.layer1.W.clone().cpu().detach().numpy()
visualize_W(W, H, L, n-1, save_file_path, 'init', phase=1)

# train W and C_alpha first
train_loss_list, val_loss_list, val_acc_list = [], [], []
C_list = []
C_list.append(C_alpha_list)
pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)

for epoch in pbar:
    model.train()
    train_loss, eval_loss = 0, 0
    for x, y in train_loader:
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        optimizer1.zero_grad()
        logits = model(x) # [bs, S]
        loss = criterion(logits, y)
        loss.backward()
        optimizer1.step()

        pbar.set_description(f'Train loss:{loss.item():.10f}')
        
        train_loss += loss.item()
        if epoch % eval_freq == 0:
            C_alpha_grad = model.layer2.C_alpha_list.grad.data.clone().detach().cpu().numpy()[0]
            C_alpha_grad = C_alpha_grad[remain_pos]
            bar_path = f"{save_file_path}/phase1_grad_C_alpha_{epoch}.png"
            draw_bar(C_alpha_grad, alphas, bar_path)
            # print(model.layer1.W.grad.data.clone().detach().cpu().numpy())
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
        C_alpha_list = model.layer2.C_alpha_list.data.cpu().detach().numpy()[0]
        C_alpha_list = C_alpha_list[remain_pos]
        C_list.append(C_alpha_list)
    if epoch % eval_freq == 0:
        curve_path = f"{save_file_path}/phase1_curve.png"
        draw_curves(train_loss_list, val_loss_list, val_acc_list, curve_path)
        W = model.layer1.W.clone().cpu().detach().numpy()
        visualize_W(W, H, L, n-1, save_file_path, epoch, phase=1)
        C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
        C_alpha_list = C_alpha_list[remain_pos]
        curve_path = f"{save_file_path}/phase1_C_alpha_curve.png"
        draw_C_alpha_curve(C_list, alphas, curve_path)
        bar_path = f"{save_file_path}/phase1_C_alpha_{epoch}.png"
        draw_bar(C_alpha_list, alphas, bar_path)
# train a second
train_loss_list, val_loss_list, val_acc_list = [], [], []
a_list = []
a_list.append(model.layer2.a.item())
pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)


for epoch in pbar:
    model.train()
    train_loss, eval_loss = 0, 0
    for x, y in train_loader:
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        optimizer2.zero_grad()
        logits = model(x) # [bs, S]
        loss = criterion(logits, y)
        loss.backward()
        optimizer2.step()
        pbar.set_description(f'Train loss:{loss.item():.10f}')
        train_loss += loss.item()
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
        a_list.append(model.layer2.a.item())
    if epoch % eval_freq == 0:
        curve_path = f"{save_file_path}/phase2_curve.png"
        draw_curves(train_loss_list, val_loss_list, val_acc_list, curve_path)
        curve_path = f"{save_file_path}/phase2_a_curve.png"
        draw_a_curve(a_list, curve_path)



W = model.layer1.W.clone().cpu().detach().numpy()
C_alpha_list = model.layer2.C_alpha_list.clone().cpu().detach().numpy()[0]
C_alpha_list = C_alpha_list[remain_pos]
visualize_W(W, H, L, n-1, save_file_path, 'end', phase=2)
bar_path = f"{save_file_path}/phase2_C_alpha_end.png"
draw_bar(C_alpha_list, alphas, bar_path)
curve_path = f"{save_file_path}/phase2_curve.png"
draw_curves(train_loss_list, val_loss_list, val_acc_list, curve_path)
curve_path = f"{save_file_path}/phase2_a_curve.png"
draw_a_curve(a_list, curve_path)
curve_path = f"{save_file_path}/phase2_C_alpha_curve.png"
draw_C_alpha_curve(C_list, alphas, curve_path)
