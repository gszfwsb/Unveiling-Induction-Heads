import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import os.path as osp
sys.path.append(osp.abspath('../'))
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MarkovDataset, NGramDataset
from tools import makedirs, set_seed
import argparse
import numpy as np
from RPE.utils import *
from RPE.model import TwoLayerTransformer



def plot_begin(model, remain_pos, alphas, H, L, n, save_file_path):
    C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
    C_alpha_list = C_alpha_list[remain_pos]
    bar_path = osp.join(save_file_path, 'C_alpha', 'value', 'init.svg')
    draw_bar(C_alpha_list, alphas, bar_path)
    W = model.layer1.W.clone().cpu().detach().numpy()
    W_path = osp.join(save_file_path)
    visualize_W(W, H, L, n-1, W_path, 'init')
    
    
def plot_end(model, train_loss_list, val_loss_list, remain_pos, alphas, H, L, n, save_file_path):
    C_alpha_list = model.layer2.C_alpha_list.clone().cpu().detach().numpy()[0]
    C_alpha_list = C_alpha_list[remain_pos]
    W = model.layer1.W.clone().cpu().detach().numpy()
    visualize_W(W, H, L, n-1, save_file_path, 'end')
    bar_path = osp.join(save_file_path,  'C_alpha', 'value', 'end.svg')
    draw_bar(C_alpha_list, alphas, bar_path)
    curve_path = osp.join(save_file_path, 'curve.svg')
    draw_curves(train_loss_list, val_loss_list, curve_path)


def train(model, 
        train_loader, 
        val_loader, 
        n_train, 
        n_val,
        n_epochs,
        device,
        H,
        L,
        n,
        criterion, 
        lr,
        save_file_path,
        eval_freq=500,
        ):
    
    ########################### begin ###########################
    degrees = model.layer2.degrees.data.clone().cpu().detach().numpy()
    remain_pos = np.where(degrees[:, 0]==0)[0]
    degrees = degrees[remain_pos] # only for those exclude X_tilde
    alphas =  [''.join(row.astype(int).astype(str)) for row in degrees]
    C_list = []
    C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
    C_alpha_list = C_alpha_list[remain_pos]
    C_list.append(C_alpha_list)
    a_list = []
    a_list.append(model.layer2.a.item())
    ############################################################
    
    train_loss_list, val_loss_list = [], []
    pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    

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
                C_alpha_grad = model.layer2.C_alpha_list.grad.data.clone().detach().cpu().numpy()[0]
                C_alpha_grad = C_alpha_grad[remain_pos]
                bar_path = osp.join(save_file_path, 'C_alpha', 'grad', f'{epoch}.svg')
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
            a_list.append(model.layer2.a.item())
            C_alpha_list = model.layer2.C_alpha_list.data.cpu().detach().numpy()[0]
            C_alpha_list = C_alpha_list[remain_pos]
            C_list.append(C_alpha_list)
            
        if epoch % eval_freq == 0:
            curve_path = osp.join(save_file_path, 'curve.svg')
            draw_curves(train_loss_list, val_loss_list, curve_path)
            C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
            C_alpha_list = C_alpha_list[remain_pos]
            curve_path = osp.join(save_file_path, 'C_alpha', 'curve.svg')
            draw_C_alpha_curve(C_list, alphas, curve_path)
            bar_path = osp.join(save_file_path, 'C_alpha', 'value', f'{epoch}.svg')
            draw_bar(C_alpha_list, alphas, bar_path)
            W = model.layer1.W.clone().cpu().detach().numpy()
            W_path = osp.join(save_file_path)
            visualize_W(W, H, L, n-1, W_path, epoch)
            curve_path = osp.join(save_file_path, 'a', 'curve.svg')
            draw_a_curve(a_list, curve_path)
    return train_loss_list, val_loss_list

def main():
    parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
    parser.add_argument('--vocab-size',type=int,default=3)
    parser.add_argument('--seq-length',type=int, default=100)
    parser.add_argument('--n-heads',type=int, default=3)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--batch-size',type=int, default=100000)
    parser.add_argument('--seed',type=int, default=2024)
    parser.add_argument('--n-sample',type=int,default=10000)
    parser.add_argument('--device',type=str, default='cuda:3')
    parser.add_argument('--dataset',type=str,default='NGram')
    parser.add_argument('--w-plus',type=float,default=1)
    parser.add_argument('--w-minus',type=float,default=0.01)
    parser.add_argument('--optim',type=str,default='sgd')
    parser.add_argument('--a',type=float,default=0.01)
    parser.add_argument('--c-alpha',type=float,default=0.01)
    parser.add_argument('--alpha',type=float,default=0.1)
    parser.add_argument('--n-epochs',type=int,default=10000)
    parser.add_argument('--n-gram',type=int,default=3)
    parser.add_argument('--low-degree',type=int,default=-1)


    args = parser.parse_args()

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
    w_plus = args.w_plus
    w_minus = args.w_minus
    low_degree = args.low_degree
    # d_mlp = args.d_mlp
    # Define the file paths
    method_args = f'Full_parent{n-1}_n{n_sample}_L{L}_S{S}_H{H}_lr{lr}_opt{optim_method}_w+{w_plus}_w-{w_minus}__alpha{alpha}_n-epochs{n_epochs}'
    root_path = './data'
    save_file_path = osp.join(f'./results_paper', dataset, method_args)
    os.makedirs(save_file_path, exist_ok=True)
    # Generate the TwoLayerCausalTransformer
    if low_degree != -1:
        model = TwoLayerTransformer(S, L, H, w_plus, w_minus, a_init, c_alpha_init, n-1, low_degree, proj_init=0.001)
    else:
        model = TwoLayerTransformer(S, L, H, w_plus, w_minus, a_init, c_alpha_init, n-1, proj_init=0.001)
    model.to(device)


    criterion = population_loss(ignore_idx)
    if dataset == 'Markov':
        data_path = osp.join(root_path, dataset, f'vocab{S}_seq{L}_alpha{alpha}_{n_sample}')
    else:
        data_path = osp.join(root_path, dataset, f'vocab{S}_seq{L}_n{n}_alpha{alpha}_{n_sample}')
    os.makedirs(data_path, exist_ok=True)
    n_train, n_val = int(n_sample * 0.9), int(n_sample * 0.1)

    # Save the datasets
    train_set_path = osp.join(data_path, 'train_set.pt')
    val_set_path = osp.join(data_path, 'val_set.pt')
    if osp.exists(train_set_path) and osp.exists(val_set_path):
        train_dataset = torch.load(train_set_path)
        val_dataset = torch.load(val_set_path)
    else:
        if dataset == 'Markov':
            dataset = MarkovDataset(S, L, alpha, n_sample)
        else:
            dataset = NGramDataset(S, L, n, alpha, n_sample)
        # Split into train and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        print(f"save dataset to {data_path}")
        torch.save(train_dataset, train_set_path)
        torch.save(val_dataset, val_set_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

    eval_freq = min(n_epochs//10, 500)



    degrees = model.layer2.degrees.data.clone().cpu().detach().numpy()
    remain_pos = np.where(degrees[:, 0]==0)[0]
    degrees = degrees[remain_pos] # only for those exclude X_tilde
    alphas =  [''.join(row.astype(int).astype(str)) for row in degrees]

    plot_begin(model, remain_pos, alphas, H, L, n, save_file_path)


    train_loss_list, val_loss_list  = train(model, 
                                            train_loader, 
                                            val_loader, 
                                            n_train, 
                                            n_val,
                                            n_epochs,
                                            device,
                                            H,
                                            L,
                                            n,
                                            criterion, 
                                            lr,
                                            save_file_path,
                                            eval_freq=eval_freq,
                                        )

    plot_end(model, train_loss_list, val_loss_list, remain_pos, alphas, H, L, n, save_file_path)

if __name__ == "__main__":
    main()