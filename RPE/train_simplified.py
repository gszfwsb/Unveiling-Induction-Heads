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
from RPE.model import SimplifiedTwoLayerTransformer
from collections import Counter


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



def save_params(file_path, params_dict):
    np.savez(file_path, **params_dict)


def combine_and_save_results(file_path, results):
    combined_results = {}
    for result in results:
        for key, value in result.items():
            if key not in combined_results:
                combined_results[key] = []
            combined_results[key].extend(value if isinstance(value, list) else [value])
    
    save_params(file_path, combined_results)


def train(model, 
        train_loader, 
        val_loader, 
        n_train, 
        n_val,
        n_epochs,
        device,
        remain_pos,
        criterion, 
        lr,
        phase,
        train_C=False, 
        train_W=False, 
        train_a=False,
        ):
    train_loss_list, val_loss_list = [], []
    pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)

    param_groups = []
    phase_results = {}

    if train_C:
        param_groups.append({'params': model.layer2.C_alpha_list})
        C_list = []
        C_alpha_list = model.layer2.C_alpha_list.data.clone().cpu().detach().numpy()[0]
        C_alpha_list = C_alpha_list[remain_pos]
        C_list.append(C_alpha_list)
        phase_results['C_list'] = C_list

    if train_W:
        param_groups.append({'params': model.layer1.W})
        W_before = model.layer1.W.clone().cpu().detach().numpy()
        phase_results['W_before'] = W_before

    if train_a:
        param_groups.append({'params': model.layer2.a})
        a_list = []
        a_list.append(model.layer2.a.item())
        phase_results['a_list'] = a_list

    optimizer = optim.SGD(param_groups, lr=lr, momentum=0, weight_decay=0)

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

    phase_results[f'train_loss_list_{phase}'] = train_loss_list
    phase_results[f'val_loss_list_{phase}'] = val_loss_list

    if train_W:
        W_after = model.layer1.W.clone().cpu().detach().numpy()
        phase_results['W_after'] = W_after

    return phase_results

def main():
    parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
    parser.add_argument('--vocab-size',type=int,default=3)
    parser.add_argument('--seq-length',type=int, default=100)
    parser.add_argument('--n-heads',type=int, default=3)
    parser.add_argument('--lr', action='append', type=float)
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
    parser.add_argument('--n-epochs',type=int,action='append')
    parser.add_argument('--n-gram',type=int,default=3)
    parser.add_argument('--low-degree',type=int,default=3)
    parser.add_argument('--train-cmd', action='append', type=str)


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
    lr_list = args.lr
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
    assert len(lr_list) == len(train_cmd)
    assert len(train_cmd) == len(n_epochs)
    char_count = Counter(''.join(train_cmd))
    assert char_count['C'] == 1 and char_count['W'] == 1 and char_count['a'] == 1
    cmd_args = '_'.join([''.join(sorted(s)) for s in train_cmd])
    lr_args = '_'.join(str(_) for _ in lr_list)
    method_args = f'{cmd_args}_parent{n-1}_n{n_sample}_L{L}_S{S}_H{H}_{lr_args}_opt{optim_method}_w+{w_plus}_w-{w_minus}_D{low_degree}_c_alpha_init{c_alpha_init}_a_init{a_init}_alpha{alpha}_n-epochs{n_epochs}'
    root_path = './data'
    save_file_path = osp.join(f'./results_paper', dataset, method_args)
    os.makedirs(save_file_path, exist_ok=True)
    # Generate the TwoLayerCausalTransformer
    if low_degree != -1:
        model = SimplifiedTwoLayerTransformer(S, L, H, w_plus, w_minus, a_init, c_alpha_init, n-1, low_degree)
    else:
        model = SimplifiedTwoLayerTransformer(S, L, H, w_plus, w_minus, a_init, c_alpha_init, n-1)
    model.to(device)


    criterion = population_loss(ignore_idx)
    if dataset == 'Markov':
        data_path = osp.join(root_path, dataset, f'vocab{S}_seq{L}_alpha{alpha}')
    else:
        data_path = osp.join(root_path, dataset, f'vocab{S}_seq{L}_n{n}_alpha{alpha}')
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



    degrees = model.layer2.degrees.data.clone().cpu().detach().numpy()
    remain_pos = np.where(degrees[:, 0]==0)[0]
    degrees = degrees[remain_pos] # only for those exclude X_tilde
    alphas =  [''.join(row.astype(int).astype(str)) for row in degrees]
    print(alphas)






    train_flags = generate_train_flags(train_cmd)

    print(train_flags)
    results = []
    for phase, train_flag in enumerate(train_flags):
        train_C, train_W, train_a = train_flag[0], train_flag[1], train_flag[2]
        phase_results = train(model, 
                            train_loader, 
                            val_loader, 
                            n_train, 
                            n_val,
                            n_epochs[phase],
                            device,
                            remain_pos,
                            criterion, 
                            lr_list[phase],
                            phase+1,
                            train_C=train_C, 
                            train_W=train_W, 
                            train_a=train_a,
                        )
        results.append(phase_results)
        print(phase_results.keys())
        

    combine_and_save_results(save_file_path, results)

if __name__ == "__main__":
    main()