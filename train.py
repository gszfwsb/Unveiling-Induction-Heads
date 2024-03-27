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


def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion

parser = argparse.ArgumentParser('train 2-layer disentangled Transformer')
parser.add_argument('--vocab-size',type=int,default=10)
parser.add_argument('--seq-length',type=int, default=20)
parser.add_argument('--n-layers',type=int, default=2)
parser.add_argument('--lr',type=float, default=0.01)
parser.add_argument('--n-heads',type=list,default=[1,1])
parser.add_argument('--batch-size',type=int, default=1024)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--seed',type=int, default=2024)
parser.add_argument('--ignore-idx',type=int, default=-100)
parser.add_argument('--n-sample',type=int,default=2**14)
parser.add_argument('--device',type=str, default='cuda:0')
parser.add_argument('--n-epochs',type=int, default=1000)
parser.add_argument('--enable-wandb',type=bool,default=False)

args = parser.parse_args()

set_seed(args.seed)
device = args.device
S = args.vocab_size  # Define your vocab size here (size of alphabet)
T = args.seq_length
n_layers = args.n_layers
n_heads = args.n_heads
n_sample = args.n_sample
bs = args.batch_size
alpha = args.alpha  # Dirichlet parameter
ignore_idx = args.ignore_idx
n_sample = args.n_sample
lr = args.lr
n_epochs = args.n_epochs

if not args.enable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'

# wandb init
wandb.init(project='In-Context-Learning', 
           entity='shaobowang', 
           name=f'Task1_random_bs{bs}_a{alpha}_T{T}_S{S}',
           config=vars(args)
        )

# Define the file paths
# root_path = '/cpfs01/user/luanqi.p/wangshaobo/data'
root_path = './data'
save_file_path = f'results/Task1_random_Markov/{bs}_{lr}_{alpha}_T{T}_S{S}'
makedirs(save_file_path)

# Generate the DisentangledTransformer
model = DisentangledTransformer(S, n_heads, n_layers, T, S)
model.to(device)

# set markov converged matrices
A1 = torch.zeros((S+T,S+T)).to(device)
A1[S+1:-1, S:-2] = torch.eye(T-2).to(device)
A2 = torch.zeros((2*(S+T)), 2*(S+T)).to(device)
A2[:S, S+T:S+T+S] = torch.eye(S).to(device)
Wo = torch.zeros((S, 4*(S+T))).to(device)
Wo[:, 2*(S+T):2*(S+T)+S] = torch.eye(S).to(device)

model.layers[0].A.weight.data = A1
model.layers[1].A.weight.data = A2
model.Wo.weight.data = Wo

# Generate the population loss
criterion = population_loss(args.ignore_idx)
 
# define optimizers and schedulars
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0, momentum=0)
# optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50)

c_dict = {
    'A_type': 'Markov chain',
    'T': T,
    'dim': S,
    'number_of_samples': n_sample
}
data_method = GraphCausalModel(c_dict)
# Call the __generatedata__ method
generated_data = data_method.__generatedata__()
x,y= data_method.__transform__(generated_data)

dataset_file_path = f'{root_path}/Task1_new_data_seed{args.seed}_n{n_sample}.pt'  # Specify your path here

if not os.path.isfile(dataset_file_path):
# if True:
    print('generate and save the dataset')
    # If not, generate and save the dataset
    save_dataset(x,y, dataset_file_path)
    print('finishd dataset generation')
else:
    x, y = load_dataset(dataset_file_path)
    print('already cache, load from disk!')

dataset = TensorDataset(x,y)

dataloader = DataLoader(dataset, batch_size=bs)

# visualize before train
visualize(model, save_file_path, 'init')
# save(model,save_file_path,'init')

pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)
step = 0
global_step = 0


for epoch in pbar:
    for x, y in dataloader:
        # assert not (torch.isnan(x).any() or torch.isnan(x).any())
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x) # [bs, T, S]
        logits[:,:T-1,:] = ignore_idx # set to ignore index, only T is valid
        loss = criterion(logits, y.long())
        loss.backward()
        optimizer.step()
        # Update the learning rate
        # scheduler.step()
        pbar.set_description(f'loss:{loss.item():.10f}')
        
        step += 1
        global_step += bs

        # Log the loss and heatmap of A1 after every update
        if step % 50 == 0:   
            visualize(model, save_file_path, step)
            
        # if step % 100 == 0:   
        #     save(model,save_file_path,step)

# visualize at the end
visualize(model, save_file_path)
save(model,save_file_path)

# Finish the wandb run
wandb.finish()