from model import toyModel
from dataset import NGramDataset
import torch 
import torch.nn as nn
from tqdm import tqdm, trange
from tools import plot_hist

S = 3
L = 500
n_language = 3
alpha = 0.3
n_sample = 5000
H = 3
power = 2
bs, n_epochs = 5000, 1


n = n_language - 1
model = toyModel(H=H, dim=S)


# training loop
optimizer = torch.optim.SGD(model.parameters(), lr=10)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
train_loader = torch.utils.data.DataLoader(
    NGramDataset(S, L, n_language, alpha, n_sample, output=True), 
    batch_size=bs, 
    shuffle=True)
pbar = tqdm(range(n_epochs),ncols=100,mininterval=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in pbar:
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.4f}")

# plot the histogram of the weights 
plot_hist(model.Encoder.coeff.grad.detach().cpu(), 2, H, title='coeff_grad')
