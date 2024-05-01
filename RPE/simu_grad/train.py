from model import toyModel
from dataset import NGramDataset
import torch 
import torch.nn as nn
from tqdm import tqdm, trange
from tools import plot_hist

S, L, n_language, alpha, n_sample, H = 3, 100, 3, 0.3, 1000, 3
n = n_language - 1
bs, n_epochs = 1000, 5000
power = 2

dataset = NGramDataset(S, L, n_language, alpha, n_sample)
model = toyModel(H=2, dim=S)

# pass one batch through the model
data = next(iter(dataset))
x = data[0]
y = data[1]
# forward pass
z = model(x)

# training loop
optimizer = torch.optim.SGD(model.parameters(), lr=10)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
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
plot_hist(model.Encoder.coeff.data.detach().cpu(), 2, n, title='coeff')
