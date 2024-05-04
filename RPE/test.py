import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.stdout = open('log.txt', 'w')

def set_seed(seed=3407):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.enabled = False  # type: ignore
    return

set_seed(2024)
class Simplified_MultiHeadAttention(nn.Module):
    def __init__(self, T, n_parent, H, w_plus, w_minus):
        super(Simplified_MultiHeadAttention, self).__init__()
        self.T = T
        self.H = H
        self.W = torch.ones((self.T,self.H)) * w_minus
        torch.diagonal(self.W, 0).fill_(w_plus)
        self.W = nn.Parameter(self.W)
    def forward(self, X):
        X_tilde = torch.cat([X, torch.zeros_like(X[..., :1, :], device=X.device)], dim=-2)
        # V.append(X_tilde)
        for h in range(self.H):
            W_h = torch.full((self.T+1, self.T+1), float('-inf'), device=X.device) # [T+1, T+1]
            # torch.diagonal(W_h, 0).fill_(self.W[:, h][-1])  # TODO: Set the main diagonal!
            for j in range(self.H):
                torch.diagonal(W_h, -(j+h+1)).fill_(self.W[:, h][j+h])  # Set the (j)-th negative diagonal
            # print(W_h)
            W_h = F.softmax(W_h, dim=-1)
            W_h[torch.isnan(W_h)] = 0
            v_h = torch.einsum("mn,bnd->bmd",W_h, X_tilde) # [T+1, T+1], [bs, T+1, d] -> [bs, T+1, d]
            X_tilde = torch.cat([X_tilde, v_h.clone()], dim=-1)
        V = X_tilde.clone()
        return V


class Copier:
    def __init__(self, H: int):
        self.H = H # Window length for copying
    
    def forward(self, x):
        """
        x: [bs, seq_len, S]
        """
        assert x.shape[-2] >= self.H, "Sequence length must be at least H"

        # Add a zero column to the end of x
        x = torch.cat([x, torch.zeros_like(x[..., :1, :], device=x.device)], dim=-2)
        y = x.clone()
        for h in range(self.H):
            # delete the last (h+1) tokens
            y = y[..., :-1, :]
            # add (h+1) zeros to the beginning
            y = torch.cat([torch.zeros_like(y[..., :1, :], device=y.device), y], dim=-2)
            x = torch.cat([x, y.clone()], dim=-1)


        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x.reshape(x.shape[0], x.shape[1], -1)

# 初始输入张量 x
H = 2
x = torch.randn(1, 10, 3)
print(x)

print(x.shape)


# 创建 Copier 实例
copier = Copier(H)

# 应用 forward 方法
output = copier.forward(x)

# 输出结果
print(output)

print("###############################################")


model = Simplified_MultiHeadAttention(10, 3, H=H, w_plus=1e5, w_minus=0.5)

# 应用 forward 方法
output = model(x)
output = output.reshape(x.shape[0], x.shape[1]+1, -1)

# 输出结果
print(output)

# print(output.shape)


