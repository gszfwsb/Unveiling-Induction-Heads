import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import sys

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
        V = X_tilde.clone()
        for h in range(self.H):
            W_h = torch.full((self.T+1, self.T+1), float('-inf'), device=X.device) # [T+1, T+1]
            for j in range(self.H):
                torch.diagonal(W_h, -(j+h+1)).fill_(self.W[:, h][j+h])  # Set the (j)-th negative diagonal
            W_h = F.softmax(W_h, dim=-1)
            W_h = torch.nan_to_num(W_h, nan=0.0)  # Safely convert NaNs to zero after softmax
            v_h = torch.matmul(W_h, X_tilde) # [T+1, T+1], [bs, T+1, d] -> [bs, T+1, d]
            V = torch.cat([V, v_h.clone()], dim=-1)
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
H = 8
bs = 100
seq_length = 100
vocab_size = 10
x = torch.randn(bs, seq_length, vocab_size)


# 创建 Copier 实例
copier = Copier(H)

# 应用 forward 方法
output1 = copier.forward(x)

# 输出结果



model = Simplified_MultiHeadAttention(seq_length, vocab_size, H=H, w_plus=1e5, w_minus=0.5)

# 应用 forward 方法
output = model(x)
output = output.reshape(x.shape[0], x.shape[1]+1, -1).detach()



# 输出结果
yeah = (output==output1).all()
print(yeah) 


