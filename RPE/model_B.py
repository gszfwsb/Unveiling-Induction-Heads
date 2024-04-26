import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerTransformer(nn.Module):
    def __init__(self, 
                vocab_size, 
                seq_length, 
                num_heads, 
                window_length, 
                w_plus, 
                w_minus, 
                a):
        super(TwoLayerTransformer, self).__init__()
        self.L = seq_length
        self.H = num_heads
        self.M = window_length
        # Assuming d_model is equal to input_dim
        self.T = vocab_size
        self.W = torch.Tensor(self.L,self.H)
        self.A = torch.Tensor(self.H*self.T,self.H*self.T)
        self.init(w_plus, w_minus, a)
        # set to parameters
        self.W = nn.Parameter(self.W) 
        self.A = nn.Parameter(self.A)
        self.norm = nn.LayerNorm(self.T) # norm for vocab_size
    def init(self, w_plus, w_minus, a):
        # for W
        nn.init.constant_(self.W, w_minus)
        self.W.diagonal()[:self.M].fill_(w_plus) 
        # for A
        nn.init.zeros_(self.A)
        self.A.diagonal().fill_(a)

    def forward(self, x):
        # x: (bs, L, S)
        # First layer
        V = []
        for i in range(self.H):
            W_h = torch.zeros((self.L, self.L)).to(x.device) # L*L
            for j in range(self.L):
                torch.diagonal(W_h, -j).fill_(self.W[:, i][j])  # Set the (j+1)-th negative diagonal
            mask = torch.triu(torch.full_like(W_h, float('-inf')), diagonal=1)
            W_h = W_h + mask
            W_h = F.softmax(W_h, dim=-1)
            attn_h = torch.matmul(W_h, x) # [L, L], [bs, L, S]
            V.append(attn_h)
        V.append(x) # [bs, L, S, (H+1)]
        V = torch.stack(V) # [bs, L, S, (H+1)]
        V = self.norm(V) # [bs, L, S, (H+1)]
        W_2 = torch.matmul(torch.matmul(V[:,-1:,:self.H*self.T], self.A),
                            V[:,:-1,:self.H*self.T].permute(0,2,1)) # [bs, 1, HS], [HS, HS], [bs, HS, L-1]
        W_2 = F.softmax(W_2, dim=-1) # [bs, L, L-1]
        Y = torch.matmul(W_2, V[:,:-1,self.H*self.T:]) # [bs, 1, L-1], [bs, L-1, S] -> [bs, 1, S]
        Y = Y.squeeze(1) # [bs, S]
        return Y

def test_model():
    # Parameters
    vocab_size = 5 # S
    seq_length = 3 # L
    num_heads = 2 # H
    window_length = 2 # M
    w_plus = 0.1
    w_minus = 0.05
    a = 0.1
    # Create the model
    model = TwoLayerTransformer(vocab_size, seq_length, num_heads, window_length, w_plus, w_minus, a)
    # Test inputs
    bs = 2  # batch size
    x = torch.rand(bs, seq_length, vocab_size)
    # Forward pass
    output = model(x)
    print(output)

# test_model()