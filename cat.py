import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d, d_out, heads):
        super().__init__()
        self.d = d
        self.heads = heads
        self.d_out = d_out
        self.A = nn.Parameter(torch.Tensor(heads, d, d))
        nn.init.zeros_(self.A)

    def forward(self, h): 
        B, T, d = h.size() # [bs, T, d]
        outs = []
        for i in range(self.heads):
            scores = torch.matmul(torch.matmul(h, self.A[i]), h.transpose(-2,-1)) # [bs, T, d]
            mask = torch.tril(torch.ones(T, T)).to(h.device).unsqueeze(0) # [1, T, T]
            scores = scores.masked_fill(mask == 0, float('-inf')) # Apply causal mask
            attn = F.softmax(scores, dim=-1) # [bs, T, T]
            out = torch.matmul(attn, h) # [bs, T, d]
            outs.append(out) # [head, bs, T, d]
        outs = torch.stack(outs).permute(1, 2, 0, 3)
        assert outs.shape == (B, T, self.heads, d)
        return outs.reshape(B, T, -1)  # [bs, T, h*d]

class DisentangledTransformer(nn.Module):
    def __init__(self, S, n_heads, n_layers, T, d_out):
        super().__init__()
        self.get_dims(S+T,n_heads,n_layers)
        self.layers = nn.ModuleList([
            CausalSelfAttention(self.dims[_], self.dims[_+1], n_heads[_],) 
            for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(self.dims[-1], d_out, bias=False) # d_L, d_out
        nn.init.zeros_(self.output_layer.weight.data)

    def get_dims(self, d0, n_heads, n_layers):
        self.dims = [d0]
        for i in range(n_layers):
            self.dims.append(self.dims[-1]*(1+n_heads[i]))
    
    def forward(self, x):
        B, T, S = x.size()
        position = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1) # (bs, T) 
        position = F.one_hot(position, num_classes=T).float()  # (bs, T, T) T pos emb indices
        h = torch.cat([x, position],-1) # (bs, T, d0)
        assert h.shape[-1] == S + T
        for attn_layer in self.layers:
            # print(h.shape)
            h_attn = attn_layer(h) # # (bs, T, m_{l-1} * d_{l-1})
            h = torch.cat([h, h_attn], -1) # (bs, T, d_l)
        logits = self.output_layer(h)  # [bs, T, d_out]
        return logits
