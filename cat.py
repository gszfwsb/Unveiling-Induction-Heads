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
        # self.A = nn.Linear(d, heads*d, bias=False)


    def forward(self, h, return_score=False): 
        B, T, d = h.size() # [bs, T, d]
        outs = []
        for i in range(self.heads):
            scores = torch.matmul(torch.matmul(h, self.A[i]), h.transpose(-2,-1)) # [bs, T, T]
            # mask = torch.tril(torch.ones(T, T)).to(h.device).unsqueeze(0) # [1, T, T]
            # scores = scores.masked_fill(mask == 0, float('-inf')) # Apply causal mask
            # causal mask
            mask = torch.full((1, T, T), float('-inf'), device=h.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)
            scores = scores + mask  # (bs, T, T)
            attn = F.softmax(scores, dim=-1) # [bs, T, T]
            out = torch.matmul(attn, h) # [bs, T, d]
            outs.append(out) # [head, bs, T, d]
        outs = torch.stack(outs).permute(1, 2, 0, 3)
        assert outs.shape == (B, T, self.heads, d)
        if return_score:
            return outs.reshape(B, T, -1), attn # [bs, T, h*d]
        else:
            return outs.reshape(B, T, -1)  # [bs, T, h*d]
            
class DisentangledTransformer(nn.Module):
    def __init__(self, S, n_heads, n_layers, T, d_out):
        super().__init__()
        self.get_dims(S+T,n_heads,n_layers)
        self.layers = nn.ModuleList([
            CausalSelfAttention(self.dims[_], self.dims[_+1], n_heads[_],) 
            for _ in range(n_layers)
        ])
        # self.Wo = nn.Linear(self.dims[-1], d_out, bias=False) # d_L, d_out
        self.Wo = nn.Parameter(torch.Tensor(d_out, self.dims[-1]))
        position = torch.arange(T, dtype=torch.long) # (T) 
        self.position = F.one_hot(position, num_classes=T).float()  # (T, T) T pos emb indices

    def get_dims(self, d0, n_heads, n_layers):
        self.dims = [d0]
        for i in range(n_layers):
            self.dims.append(self.dims[-1]*(1+n_heads[i]))
            
    def forward(self, x):
        B, T, S = x.size()
        # print(f'x:{x.shape},{x[0]}')
        position = self.position.unsqueeze(0).expand(B,T,T)
        # print(f'position:{position.shape},{position[0]}')
        position = position.to(x.device)
        h = torch.cat([x, position],-1) # (bs, T, d0)
        # print(f'h:{h.shape},{h[0]}')
        assert h.shape[-1] == S + T
        for attn_layer in self.layers:
            # print(h.shape)
            h_attn = attn_layer(h) # # (bs, T, m_{l-1} * d_{l-1})
            h = torch.cat([h, h_attn], -1) # (bs, T, d_l)
            # print(f'h:{h.shape},{h[0]}')
        # print(f'h:{h.shape},{h[0]}')
        logits = torch.matmul(h, self.Wo.T)  # [bs, T, d_out]
        # logits = self.Wo(h)  # [bs, T, d_out]
        # print(f'logits:{logits.shape},{logits[0]}')
        return logits
