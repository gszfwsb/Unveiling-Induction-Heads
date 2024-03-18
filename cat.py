import torch
import torch.nn as nn
import torch.nn.functional as F

# class CausalSelfAttention_QKV(nn.Module):
#     def __init__(self, d, heads):
#         super().__init__()
#         self.d = d
#         self.heads = heads
#         self.scale = (self.d) ** -0.5

#         self.query = nn.Linear(self.d, self.heads*self.d, bias=False)
#         self.key = nn.Linear(self.d, self.heads*self.d, bias=False)
#         self.value = nn.Linear(self.d, self.heads*self.d, bias=False)

#     def forward(self, h): 
#         B, T, d = h.size() # [bs, T, d]
#         # Split the input into multiple heads for Q, K, and V
#         Q = self.query(h).view(B, T, self.heads, self.d).transpose(1, 2)  # [bs, heads, T, d]
#         K = self.key(h).view(B, T, self.heads, self.d).transpose(1, 2)    # [bs, heads, T, d]
#         V = self.value(h).view(B, T, self.heads, self.d).transpose(1, 2)  # [bs, heads, T, d]

#         # Causal attention with scale
#         A = torch.matmul(Q.transpose(-2, -1), K) * self.scale # [bs, heads, d, d]
#         assert A.shape == (B, self.heads, d, d)
#         scores = torch.zeros((B, self.heads, T, T))
#         for i in range(self.heads):
#             score = torch.matmul(torch.matmul(h, A[:, i]), h.transpose(-2,-1))
#             scores[:,i] =  score# [bs, T, T], 
#         mask = torch.tril(torch.ones(T, T)).to(h.device).unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
#         scores = scores.masked_fill(mask == 0, float('-inf')) # Apply causal mask
#         attn = F.softmax(scores, dim=-1) # [bs, heads, T, T]

#         # Apply attention to V
#         out = torch.matmul(attn, V) # [bs, heads, T, d]
#         # Concatenate the attention outputs from all heads
#         assert out.shape == (B,self.heads,T,d)
#         return out.view(B, T, -1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d,T,d_out, heads):
        super().__init__()
        self.d = d
        self.heads = heads
        self.d_out = d_out
        self.scale = (d) ** -0.5
        self.A = nn.Parameter(torch.Tensor(heads, d, d))
        nn.init.xavier_uniform_(self.A)

    def forward(self, h): 
        B, T, d = h.size() # [bs, T, d]
        assert self.A.shape == (self.heads, d, d)
        scores = torch.zeros((B, self.heads, T, T)).to(h.device)
        for i in range(self.heads):
            score = torch.matmul(torch.matmul(h, self.A[i]), h.transpose(-2,-1))
            scores[:,i] =  score# [bs, T, T], 
        mask = torch.tril(torch.ones(T, T)).to(h.device).unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        scores = scores.masked_fill(mask == 0, float('-inf')) # Apply causal mask
        attn = F.softmax(scores, dim=-1) # [bs, heads, T, T]

        # Apply attention to V
        attn = torch.einsum("bhtt,btd->bhtd",attn, h) # [bs, heads, T, d]
        # Concatenate the attention outputs from all heads
        assert attn.shape == (B,self.heads,T,d)
        return attn.view(B, T, -1)

class DisentangledTransformer(nn.Module):
    def __init__(self, S, n_heads, n_layers, T, d_out):
        super().__init__()
        self.get_dims(S+T,n_heads,n_layers)
        self.layers = nn.ModuleList([
            CausalSelfAttention(self.dims[_], T, self.dims[_+1], n_heads[_],) 
            for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(self.dims[-1], d_out) # d_l, d_out
        # print(self.dims)

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
        logits = self.output_layer(h)  # [bs, T, S]
        return logits



# Example usage:
# S = 10  # Define your vocab size here (size of alphabet)
# d_out = 10
# n_layers = 2
# n_heads = [1,1]
# T = 20
# bs = 10

# Example input
# x = torch.randint(0, S, (bs, T))  # (bs, T) 
# x = F.one_hot(x, num_classes=S).float()  # (bs, T, S) S word emb indices

# Generate a function and a prompt
# x = torch.randint(0, S, (bs, T)) # (bs, T) 
# x = F.one_hot(x, num_classes=S).float()  # (bs, T, S) S word emb indices
# model = DisentangledTransformer(S, n_heads, n_layers, T, d_out)
# pred = model(x)
# print(pred.shape)