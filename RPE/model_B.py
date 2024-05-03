import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_base import MultiHeadAttention
import itertools
from typing import Literal


class SimplifiedLayerNorm(nn.Module):
    def __init__(self, dim=-1, eps=1e-7):
        super(SimplifiedLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, x):
        norm = torch.norm(x, dim=self.dim, keepdim=True)
        out = x / (norm + self.eps)
        return out

class Simplified_MultiHeadAttention(nn.Module):
    def __init__(self, T, H, w_plus, w_minus):
        super(Simplified_MultiHeadAttention, self).__init__()
        self.T = T
        self.H = H
        self.W = torch.ones((self.T,self.H)) * w_minus
        self.W.diagonal().fill_(w_plus)
        self.W = nn.Parameter(self.W)
        self.norm = SimplifiedLayerNorm(dim=-1)
    def forward(self, X):
        X_tilde = torch.cat([X, torch.zeros((X.size(0), 1, X.size(2))).to(X.device)], dim=1) # (bs, T+1, d)
        V = []
        for h in range(self.H):
            # W_h = torch.zeros((self.T+1, self.T+1)).to(X_tilde.device) # [T+1, T+1]
            W_h = torch.full((self.T+1, self.T+1), float('-inf'), device=X.device) # [T+1, T+1]
            for j in range(self.T):
                torch.diagonal(W_h, -j).fill_(self.W[:, h][j])  # Set the (j+1)-th negative diagonal
            W_h = F.softmax(W_h, dim=-1)
            v_h = torch.matmul(W_h, X_tilde) # [T+1, T+1], [bs, T+1, d] -> [bs, T+1, d]
            v_h_normalized = self.norm(v_h)
            V.append(v_h_normalized)
        V.append(X_tilde) # [bs, T+1, d, (H+1)]
        V = torch.stack(V, -1) # [bs, T+1, d, (H+1)]
        return V





class PolyKernel_MultiHeadAttention(MultiHeadAttention):
    def __init__(self, 
                 num_heads: int,
                 num_components: int, 
                 dimension: int,
                 max_individual_degree: int = 2, 
                 init_method: Literal["normal", "zero", "ones"] = "zero", 
                    **kwargs
                 ):
        """
        Initialize the PolyKernel_Attention module.

        Args:
            num_heads: Number of heads in the multi-head attention.
            num_components: Number of components in the query/key/value.
            max_individual_degree: Maximum degree of the individual component.
            init_method: Initialization method for the C_alpha_listicients.
        Returns:
            None
        """
        super().__init__(
            num_heads=num_heads, 
            q_dim=dimension * num_components, 
            v_dim=dimension,
            q_k_v_o_proj_enabled=[False, False, False, False],
        )
        self.max_individual_degree = max_individual_degree
        self.num_components = num_components

        # initialize the C_alpha_listicients
        if init_method == "normal":
            self.C_alpha_list = nn.Parameter(torch.randn(num_heads, (max_individual_degree + 1) ** num_components)) 
        elif init_method == "zero":
            self.C_alpha_list = nn.Parameter(torch.zeros(num_heads, (max_individual_degree + 1) ** num_components))
        elif init_method == "ones":
            self.C_alpha_list = nn.Parameter(torch.ones(num_heads, (max_individual_degree + 1) ** num_components))
        else:
            raise ValueError("init_method should be either 'normal' or 'zero'")

        # generate a matrix where each row is a vector of degrees on each component with total degree <= max_individual_degree
        self.degrees = torch.tensor(list(itertools.product(range(max_individual_degree + 1), repeat=num_components))) # [max_individual_degree ** num_components, num_components]
        # float type
        self.degrees = self.degrees.type(torch.float32)
        # register the degrees as a buffer
        self.register_buffer("mydegree", self.degrees)

        if "a" in kwargs:
            if isinstance(kwargs["a"], float):
                self.a = kwargs["a"]
            elif kwargs["a"] == "learnable":
                self.a = nn.Parameter(torch.ones(1)*kwargs["a_init"])
        else:
            self.a = 1


    def forward(self, query, key, value):
        """
        Args:
            query: [batch_size, query_len, q_dim]
            key: [batch_size, seq_len, q_dim]
            value: [batch_size, seq_len, q_dim]

        Returns:
            y: [batch_size, seq_len, seq_len]
        """
        if query.dim() == 2:
            query = query.unsqueeze(0)
        batch_size, query_len, q_dim = query.size()
        if key.dim() == 2:
            key = key.unsqueeze(0)
        batch_size, seq_len, _ = key.size()
        
        # reshape the key and the value to [batch_size, seq_len, num_components, q_dim / num_components]
        assert q_dim % self.num_components == 0
        key_new = key.view(batch_size, seq_len, self.num_components, -1)
        query_new = query.view(batch_size, query_len, self.num_components, -1)

        logits_shift = torch.einsum("bqcd,bscd->bqsc", query_new, key_new) # [batch_size, query_len, seq_len, num_components]
        logits_shift = torch.exp(torch.einsum("bqsc,lc->bqsl", torch.log(logits_shift + 1e-24), self.degrees.to(logits_shift.device))) # [batch_size, query_len, seq_len, num_components ** max_individual_degree]
        logits_shift = torch.einsum("bqsl,hl->bhqs", logits_shift, self.C_alpha_list ** 2) # [batch_size, num_heads, query_len, seq_len]

        # layer normalization
        logits_shift = logits_shift / self.C_alpha_list.norm(dim=-1, keepdim=True) ** 2
        
        o, _ = super().forward(query, torch.zeros_like(key, device=key.device), value, logits_shift=logits_shift * self.a)
        return o.squeeze(1)
  

class TwoLayerTransformer(nn.Module):
    def __init__(self, 
                vocab_size,
                seq_length,
                num_heads,
                w_plus,
                w_minus,
                a_init,
                c_alpha_init):
        super(TwoLayerTransformer, self).__init__()
        self.T = seq_length
        self.H = num_heads
        self.d = vocab_size
        # layer 1: attention
        self.layer1 = Simplified_MultiHeadAttention(
            self.T, 
            self.H, 
            w_plus, 
            w_minus
        )
        # layer 2: attention
        self.layer2 = PolyKernel_MultiHeadAttention(
            num_heads=1,
            num_components=self.H,
            dimension=self.d,
            max_individual_degree=1,
            init_method="ones",
            a = "learnable",
            a_init = a_init
        )
        # init params
        self.layer2.C_alpha_list.data = torch.ones_like(self.layer2.C_alpha_list.data) * c_alpha_init

    def forward(self, X):
        X = self.layer1(X) # [bs, T+1, d, H]
        X = X.view(X.shape[0], X.shape[1], -1)
        X = self.layer2(X[..., -1:, self.d:], X[..., :-1, self.d:], X[..., :-1, 0:self.d])
        return X
  

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion