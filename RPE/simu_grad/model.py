import torch
import torch.nn as nn
from transformer_base import MultiHeadAttention
from typing import Literal
import itertools

class toyModel(nn.Module):
    def __init__(self, H: int, dim: int):
        super().__init__()
        self.Copier = Copier(H=H)
        self.Encoder = PolyKernel_MultiHeadAttention(
            num_heads=1,
            num_components=H,
            dimension=dim,
            max_individual_degree=1,
            init_method="ones",
            a = 0.01
        )
        self.Encoder.coeff.data = torch.ones_like(self.Encoder.coeff.data) * .01
    
    def forward(self, x):
        d = x.shape[-1]
        x = self.Copier.forward(x)
        x = self.Encoder(x[..., -1:, d:], x[..., :-1, d:], x[..., :-1, 0:d])
        return x
    

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
            init_method: Initialization method for the coefficients.
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

        # initialize the coefficients
        if init_method == "normal":
            self.coeff = nn.Parameter(torch.randn(num_heads, (max_individual_degree + 1) ** num_components)) 
        elif init_method == "zero":
            self.coeff = nn.Parameter(torch.zeros(num_heads, (max_individual_degree + 1) ** num_components))
        elif init_method == "ones":
            self.coeff = nn.Parameter(torch.ones(num_heads, (max_individual_degree + 1) ** num_components))
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
                self.a = nn.Parameter(torch.ones(1))
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
        logits_shift = torch.einsum("bqsl,hl->bhqs", logits_shift, self.coeff ** 2) # [batch_size, num_heads, query_len, seq_len]

        # layer normalization
        logits_shift = logits_shift / self.coeff.norm(dim=-1, keepdim=True) ** 2
        
        o, _ = super().forward(query, torch.zeros_like(key, device=key.device), value, logits_shift=logits_shift * self.a)
        return o.squeeze(1)
    

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion