import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformer_base import MultiHeadAttention
import itertools
import math
from typing import Optional, Union, Literal

class MultiIdentity(nn.Module):
    """A module that applies multiple identity operations."""
    def __init__(self, num_copies: int):
        """
        Initialize the MultiIdentity module.

        Args:
            num_copies (int): Number of copies.
        """
        super().__init__()
        self.num_copies = num_copies

    def forward(self, x):
        return x.repeat(-1, self.num_copies)
    


class MultiHeadAttention(nn.Module):
    """
    Implementation of multi-head self/cross attention.
    
    In this implementation, the query and key's embedding dimension does not need to be a split of the number of heads, and the attention scores are normalized by the square root of the embedding dimension of the query and key for each head.
    """

    def __init__(
            self,
            num_heads: int,
            q_dim: int = None,
            k_dim: int = None,
            v_dim: int = None,
            o_dim: int = None,
            qk_embed_size_per_head: int = None,
            vo_embed_size_per_head: int = None,
            attention_type: Literal["softmax", "relu", "linear"] = "softmax",
            use_bias: bool = False,
            dropout_rate: float = 0.0,
            q_k_v_o_proj_enabled: list = [True, True, True, True],
            initialization_method: Literal["normal", "small identity", None] = None,
            use_rel_pos: bool = False,
            use_rel_pos_proj: bool = True, 
            rel_pos_win: tuple = (-32, 32),
            rel_pos_embed_size: int = 8,
            use_causal_attn: bool = False,
            **kwargs,
    ):
        """
        Initialize the MultiHeadAttention module.

        Args:
            qk_embed_size_per_head (int): Size of each head for queries and keys.
            v_embed_size (int): Size of each head for values.
            num_heads (int): Number of heads.
            attention_type (str): Type of attention. Can be 'softmax', 'relu' or 'linear'.
            use_bias (bool): Whether to use bias term in projection layers.
            dropout_rate (float): Dropout rate.
            q_k_v_o_proj_enabled (list): List of booleans indicating whether to enable projection layers for queries, keys, values and outputs.
            q_dim (int): Dimension of queries. If None, defaults to qk_embed_size_per_head * num_heads.
            k_dim (int): Dimension of keys. If None, defaults to qk_embed_size_per_head * num_heads.
            v_dim (int): Dimension of values. If None, defaults to v_embed_size * num_heads.
            o_dim (int): Dimension of outputs. If None, defaults to v_embed_size * num_heads.
            initialization_method (str): Initialization method. Can be 'normal', 'small identity' or None.
            use_rel_pos (bool): Whether to use relative positional encoding.
            use_rel_pos_proj (bool): Whether to use projection for relative positional encoding.
            rel_pos_win (tuple): Tuple of two integers indicating the relative positional window.
            rel_pos_embed_size (int): Embedding size for relative positional encoding.
            attn_msk (str): Attention mask.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If qk_embed_size is not divisible by num_heads.
        """
        super().__init__()
        self._num_heads = num_heads
        self.q_k_v_o_proj_enabled = q_k_v_o_proj_enabled

        # Set the size of for queries, keys and values and outputs.
        self.q_dim = q_dim
        self.k_dim = k_dim if k_dim is not None else self.q_dim

        self.v_dim = v_dim if v_dim is not None else self.q_dim
        self.o_dim = o_dim if o_dim is not None else self.v_dim

        # find the maximum of q_dim, k_dim, v_dim and o_dim
        max_dim = max(self.q_dim, self.k_dim)

        # Initialization of embedding sizes per head.
        self._qk_embed_size_per_head = qk_embed_size_per_head if qk_embed_size_per_head is not None else int(math.ceil(max_dim / self._num_heads))
        self._qk_embed_size = self._qk_embed_size_per_head * self._num_heads
        
        self._vo_embed_size_per_head = vo_embed_size_per_head if vo_embed_size_per_head is not None else int(math.ceil(self.v_dim / self._num_heads))
        self._vo_embed_size = self._vo_embed_size_per_head * self._num_heads
        
        


        # Initialization of attention activation.
        if attention_type == 'softmax':
            self.attention_activation = nn.Softmax(dim=-1)
        elif attention_type == 'relu':
            self.attention_activation = nn.ReLU()
        elif attention_type == 'linear':
            self.attention_activation = nn.Identity()
        else:
            raise NotImplementedError(
                f"Attention type {attention_type} is not implemented!"
            )
        self.attention_type = attention_type


        # initialize the q_proj, k_proj, v_proj and o_proj layers for each head
        if q_k_v_o_proj_enabled[0]:
            self.q_proj = nn.Linear(
                in_features=self.q_dim,
                out_features=self._qk_embed_size,
                bias=use_bias,
            )  
        else:
            if self._qk_embed_size == self.q_dim:
                self.q_proj = nn.Identity()
            elif self._qk_embed_size == self.q_dim * self._num_heads:
                # nn.Identity() copied for each head
                self.q_proj = MultiIdentity(self._num_heads)
            else:
                raise ValueError(
                    f"q_proj must be enabled for q_dim {self.q_dim} and qk_embed_size {self._qk_embed_size}!"
                )
        
        if q_k_v_o_proj_enabled[1]:
            self.k_proj = nn.Linear(
                in_features=self.k_dim,
                out_features=self._qk_embed_size,
                bias=use_bias,
            )
        else:
            if self._qk_embed_size == self.k_dim:
                self.k_proj = nn.Identity()
            elif self._qk_embed_size == self.k_dim * self._num_heads:
                # nn.Identity() copied for each head
                self.k_proj = MultiIdentity(self._num_heads)
            else:
                raise ValueError(
                    f"k_proj must be enabled for k_dim {self.k_dim} and qk_embed_size {self._qk_embed_size}!"
                )
        
        if q_k_v_o_proj_enabled[2]:
            self.v_proj = nn.Linear(
                in_features=self.v_dim,
                out_features=self._vo_embed_size, 
                bias=use_bias,
            )
        else:
            if self._vo_embed_size == self.v_dim:
                self.v_proj = nn.Identity()
            elif self._vo_embed_size == self.v_dim * self._num_heads:
                # nn.Identity() copied for each head
                self.v_proj = MultiIdentity(self._num_heads)
            else:
                raise ValueError(
                    f"v_proj must be enabled for v_dim {self.v_dim} and vo_embed_size {self._vo_embed_size}!"
                )
        
        if q_k_v_o_proj_enabled[3]:
            self.o_proj = nn.Linear(
                in_features=self._vo_embed_size,
                out_features=self.o_dim,
                bias=use_bias,
            )
        else:
            if self._vo_embed_size == self.o_dim:
                self.o_proj = nn.Identity()
            elif self._vo_embed_size == self.o_dim * self._num_heads:
                # nn.Identity() copied for each head
                self.o_proj = MultiIdentity(self._num_heads)
            else:
                raise ValueError(
                    f"o_proj must be enabled for o_dim {self.o_dim} and vo_embed_size {self._vo_embed_size}!"
                )

        # Initialization of dropout layer.
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize the weights.
        if initialization_method == "normal":
            if q_k_v_o_proj_enabled[0]:
                self.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            if q_k_v_o_proj_enabled[1]:
                self.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            if q_k_v_o_proj_enabled[2]:
                self.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            if q_k_v_o_proj_enabled[3]:
                self.o_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif initialization_method == "small identity":
            if q_k_v_o_proj_enabled[0]:
                assert self.q_dim == self._qk_embed_size, f"q_dim {self.q_dim} is not equal to qk_embed_size {self._qk_embed_size} for small identity initialization! Please set q_dim to qk_embed_size_per_head * num_heads."
                self.q_proj.weight.data = torch.eye(self.q_dim) * 1e-4
            if q_k_v_o_proj_enabled[1]:
                assert self.k_dim == self._qk_embed_size, f"k_dim {self.k_dim} is not equal to qk_embed_size {self._qk_embed_size} for small identity initialization! Please set k_dim to qk_embed_size_per_head * num_heads."
                self.k_proj.weight.data = torch.eye(self.k_dim) * 1e-4
            if q_k_v_o_proj_enabled[2]:
                assert self.v_dim == self._vo_embed_size, f"v_dim {self.v_dim} is not equal to vo_embed_size {self._vo_embed_size} for small identity initialization! Please set v_dim to vo_embed_size_per_head * num_heads."
                self.v_proj.weight.data = torch.eye(self.v_dim) * 1e-4
            if q_k_v_o_proj_enabled[3]:
                assert self.o_dim == self._vo_embed_size, f"o_dim {self.o_dim} is not equal to vo_embed_size {self._vo_embed_size} for small identity initialization! Please set o_dim to vo_embed_size_per_head * num_heads."
                self.o_proj.weight.data = torch.eye(self.o_dim) * 1e-4
        elif initialization_method == None:
            pass
        else:
            raise NotImplementedError(
                f"Initialization method {initialization_method} is not implemented!"
            )
        
        # relative positional encoding
        self.use_rel_pos = use_rel_pos
        self.rel_pos_win = rel_pos_win
        if self.use_rel_pos:
            assert isinstance(self.rel_pos_win, tuple), "Relative positional window must be a tuple!"
            assert len(self.rel_pos_win) == 2, "Relative positional window must have two elements!"
            assert self.rel_pos_win[0] < self.rel_pos_win[1], "The first element of the relative positional window must be less than the second element!"
            self.rel_pos_win_size = self.rel_pos_win[1] - self.rel_pos_win[0]
            self.rel_pos_embed_size = rel_pos_embed_size
            self.pos_key_proj = nn.Linear(self._qk_embed_size, self.rel_pos_embed_size, bias=False) if use_rel_pos_proj else nn.Identity()
            self.pos_query_proj = nn.Linear(self._qk_embed_size, self.rel_pos_embed_size, bias=False) if use_rel_pos_proj else nn.Identity()
            if not use_rel_pos_proj:
                assert self.rel_pos_embed_size == self._qk_embed_size, f"Specified relative positional embedding size {self.rel_pos_embed_size} does not match the query-key embedding size {self._qk_embed_size}!"
            self.pos_dropout = nn.Dropout(p=dropout_rate)
        
        self.use_causal_attn = use_causal_attn

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            rel_pos_embedding: Optional[nn.parameter.Parameter] = None,
            mask: Optional[Union[torch.Tensor, None]] = None,
            logits_shift: Optional[torch.Tensor] = None,
    ):
        """
        Apply a forward pass of attention.

        Args:
            query (torch.Tensor): Query tensor 
                of shape [batch_size, query_seq_len, q_dim].
            key (torch.Tensor): Key tensor
                of shape [batch_size, enc_seq_len, k_dim].
            value (torch.Tensor): Value tensor
                of shape [batch_size, enc_seq_len, v_dim].
            rel_pos_embedding (Optional[nn.parameter.Parameter]): Optional relative positional embedding.
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor of shape [batch_size, 1, x_seq_len, enc_seq_len].
            

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, x_seq_len, qk_embed_size].
        """

        # Linear projections for queries, keys and values.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q_seq_len = q.shape[-2]
        k_seq_len = k.shape[-2]

        # Reshape to 4D tensors of shape
        # [batch_size, seq_len, num_heads, qkv_size_per_head].
        q = q.reshape(-1, q.shape[1], self._num_heads, self._qk_embed_size_per_head)
        k = k.reshape(-1, k.shape[1], self._num_heads, self._qk_embed_size_per_head)
        v = v.reshape(-1, v.shape[1], self._num_heads, self._vo_embed_size_per_head)

        if self.use_rel_pos:
            assert rel_pos_embedding is not None, "Please provide the relative positional embedding!"
            assert rel_pos_embedding.shape[0] == self.rel_pos_win_size, f"Relative positional embedding shape {rel_pos_embedding.shape} does not match the window size {self.rel_pos_win_size}!"
            if self.use_rel_pos_proj:
                assert rel_pos_embedding.shape[1] == self.rel_pos_embed_size, f"Relative positional embedding shape {rel_pos_embedding.shape} does not match the relative positional embedding size {self.rel_pos_embed_size}!"
            else: 
                assert rel_pos_embedding.shape[1] == self._qk_embed_size, f"Relative positional embedding shape {rel_pos_embedding.shape} does not match the query-key embedding size {self._qk_embed_size}! Please set the relative positional embedding size to the query-key embedding size or enable projection for relative positional encoding."
            # compute the relative positional encoding
            rel_pos_k = self.pos_key_proj(rel_pos_embedding) # [rel_pos_win_size, rel_pos_embed_size]
            rel_pos_q = self.pos_query_proj(rel_pos_embedding)  # [rel_pos_win_size, rel_pos_embed_size]

            rel_pos_k = rel_pos_k.view(-1, self._num_heads, self._qk_embed_size_per_head)
            rel_pos_q = rel_pos_q.view(-1, self._num_heads, self._qk_embed_size_per_head)

            # compute the query-position scores
            query_pos_scores = torch.einsum("bnhx,khx->bhnk", q, rel_pos_q) # [batch_size, num_heads, query_seq_len, rel_pos_win_size]
            # compute the key-position scores
            key_pos_scores = torch.einsum("bnhx,khx->bhnk", k, rel_pos_k) # [batch_size, num_heads, key_seq_len,  rel_pos_win_size]

            # copy the last dimension of query_pos_scores for max(rel_pos_win[0] + q_seq_len - 1, 0) times
            ext_before = max(0, self.rel_pos_win[0] + q_seq_len - 1)
            query_pos_scores = torch.cat([query_pos_scores[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, ext_before), query_pos_scores], dim=-1) # [batch_size, num_heads, query_seq_len, rel_pos_win_size + ext_before]
            key_pos_scores = torch.cat([key_pos_scores[:, :, :, 0].unsqueeze(-1).repeat(1, 1, 1, ext_before), key_pos_scores], dim=-1) # [batch_size, num_heads, key_seq_len, rel_pos_win_size + ext_before]

            # copy the last row of query_pos_scores for max(k_seq_len - 1 - rel_pos_win[1], 0) times
            ext_after = max(0, k_seq_len - self.rel_pos_win[1])
            query_pos_scores = torch.cat([query_pos_scores, query_pos_scores[:, :, :, -1].unsqueeze(-1).repeat(1, 1, 1, max(0, k_seq_len - self.rel_pos_win[1]))], dim=-1) # [batch_size, num_heads, query_seq_len, rel_pos_win_size + ext_before + ext_after]
            key_pos_scores = torch.cat([key_pos_scores, key_pos_scores[:, :, :, -1].unsqueeze(-1).repeat(1, 1, 1, max(0, k_seq_len - self.rel_pos_win[1]))], dim=-1) # [batch_size, num_heads, key_seq_len, rel_pos_win_size + ext_before + ext_after]

            # flip the last dimension of query_pos_scores only 
            query_pos_scores = torch.flip(query_pos_scores, dims=[-1]) # [batch_size, num_heads, query_seq_len, rel_pos_win_size + ext_before + ext_after]

            # reduce the last two dimensions of query_pos_scores and key_pos_scores to one dimension
            query_pos_scores = query_pos_scores.reshape(-1, -1, query_pos_scores.shape[-2] * query_pos_scores.shape[-1]) # [batch_size, num_heads, query_seq_len * (rel_pos_win_size + ext_before + ext_after)]
            key_pos_scores = key_pos_scores.reshape(-1, -1, key_pos_scores.shape[-2] * key_pos_scores.shape[-1]) # [batch_size, num_heads, key_seq_len * (rel_pos_win_size + ext_before + ext_after)]

            # add q_seq_len number of zeros at the end of the last dimension of query_pos_scores
            query_pos_scores = torch.cat([query_pos_scores, torch.zeros(query_pos_scores.shape[0], query_pos_scores.shape[1], q_seq_len, device=query_pos_scores.device)], dim=-1) # [batch_size, num_heads, query_seq_len * (rel_pos_win_size + ext_before + ext_after + 1)]
            # reshape the query_pos_scores to 4D tensor
            query_pos_scores = query_pos_scores.view(-1, self._num_heads, q_seq_len, self.rel_pos_win_size + ext_before + ext_after + 1)
            # only keep the first k_seq_len elements of query_pos_scores in the last dimension
            query_pos_scores = query_pos_scores[:, :, :, :k_seq_len]


            # add k_seq_len number of zeros at the end of the last dimension of key_pos_scores
            key_pos_scores = torch.cat([key_pos_scores, torch.zeros(key_pos_scores.shape[0], key_pos_scores.shape[1], k_seq_len, device=key_pos_scores.device)], dim=-1) # [batch_size, num_heads, key_seq_len * (rel_pos_win_size + ext_before + ext_after + 1)]
            # reshape the key_pos_scores to 4D tensor
            key_pos_scores = key_pos_scores.view(-1, self._num_heads, k_seq_len, self.rel_pos_win_size + ext_before + ext_after + 1)
            # only keep the first q_seq_len elements of key_pos_scores in the last dimension
            key_pos_scores = key_pos_scores[:, :, :, :q_seq_len]

            # flip the last dimension of query_pos_scores
            query_pos_scores = torch.flip(query_pos_scores, dims=[-1]) # [batch_size, num_heads, query_seq_len, k_seq_len]
            # transpose the last two dimensions of key_pos_scores
            key_pos_scores = key_pos_scores.transpose(-1, -2) # [batch_size, num_heads, query_seq_len, k_seq_len]
        else: 
            query_pos_scores = 0
            key_pos_scores = 0
        
        # Compute attention weights.
        logits = (torch.einsum("bnhk,bmhk->bhnm", q, k) + query_pos_scores + key_pos_scores) * self._qk_embed_size_per_head ** (-0.5) # [batch_size, num_heads, query_seq_len, key_seq_len]
        if logits_shift is not None:
            logits = logits + logits_shift

        if mask is not None:
            logits.masked_fill_(mask == 0.0, float("-inf"))
        if self.use_causal_attn:
            logits = torch.triu(torch.ones_like(logits) * (- 1.0e12), diagonal=1) + logits
            pass
        
        weights = self.attention_activation(logits)

        if mask is not None:
            weights.masked_fill_(mask == 0, 0.0)
        if self.use_causal_attn:
            weights = torch.tril(torch.ones_like(weights), diagonal=0) * weights
            pass

        # Apply attention weights dropout.
        weights = self.dropout(weights)
        o = torch.einsum("bhnm,bmhk->bnhk", weights, v) # [batch_size, query_seq_len, num_heads, vo_embed_size_per_head]
        # Reshape to 3D tensor.
        o = torch.reshape(o, (-1, o.shape[1], self._vo_embed_size_per_head * self._num_heads)) # [batch_size, query_seq_len, vo_embed_size]

        # Linear projection for outputs.
        o = self.o_proj(o) # [batch_size, query_seq_len, o_dim]

        return o, weights
    
    def __view__(self, k_win=None, q_win=None, v_win=None, o_win=None):
        """
        Get the attention weights.
        Here qk_effect_weights = [qk_effect_weight_1, ..., qk_effect_weight_num_heads], where qk_effect_weight_i = q_proj_weights[i].T @ k_proj_weights[i] * qk_embed_size_per_head ** -.5;
        ov_effect_weights = [ov_effect_weight_1, ..., ov_effect_weight_num_heads], where ov_effect_weight_i = o_proj_weights[i] @ v_proj_weights[i];

        Args:
            query (torch.Tensor): Query tensor 
                of shape [batch_size, query_seq_len, q_dim].
            key (torch.Tensor): Key tensor
                of shape [batch_size, enc_seq_len, k_dim].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, x_seq_len, enc_seq_len].

        Returns:
            torch.Tensor: A 4D tensor of shape [batch_size, num_heads, query_seq_len, key_seq_len].
        """

        # Linear projections for queries and keys.
        k_proj_weights = self.k_proj.weight.data if self.q_k_v_o_proj_enabled[1] else torch.eye(self.k_dim).repeat(self._num_heads, 1)
        q_proj_weights = self.q_proj.weight.data if self.q_k_v_o_proj_enabled[0] else torch.eye(self.q_dim).repeat(self._num_heads, 1)

        if k_win is not None:
            k_proj_weights = k_proj_weights * k_win.to(k_proj_weights.device)
        if q_win is not None:
            q_proj_weights = q_proj_weights * q_win.to(q_proj_weights.device)

        # split the weights into num_heads using torch.view method
        k_proj_weights = k_proj_weights.view(self._num_heads, self._qk_embed_size_per_head, self.k_dim) # shape: (num_heads, qk_size_split, k_dim)
        q_proj_weights = q_proj_weights.view(self._num_heads, self._qk_embed_size_per_head, self.q_dim) # shape: (num_heads, qk_size_split, q_dim)

        # compute the attention weights
        kq_effect_weights = torch.einsum("hdk,hdq->hkq", k_proj_weights, q_proj_weights) * self._qk_embed_size_per_head ** -.5 # shape: (num_heads, k_dim, q_dim)

        v_proj_weights = self.v_proj.weight.data if self.q_k_v_o_proj_enabled[2] else torch.eye(self.v_dim).repeat(self._num_heads, 1)
        o_proj_weights = self.o_proj.weight.data if self.q_k_v_o_proj_enabled[3] else torch.eye(self.o_dim).repeat(1, self._num_heads)

        if v_win is not None:
            v_proj_weights = v_proj_weights * v_win.to(v_proj_weights.device)
        if o_win is not None:
            o_proj_weights = o_proj_weights * o_win.to(o_proj_weights.device)

        # split the weights into num_heads
        v_proj_weights = v_proj_weights.view(self._num_heads, self._vo_embed_size_per_head, self.v_dim)  # shape: (num_heads, vo_size_per_head, v_dim)
        o_proj_weights = o_proj_weights.view(self.o_dim, self._num_heads, self._vo_embed_size_per_head).transpose(1, 0)  # shape: (num_heads, o_dim, vo_size_per_head)

        # compute the output weights
        ov_effect_weights = torch.einsum("hod,hdv->hov", o_proj_weights, v_proj_weights) # shape: (num_heads, o_dim, v_dim)

        # return kq_effect_weights, ov_effect_weights, q_proj_weights, k_proj_weights, v_proj_weights, o_proj_weights
        return {
            "kq_effect_weights": kq_effect_weights,# shape: (num_heads, k_dim, q_dim)
            "ov_effect_weights": ov_effect_weights,# shape: (num_heads, o_dim, v_dim)
            "q_proj_weights": q_proj_weights,   # shape: (num_heads, qk_size_split, q_dim)
            "k_proj_weights": k_proj_weights,
            "v_proj_weights": v_proj_weights,   # shape: (num_heads, vo_size_per_head, v_dim)
            "o_proj_weights": o_proj_weights
        }



class SimplifiedLayerNorm(nn.Module):
    def __init__(self, dim=-1, eps=1e-7):
        super(SimplifiedLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, x):
        norm = torch.norm(x, dim=self.dim, keepdim=True)
        out = x / (norm + self.eps)
        return out

class SimplifiedRelativePositionalEmbedding(nn.Module):
    def __init__(self, T, n_parent, H, w_plus, w_minus):
        super(SimplifiedRelativePositionalEmbedding, self).__init__()
        self.T = T
        self.H = H
        self.W = torch.ones((self.T,self.H)) * w_minus
        torch.diagonal(self.W, 0).fill_(w_plus)
        self.W = nn.Parameter(self.W)
        self.norm = SimplifiedLayerNorm(dim=-1)
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
            v_h = self.norm(v_h)
            V = torch.cat([V, v_h.clone()], dim=-1)
        V = V.to(X.device)
        return V




class PolyKernelMultiHeadAttention(MultiHeadAttention):
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
            init_method: Initialization method for the C_alpha_list.
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

        # initialize the C_alpha_list
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
        if "low_degree" in kwargs:
            if kwargs["low_degree"] != -1:
                low_degree = torch.ones(1) * kwargs["low_degree"]
                row_sum = torch.sum(self.degrees,1)
                selection = (row_sum<=low_degree)
                self.C_alpha_list.data = self.C_alpha_list.data[:,selection]
                self.degrees = self.degrees[selection]
        print(self.degrees)


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
        logits_shift = logits_shift / self.C_alpha_list.norm(dim=-1, keepdim=True) ** 2
        logits_shift = logits_shift / self.C_alpha_list.norm(dim=-1, keepdim=True) ** 2
        # layer normalization
        
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
                c_alpha_init,
                n_parent,
                low_degree=-1):
        super(TwoLayerTransformer, self).__init__()
        self.T = seq_length
        self.H = num_heads
        self.d = vocab_size
        self.n_parent = n_parent
        # layer 1: attention
        self.layer1 = SimplifiedRelativePositionalEmbedding(
            T=self.T, 
            n_parent=self.n_parent,
            H=self.H, 
            w_plus=w_plus, 
            w_minus=w_minus
        )
        # layer 2: attention
        self.layer2 = PolyKernelMultiHeadAttention(
            num_heads=1,
            num_components=self.H+1,
            dimension=self.d,
            max_individual_degree=1,
            init_method="ones",
            a = "learnable",
            a_init = a_init,
            low_degree = low_degree
        )
        # init params
        self.layer2.C_alpha_list.data = torch.ones_like(self.layer2.C_alpha_list.data) * c_alpha_init

    def forward(self, X):
        X = self.layer1(X) # [bs, T+1, d*(H+1)]
        assert X.shape[1] == self.T+1 and X.shape[2] == self.d * (1+self.H)
        X = X.view(X.shape[0], X.shape[1], -1)
        q, k, v = X[..., -1:, :], X[..., :-1, :], X[..., :-1, 0:self.d]
        X = self.layer2(q,k,v)
        return X
  

def population_loss(ignore_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
    return criterion