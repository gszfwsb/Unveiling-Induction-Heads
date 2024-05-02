from typing import Optional, Union, Literal, Any
import torch
import torch.nn as nn
import math


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
    

class FFN(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
            self,
            hidden_size: int,
            expansion_factor: int,
            use_bias: bool = False,
            dropout_rate: float = 0.0,
    ):
        """
        Initialize the FeedForwardNetwork module.

        Args:
            hidden_size (int): The dimensionality of the input embeddings.
            expansion_factor (int): The factor by which to expand 
                the dimensionality in the hidden layer.
            use_bias (bool, optional): Whether to use bias in the linear layers. 
                Defaults to False.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()

        # Define the inner size of the hidden layer.
        inner_size = hidden_size * expansion_factor
        # Initialization of the linear layers.
        self.dense1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=use_bias,
        )
        self.dense2 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=use_bias,
        )
        self.activation = nn.functional.gelu
        self.dropout = nn.Dropout(p=dropout_rate)

        # Initialize the weights.
        self.dense1.weight.data.normal_(mean=0.0, std=0.02)
        self.dense2.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        """
        Apply a position-wise feed-forward network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, seq_len, hidden_size].
        """
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        return x
    
    def __view__(self):
        return {
            "dense1_weight": self.dense1.weight.data,
            "dense2_weight": self.dense2.weight.data
        }


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)
    
    def __view__(self):
        return {
            "dense1_weight": self.layers[0].weight.data,
            "dense2_weight": self.layers[2].weight.data
        }
    

# Write a nn that fetch the last coordinate of the last token as the prediction.
class ReadOut(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 use_dense_outlayer: bool = False, 
                 activation_type: Literal["softmax", "relu", "linear"] = "linear", 
                 seq_msk: Optional[torch.Tensor] = None, # 1-dimensional mask only for the sequence length
                 ):
        super().__init__()
        self.use_dense_outlayer = use_dense_outlayer
        if self.use_dense_outlayer:
            self.ff = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim
        self.seq_msk = torch.tensor(seq_msk) if seq_msk is not None else None
        self.activation_type = activation_type

    def forward(self, x):
        if self.use_dense_outlayer:
            y_hat = self.ff(x[..., self.seq_msk, :]) if self.msk is not None else self.ff(x)
        else:
            y_hat = x[..., self.seq_msk, - self.out_dim:] if self.seq_msk is not None else x[..., - self.out_dim:]
        # apply activation function
        if self.activation_type == "softmax":
            y_hat = nn.functional.softmax(y_hat, dim=-1)
        elif self.activation_type == "relu":
            y_hat = nn.functional.relu(y_hat)
        elif self.activation_type == "linear":
            pass
        return y_hat.squeeze()


class ReadIn(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 use_dense_inlayer: bool = False, 
                 ):
        super().__init__()
        self.use_dense_inlayer = use_dense_inlayer
        if self.use_dense_inlayer:
            self.ff = nn.Linear(in_dim, out_dim)
        else:
            self.ff = nn.Identity()
        self.out_dim = out_dim

    def forward(self, x):
        return self.ff(x)


class PositionalEncoding(nn.Module):
    """Implementation of positional encoding."""
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.1, 
                 max_len: int = 5000, 
                 _type: Literal["NoPE", "RoPECat", "RoPEAdd", "OrthCat", "OrthAdd"] = "NoPE",
                 ):
        """
        Initialize the PositionalEncoding module.
        
        If _type is "NoPE", the module will not add any positional encoding to the input tensor, and in the forward pass, it will return the input tensor directly.
        
        If _type is "RoPEAdd" or "OrthAdd", the module will add positional encoding to the input tensor, and in the forward pass, it will return the input tensor with positional encoding added.

        If _type is "RoPECat" or "OrthCat", the module will concatenate positional encoding to the input tensor, and in the forward pass, it will return the input tensor with positional encoding concatenated.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            max_len (int, optional): The maximum sequence length. Defaults to 5000.
            _type (Literal["NoPE", "RoPECat", "RoPEAdd", "OrthCat", "OrthAdd"], optional): The type of positional encoding to use. Defaults to "NoPE".

        Raises:
            ValueError: If _type is not one of "NoPE", "RoPECat", "RoPEAdd", "OrthCat", "OrthAdd".
        """
        super().__init__()
        self._type = _type

        if self._type == "NoPE":
            self.dropout = nn.Identity()
            pre = None
            self.register_buffer('pe', pre)

        elif self._type == "RoPECat" or self._type == "RoPEAdd":
            self.dropout = nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        
        elif self._type == "OrthCat" or self._type == "OrthAdd":
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            for pos in range(max_len):
                pe[pos, pos % d_model] = 1
            self.register_buffer('pe', pe)

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        if self._type == "NoPE":
            return x
        elif self._type == "RoPEAdd" or self._type == "OrthAdd":
            x = x + self.pe[:x.size(1)]
        elif self._type == "RoPECat" or self._type == "OrthCat":
            # concatenate on the embedding dimension
            pe = self.pe[:x.size(1)]
            # extend pe to the same shape as x except for the last dimension
            pe = pe.unsqueeze(0).expand(x.size(0), -1, -1)
            # concatenate
            x = torch.cat((x, pe), dim=-1)
        return self.dropout(x)
    
    def __pos_enc_size__ (self) -> int:
        if self._type == "NoPE":
            return 0
        elif self._type == "RoPEAdd" or self._type == "OrthAdd":
            return 0
        elif self._type == "RoPECat" or self._type == "OrthCat":
            return self.d_model

    def __view__(self):
        return {
            "positional_encoding": self.pe
        }

class TransformerBlock(nn.Module):
    """Implementation of a Transformer block.
        The graph of a Transformer block is as follows:
        [Input (x)]
            |
            v
        [Multi-Head Self-Attention]
            |
            v
        [Layer Normalization]
            |
            v
        [Residual Connection]
            |
            v
        [Feed-Forward Neural Network]
            |
            v
        [Layer Normalization]
            |
            v
        [Residual Connection]
            |
            v
        [Final Output]
    """

    def __init__(
            self,
            num_heads: int,
            hidden_size: int,
            attention_type: Literal["softmax", "relu", "linear"],
            expansion_factor: int,
            mlp_hidden_size: int,
            input_size: int = None,
            output_size: int = None,
            use_bias: bool = False,
            use_ffn: bool = True,
            use_pre_mlps: bool = False,
            use_layer_norm: bool = False,
            use_pre_norm: bool = True,
            allow_shortcut: bool = True,
            concat_shortcut: bool = False,
            dropout_rate: float = 0.0,
            q_k_v_o_proj_enabled: list = [True, True, True, True],
            attn_init_method: Literal["normal", "small identity", None] = "normal", 
            pos_enc_type: Literal["NoPE", "RoPECat", "RoPEAdd", "OrthCat", "OrthAdd"] = "NoPE",
            max_seq_len: int = 5000,
            pos_enc_size: int = 8,
            use_rel_pos: bool = False,
            use_rel_pos_proj: bool = False,
            rel_pos_win: tuple = (-32, 32),
            rel_pos_embed_size: int = 8,
            shared_rel_pos_embed: bool = True,
            use_causal_attn: bool = False,
    ):
        """
        Initialize the TransformerBlock module.

        Args:
            hidden_size (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            attention_type (str): The type of attention mechanism 
                to use ('softmax', 'relu', or 'linear').
            expansion_factor (int): The factor by which to expand 
                the dimensionality in the hidden layer.
            use_bias (bool, optional): Whether to use bias in the linear layers. 
                Defaults to False.
            use_ffn (bool, optional): Whether to use FFN in a Transformer block.
                Defaults to False.
            use_layer_norm (bool, optional): Whether to use layer normalization.
                Defaults to False.
            use_pre_norm (bool, optional): Whether to add pre-layer normalization into the shortcut.
                Defaults to True.
            allow_shortcut (bool, optional): Whether to allow the residual connection.
                Defaults to True.
            concat_shortcut (bool, optional): Whether to concatenate the shortcut.
                Defaults to False.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
            q_k_v_o_proj_enabled (list, optional): Whether to enable projection. Defaults to [True, True, True, True].
            pos_enc_type (Literal["NoPE", "RoPECat", "RoPEAdd", "OrthCat", "OrthAdd"], optional): The type of positional encoding to use. Defaults to "NoPE".
            max_seq_len (int, optional): The maximum sequence length. Defaults to 5000.
            pos_enc_size (int, optional): The dimensionality of the positional encoding. Defaults to 8.
            use_rel_pos (bool, optional): Whether to use relative positional encoding. Defaults to False.
            use_rel_pos_proj (bool, optional): Whether to use projection for relative positional encoding. Defaults to False.
            rel_pos_win (tuple, optional): The relative positional window. Defaults to (-32, 32).
            rel_pos_embed_size (int, optional): The dimensionality of the relative positional encoding. Defaults to 8.
            use_causal_attn (bool, optional): Whether to use causal attention. Defaults to False.
        """
        super().__init__()
        self._use_pre_norm = use_pre_norm
        self._use_ffn = use_ffn
        self._allow_shortcut = allow_shortcut
        self._concat_shortcut = concat_shortcut
        self._use_pre_mlps = use_pre_mlps
        

        input_size = input_size if input_size is not None else hidden_size

        output_size = output_size if output_size is not None else input_size

        
        self.pre_mlp = MLP(input_size, mlp_hidden_size, input_size) if self._use_pre_mlps else nn.Identity()

        self.pos_encoder = PositionalEncoding(
            d_model=pos_enc_size,
            dropout=dropout_rate,
            max_len=max_seq_len,
            _type=pos_enc_type,
        )
        pos_enc_size = self.pos_encoder.__pos_enc_size__()
        
        if self._concat_shortcut and self._allow_shortcut:
            assert output_size - input_size > 0, "Output size must be greater than input size for concatenation of the shortcut!"
        if self._allow_shortcut and not self._concat_shortcut:
            assert output_size == input_size, "Output size must be equal to input size for the residual connection without concatenation!"

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            q_dim=input_size + pos_enc_size,
            o_dim=output_size - input_size if self._concat_shortcut and self._allow_shortcut else output_size,
            qk_embed_size_per_head= hidden_size // num_heads, 
            ov_embed_size_per_head= hidden_size // num_heads,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            q_k_v_o_proj_enabled=q_k_v_o_proj_enabled,
            initialization_method=attn_init_method,
            use_rel_pos=use_rel_pos,
            use_rel_pos_proj=use_rel_pos_proj,
            rel_pos_win=rel_pos_win,
            rel_pos_embed_size=rel_pos_embed_size,
            use_causal_attn=use_causal_attn,
        )
        
        self.ffn = FFN(
                hidden_size=output_size,
                expansion_factor=expansion_factor,
                use_bias=use_bias,
                dropout_rate=dropout_rate,
            ) if self._use_ffn else nn.Identity()
        
        # Initialization of layer normalization layers.
        self.norm_attention = nn.LayerNorm(input_size) if use_layer_norm else nn.Identity()
        self.norm_ffn = nn.LayerNorm(output_size) if use_layer_norm and use_ffn else nn.Identity()
        self.norm_pre_mlp = nn.LayerNorm(input_size) if use_layer_norm and use_pre_mlps else nn.Identity()

        # Initialization of dropout layers.
        self.dropout_attention = nn.Dropout(p=dropout_rate)
        self.dropout_ffn = nn.Dropout(p=dropout_rate) if use_ffn else nn.Identity()
        self.dropout_pre_mlp = nn.Dropout(p=dropout_rate) if use_pre_mlps else nn.Identity()

        self.shared_rel_pos_embed = shared_rel_pos_embed
        self.use_rel_pos = use_rel_pos
        if not self.shared_rel_pos_embed and self.use_rel_pos:
            self.rel_pos_embedding = nn.Parameter(torch.randn(rel_pos_win[1] - rel_pos_win[0], rel_pos_embed_size))

    def forward(
        self, x: torch.Tensor,
        rel_pos_embedding: Optional[nn.parameter.Parameter] = None,
        mask: Optional[Union[torch.Tensor, None]] = None,
    ):
        """
        Apply a Transformer block to the input tensor.

        Args:
            x (torch.Tensor): A 3D tensor of shape [batch_size, seq_len, input_size].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, seq_len, hidden_size].
        """
        if self._use_pre_mlps:
            x_norm = self.norm_attention(x)
            if self._allow_shortcut:
                shortcut = x if self._use_pre_norm else x_norm
            else:
                shortcut = 0
            x = self.pre_mlp(x_norm)
            x = self.dropout_pre_mlp(x)
            x += shortcut

        # Apply multi-head self-attention.
        x_norm = self.norm_attention(x)
        if self._allow_shortcut:
            shortcut = x if self._use_pre_norm else x_norm
        else:
            shortcut = 0
        x_posenc = self.pos_encoder(x_norm)
        
        if self.use_rel_pos:
            if self.shared_rel_pos_embed:
                x, _ = self.attention(x_posenc, x_posenc, x_posenc, rel_pos_embedding, mask)
            else:
                x, _ = self.attention(x_posenc, x_posenc, x_posenc, self.rel_pos_embedding, mask)
        else:
            x, _ = self.attention(x_posenc, x_posenc, x_posenc, None, mask)

        x = self.dropout_attention(x)
        if self._allow_shortcut:
            x = x + shortcut if not self._concat_shortcut else torch.cat([x, shortcut], dim=-1)
        else:
            pass

        if self._use_ffn:
            x_norm = self.norm_ffn(x)
            if self._allow_shortcut:
                shortcut = x if self._use_pre_norm else x_norm
            else:
                shortcut = 0
            x = self.ffn(x_norm)
            x = self.dropout_ffn(x)
            x += shortcut

        return x

class TransformerEncoder(nn.Module):
    """Implementation of Transformer encoder."""

    def __init__(
            self,
            **kwargs: Any,
    ):
        """
        Initialize the TransformerEncoder module.

        Args:
            hidden_size (int): The dimensionality of the input embeddings.
            num_layers (int): The number of layers in the Transformer encoder.
            num_heads (int): The number of attention heads.
            attention_type (str): The type of attention mechanism 
                to use ('softmax', 'relu', or 'linear').
            expansion_factor (int): The factor by which to expand 
                the dimensionality in the hidden layer.
            mlp_hidden_size (int): The size of the hidden layer in the FFN.
            use_bias (bool, optional): Whether to use bias in the linear layers. 
                Defaults to False.
            use_ffn (bool, optional): Whether to use FFN in a Transformer block.
                Defaults to False.
            use_layer_norm (bool, optional): Whether to use layer normalization.
                Defaults to False.
            use_pre_norm (bool, optional): Whether to add pre-layer normalization into the shortcut.
                Defaults to True.
            allow_shortcut (bool, optional): Whether to allow the residual connection. Defaults to True.
            concat_shortcut (bool, optional): Whether to concatenate the shortcut. Defaults to False.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
            q_k_v_o_proj_enabled (list, optional): Whether to enable projection. Defaults to [True, True, True, True].
            attn_init_method (str, optional): Initialization method for attention weights. 
                Defaults to "normal".
            pos_enc_type (Literal["NoPE", "RoPECat", "RoPEAdd", "OrthCat", "OrthAdd"], optional): The type of positional encoding to use. Defaults to "NoPE".
            max_seq_len (int, optional): The maximum sequence length. Defaults to 5000.
            pos_enc_size (int, optional): The size of the positional encoding. Defaults to 8.
            input_size (int, optional): The size of the input tensor. Defaults to None.
            output_size (int, optional): The size of the output tensor. Defaults to None.
            
            use_rel_pos (bool, optional): Whether to use relative positional encoding. Defaults to False.
            use_rel_pos_proj (bool, optional): Whether to use projection for relative positional encoding. Defaults to False.
            rel_pos_win (tuple, optional): The relative positional window. Defaults to (-32, 32).
            rel_pos_embed_size (int, optional): The size of the relative positional encoding. Defaults to 8.
            shared_rel_pos_embed (bool, optional): Whether to share the relative positional embedding. Defaults to True.

            use_causal_attn (bool, optional): Whether to use causal attention. Defaults to False.
        """
        super().__init__()
        self.hparams_dict = kwargs
        self.num_layers = kwargs.get("num_layers", 6)
        
        
        hidden_size = self.__hparams_formater__("hidden_size", 32)
        num_heads = self.__hparams_formater__("num_heads", 8)
        attention_type = self.__hparams_formater__("attention_type", "softmax")
        expansion_factor = self.__hparams_formater__("expansion_factor", 4)
        mlp_hidden_size = self.__hparams_formater__("mlp_hidden_size", 2048)
        use_bias = self.__hparams_formater__("use_bias", False)
        use_ffn = self.__hparams_formater__("use_ffn", True)
        
        use_pre_mlps = self.__hparams_formater__("use_pre_mlps", False)
        use_layer_norm = self.__hparams_formater__("use_layer_norm", True)
        use_pre_norm = self.__hparams_formater__("use_pre_norm", True)
        allow_shortcut = self.__hparams_formater__("allow_shortcut", True)
        concat_shortcut = self.__hparams_formater__("concat_shortcut", False)
        dropout_rate = self.__hparams_formater__("dropout_rate", 0.0)
        q_k_v_o_proj_enabled = self.__hparams_formater__("q_k_v_o_proj_enabled", (True, True, True, True))

        attn_init_method = self.__hparams_formater__("attn_init_method", "normal")
        pos_enc_type = self.__hparams_formater__("pos_enc_type", "RoPECat")
        max_seq_len = self.__hparams_formater__("max_seq_len", 512)
        pos_enc_size = self.__hparams_formater__("pos_enc_size", 8)

        input_size = self.__hparams_formater__("input_size", None)
        output_size = self.__hparams_formater__("output_size", None)

        use_rel_pos = self.__hparams_formater__("use_rel_pos", False)
        use_rel_pos_proj = self.__hparams_formater__("use_rel_pos_proj", False)
        rel_pos_win = self.__hparams_formater__("rel_pos_win", (-32, 32))
        rel_pos_embed_size = self.__hparams_formater__("rel_pos_embed_size", 8)
        shared_rel_pos_embed = self.hparams_dict.get("shared_rel_pos_embed", True)

        # if an item in use_rel_pos is False, set the corresponding items in use_rel_pos_proj to True, in rel_pos_win to None, and in rel_pos_embed_size to None
        for i in range(self.num_layers):
            if not use_rel_pos[i]:
                use_rel_pos_proj[i] = True
                rel_pos_win[i] = None
                rel_pos_embed_size[i] = None

        use_causal_attn = self.__hparams_formater__("use_causal_attn", False)

        self.hparams_dict_full = {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "attention_type": attention_type,
            "expansion_factor": expansion_factor,
            "mlp_hidden_size": mlp_hidden_size,
            "use_bias": use_bias,
            "use_ffn": use_ffn,
            "use_pre_mlps": use_pre_mlps,
            "use_layer_norm": use_layer_norm,
            "use_pre_norm": use_pre_norm,
            "allow_shortcut": allow_shortcut,
            "concat_shortcut": concat_shortcut,
            "dropout_rate": dropout_rate,
            "q_k_v_o_proj_enabled": q_k_v_o_proj_enabled,
            "attn_init_method": attn_init_method,
            "pos_enc_type": pos_enc_type,
            "max_seq_len": max_seq_len,
            "pos_enc_size": pos_enc_size,
            "input_size": input_size,
            "output_size": output_size, 
            "use_rel_pos": use_rel_pos,
            "use_rel_pos_proj": use_rel_pos_proj,
            "rel_pos_win": rel_pos_win,
            "rel_pos_embed_size": rel_pos_embed_size,
            "shared_rel_pos_embed": shared_rel_pos_embed, # bool
            "use_causal_attn": use_causal_attn,
        }
        self._use_ffn = use_ffn
        self._use_pre_mlps = use_pre_mlps

        self.blocks = nn.ModuleList()
        if shared_rel_pos_embed:
            # assert that each item in rel_pos_win is the same tuple for all layers
            assert all([rel_pos_win_item == rel_pos_win[0] or rel_pos_win_item == None for rel_pos_win_item in rel_pos_win]), "The rel_pos_win must be the same for all layers if shared_rel_pos_embed is True!"
            # assert that each item in rel_pos_embed_size is the same for all layers
            assert all([rel_pos_embed_size_item == rel_pos_embed_size[0] or rel_pos_embed_size_item == None for rel_pos_embed_size_item in rel_pos_embed_size]), "The rel_pos_embed_size must be the same for all layers if shared_rel_pos_embed is True!"

            # check if all items in use_rel_pos is False
            if all([not use_rel_pos_item for use_rel_pos_item in use_rel_pos]):
                self.rel_pos_param_shared = None
            else:
                # get the first layer index where use_rel_pos is True
                layer_id = use_rel_pos.index(True)
                self.rel_pos_param_shared = nn.Parameter(torch.randn(rel_pos_win[layer_id][1] - rel_pos_win[layer_id][0], rel_pos_embed_size[layer_id]))
        else:
            self.rel_pos_param_shared = None

        for layer_id in range(self.num_layers):
            self.blocks.append(TransformerBlock(
                hidden_size=hidden_size[layer_id],
                num_heads=num_heads[layer_id],
                attention_type=attention_type[layer_id],
                expansion_factor=expansion_factor[layer_id],
                mlp_hidden_size=mlp_hidden_size[layer_id],
                use_bias=use_bias[layer_id],
                use_ffn=use_ffn[layer_id],
                use_pre_mlps=use_pre_mlps[layer_id],
                use_layer_norm=use_layer_norm[layer_id],
                use_pre_norm=use_pre_norm[layer_id],
                allow_shortcut=allow_shortcut[layer_id],
                concat_shortcut=concat_shortcut[layer_id],
                dropout_rate=dropout_rate[layer_id],
                q_k_v_o_proj_enabled=q_k_v_o_proj_enabled[layer_id],
                attn_init_method=attn_init_method[layer_id],
                pos_enc_type=pos_enc_type[layer_id],
                max_seq_len=max_seq_len[layer_id],
                pos_enc_size=pos_enc_size[layer_id],
                input_size=input_size[layer_id],
                output_size=output_size[layer_id],
                use_rel_pos=use_rel_pos[layer_id],
                use_rel_pos_proj=use_rel_pos_proj[layer_id],
                rel_pos_win=rel_pos_win[layer_id],
                rel_pos_embed_size=rel_pos_embed_size[layer_id],
                shared_rel_pos_embed=shared_rel_pos_embed,
                use_causal_attn=use_causal_attn[layer_id],
            ))
        
        

    def __hparams_formater__(self, obj_name, default_value):
        if obj_name in self.hparams_dict:
            obj = self.hparams_dict[obj_name]
            # check if hidden_size is a list or a single value
            if not isinstance(obj, list):
                obj = [obj for _ in range(self.num_layers)]
            else:
                assert len(obj) == self.num_layers, "The length of hidden_size list must be equal to num_layers!"
        else:
            obj = [default_value for _ in range(self.num_layers)]
        return obj
    
    def __read_hparams__(self, obj_name):
        return self.hparams_dict_full.get(obj_name, None)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[Union[torch.Tensor, None]] = None,
    ):
        """
        Apply a Transformer Encoder to the input tensor.

        Args:
            x (torch.Tensor): A 3D tensor of shape [batch_size, seq_len, hidden_size].
            mask (Optional[Union[torch.Tensor, None]]): Optional mask tensor 
                of shape [batch_size, 1, seq_len, seq_len].

        Returns:
            torch.Tensor: A 3D tensor of shape [batch_size, seq_len, hidden_size].
        """
        for block in self.blocks:
            x = block(x, rel_pos_embedding=self.rel_pos_param_shared, mask=mask)
        return x
    
    def __view__(self, layer_id: int, k_win=None, q_win=None, v_win=None, o_win=None):
        return {
            "attention": self.blocks[layer_id].attention.__view__(k_win, q_win, v_win, o_win), 
            "positional_encoding": self.blocks[layer_id].pos_encoder.__view__(), 
            "ffn": self.blocks[layer_id].ffn.__view__() if self._use_ffn[layer_id] else None,
            "pre_mlp": self.blocks[layer_id].pre_mlp.__view__() if self._use_pre_mlps[layer_id] else None
        }



if __name__ == '__main__':
    # Test case 1: NoPE
    pos_enc = PositionalEncoding(d_model=512, _type="NoPE")
    x = torch.randn(2, 10, 512)
    output = pos_enc(x)
    assert torch.allclose(output, x)

    # Test case 2: RoPEAdd
    pos_enc = PositionalEncoding(d_model=512, _type="RoPEAdd", dropout=0.0)
    x = torch.randn(2, 10, 512)
    output = pos_enc(x)
    assert output.shape == (2, 10, 512)
    assert torch.allclose(output, x + pos_enc.pe[:x.size(1)])

    # Test case 3: RoPECat
    pos_enc = PositionalEncoding(d_model=512, _type="RoPECat", dropout=0.0)
    x = torch.randn(2, 10, 512)
    output = pos_enc(x)
    assert output.shape == (2, 10, 1024)
    assert torch.allclose(output[:, :, :512], x)
    assert torch.allclose(output[:, :, 512:], pos_enc.pe[:x.size(1)].unsqueeze(0).expand(x.size(0), -1, -1))

    # Test case 4: OrthAdd
    pos_enc = PositionalEncoding(d_model=512, _type="OrthAdd", dropout=0.0)
    x = torch.randn(2, 10, 512)
    output = pos_enc(x)
    assert output.shape == (2, 10, 512)
    assert torch.allclose(output, x + pos_enc.pe[:x.size(1)])

    # Test case 5: OrthCat
    pos_enc = PositionalEncoding(d_model=512, _type="OrthCat", dropout=0.0)
    x = torch.randn(2, 10, 512)
    output = pos_enc(x)
    assert output.shape == (2, 10, 512 * 2)
    assert torch.allclose(output[:, :, :512], x)
    assert torch.allclose(output[:, :, 512:], pos_enc.pe[:x.size(1)].unsqueeze(0).expand(x.size(0), -1, -1))


    # Test case 6: MultiHeadAttention forward pass
    attention = MultiHeadAttention(num_heads=4, q_dim=512, k_dim=512, v_dim=512, o_dim=512, qk_embed_size_per_head=128, vo_embed_size_per_head=128, attention_type="softmax", use_bias=False, dropout_rate=0.0, q_k_v_o_proj_enabled=[True, True, True, True], initialization_method="normal")
    query = torch.randn(2, 10, 512)
    key = torch.randn(2, 20, 512)
    value = torch.randn(2, 20, 512)
    output, weights = attention(query, key, value)
    assert output.shape == (2, 10, 512)
    assert weights.shape == (2, 4, 10, 20)

    # Test case 7: MultiHeadAttention with mask
    mask = torch.ones(2, 1, 10, 20)
    output, weights = attention(query, key, value, mask)
    assert output.shape == (2, 10, 512)
    assert weights.shape == (2, 4, 10, 20)# Test case 8: FFN forward pass
    ffn = FFN(hidden_size=512, expansion_factor=4, use_bias=False, dropout_rate=0.0)
    input_tensor = torch.randn(2, 10, 512)
    output = ffn(input_tensor)
    assert output.shape == (2, 10, 512)

    # Test case 9: FFN with different hidden size
    ffn = FFN(hidden_size=256, expansion_factor=2, use_bias=True, dropout_rate=0.2)
    input_tensor = torch.randn(3, 5, 256)
    output = ffn(input_tensor)
    assert output.shape == (3, 5, 256)# Test case 10: ReadOut with dense outlayer
    readout = ReadOut(in_dim=512, out_dim=256, use_dense_outlayer=True)
    input_tensor = torch.randn(2, 10, 512)
    output = readout(input_tensor)

    # Test case 11: ReadOut without dense outlayer
    readout = ReadOut(in_dim=512, out_dim=256, use_dense_outlayer=False)
    input_tensor = torch.randn(2, 10, 512)
    output = readout(input_tensor)
    # Test case 12: TransformerBlock forward pass
    transformer_block = TransformerBlock(
        num_heads=4,
        hidden_size=512,
        attention_type="softmax",
        expansion_factor=4,
        mlp_hidden_size=1024,
        input_size=512,
        output_size=512,
        use_bias=False,
        use_ffn=True,
        use_pre_mlps=False,
        use_layer_norm=False,
        use_pre_norm=True,
        allow_shortcut=True,
        concat_shortcut=False,
        dropout_rate=0.2,
        q_k_v_o_proj_enabled=[True, True, True, True],
        attn_init_method="normal",
        pos_enc_type="RoPECat",
        max_seq_len=100,
        pos_enc_size=8
    )
    input_tensor = torch.randn(2, 10, 512)
    output = transformer_block(input_tensor)
    assert output.shape == (2, 10, 512)

    # Test case 13: TransformerBlock with mask
    mask = torch.ones(2, 1, 10, 10)
    output = transformer_block(input_tensor, mask)
    assert output.shape == (2, 10, 512)# Test case 1: Default configuration
    encoder = TransformerEncoder()
    input_tensor = torch.randn(2, 10, 32)
    output = encoder(input_tensor)
    assert output.shape == (2, 10, 32)

    # Test case 2: Custom configuration
    encoder = TransformerEncoder(
        num_layers=4,
        hidden_size=64,
        num_heads=16,
        attention_type="relu",
        expansion_factor=2,
        mlp_hidden_size=512,
        use_bias=True,
        use_ffn=False,
        use_layer_norm=True,
        use_pre_norm=False,
        dropout_rate=0.2,
        q_k_v_o_proj_enabled=(True, True, True, True),
        attn_init_method=None,
        pos_enc_type="RoPECat",
        max_seq_len=256,
        pos_enc_size=16
    )
    input_tensor = torch.randn(3, 8, 64)
    output = encoder(input_tensor)
    assert output.shape == (3, 8, 64)

    # Test case 3: With mask
    encoder = TransformerEncoder()
    input_tensor = torch.randn(2, 10, 32)
    mask = torch.ones(2, 1, 10, 10)
    output = encoder(input_tensor, mask=mask)
    assert output.shape == (2, 10, 32)
