import torch
import torch.nn as nn
import torch.nn.functional as F





class SimplifiedLayerNorm(nn.Module):
    def __init__(self, dim=-1, eps=1e-7):
        super(SimplifiedLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
    def forward(self, x):
        norm = torch.norm(x, dim=self.dim, keepdim=True)
        out = x / (norm + self.eps)
        return out

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
        self.T = seq_length
        self.H = num_heads
        self.M = window_length
        # Assuming d_model is equal to input_dim
        self.d = vocab_size
        # layer 1: attention
        self.W = torch.Tensor(self.T,self.H)
        # layer 1: norm
        self.norm = SimplifiedLayerNorm(dim=-1)
        # layer mlp
        self.alpha_list = torch.tensor([[int(i) for i in format(num, f'0{self.H+1}b')] for num in range(2**(self.H+1))])
        self.C_alpha_list = nn.Parameter(torch.zeros(2**(self.H+1)))
        self.S_alpha_list = torch.tensor([torch.nonzero(alpha, as_tuple=True)[0]] for alpha in self.alpha_list)
        # layer 2: attention
        self.a = a
             
    def init(self, w_plus, w_minus, a):
        # for W
        nn.init.constant_(self.W, w_minus)
        self.W.diagonal()[:self.M].fill_(w_plus) 

    def forward(self, x):
        # x: (bs, T, d)
        # First layer
        # concatenate an all-zero vecgor as the last row of X
        x = torch.cat([x, torch.zeros((x.size(0), 1, x.size(2)))], dim=1) # (bs, T+1, d)
        V = []
        for i in range(self.H):
            W_h = torch.zeros((self.T+1, self.T+1)).to(x.device) # L*L
            for j in range(self.T):
                torch.diagonal(W_h, -j).fill_(self.W[:, i][j])  # Set the (j+1)-th negative diagonal
            mask = torch.triu(torch.full_like(W_h, float('-inf')), diagonal=1)
            W_h = W_h + mask
            W_h = F.softmax(W_h, dim=-1)
            v_h = torch.matmul(W_h, x) # [T+1, T+1], [bs, T+1, d] -> [bs, T+1, d]
            v_h_normalized = self.norm(v_h)
            V.append(v_h_normalized)
        V.append(x) # [bs, T+1, d, (H+1)]
        V = torch.stack(V) # [bs, T+1, d, (H+1)]
        V = self.norm(V) # [bs, T+1, d, (H+1)]

    def polynomial_kernel(self, v_t, v_t_prime):
        '''
        v_t: [bs, d, (H+1)]
        v_t_prime: [bs, d, (H+1)]
        '''
        # Initialize the kernel result
        kernel_result = 0.0
        
        # For each binary vector alpha, compute the product term
        for idx, (S_alpha, C_alpha) in enumerate(zip(self.S_alpha_list, self.C_alpha_list)):
            product_term = torch.prod(torch.stack([torch.dot(v_t[...,h], v_t_prime[..., h]) for h in S_alpha]))
            kernel_result += C_alpha**2 * product_term
            
        return kernel_result


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

test_model()