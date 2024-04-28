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
                a_init,
                c_alpha_init):
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
        self.alpha_list = torch.tensor([
            [int(i) for i in format(num, f'0{self.H+1}b')] 
            for num in range(2**(self.H+1))
        ], dtype=int)
        # Extract nonzero indices for each binary vector
        self.S_alpha_list = [
            torch.nonzero(torch.tensor(alpha), as_tuple=True)[0]
            for alpha in self.alpha_list
        ]
        
        self.C_alpha_list = torch.Tensor(2**(self.H+1))
        # layer 2: attention
        self.a = torch.Tensor(1)
        ### init a with 0.01, init c_alpha with all 1
        self.init(w_plus, w_minus, a_init, c_alpha_init)
        # print("W:", self.W)
        # print("norm:", self.norm)
        # print("alpha_list:", self.alpha_list)
        # print("S_alpha_list:", self.S_alpha_list)
        # print("C_alpha_list:", self.C_alpha_list)
        # print("a:", self.a)
        
    def init(self, w_plus, w_minus, a_init, c_alpha_init):
        # for W
        nn.init.constant_(self.W, w_minus)
        self.W.diagonal()[:self.M].fill_(w_plus)
        # for a
        nn.init.constant_(self.a, a_init)
        # for c_alpha, set all c_alpha to c_alpha_init
        nn.init.constant_(self.C_alpha_list, c_alpha_init)
        
        self.W = nn.Parameter(self.W)
        self.a = nn.Parameter(self.a)
        self.C_alpha_list = nn.Parameter(self.C_alpha_list)
        

    def forward(self, X):
        # x: (bs, T, d)
        # First layer
        # concatenate an all-zero vecgor as the last row of X
        X_tilde = torch.cat([X, torch.zeros((X.size(0), 1, X.size(2))).to(X.device)], dim=1) # (bs, T+1, d)
        # print('X_tilde:', X_tilde.shape)
        V = []
        for i in range(self.H):
            W_h = torch.zeros((self.T+1, self.T+1)).to(X_tilde.device) # L*L
            for j in range(self.T):
                torch.diagonal(W_h, -j).fill_(self.W[:, i][j])  # Set the (j+1)-th negative diagonal
            mask = torch.triu(torch.full_like(W_h, float('-inf')), diagonal=1)
            W_h = W_h + mask
            W_h = F.softmax(W_h, dim=-1)
            v_h = torch.matmul(W_h, X_tilde) # [T+1, T+1], [bs, T+1, d] -> [bs, T+1, d]
            v_h_normalized = self.norm(v_h)
            V.append(v_h_normalized)
            # print('v_h_normalized:', v_h_normalized.shape)
        V.append(X_tilde) # [bs, T+1, d, (H+1)]
        V = torch.stack(V, -1) # [bs, T+1, d, (H+1)]
        # print('V:', V.shape)
        V = self.norm(V) # [bs, T+1, d, (H+1)]
        kernel_prod = [] # [bs, T]
        for i in range(self.T):
            kernel_prod.append(self.polynomial_kernel(V[:, -1], V[:, i])) # [bs, d, (H+1)], [bs, d, (H+1)] -> [bs]
        kernel_prod = torch.tensor(kernel_prod) # [bs, T]
        y = torch.matmul(F.softmax(self.a * kernel_prod, dim=-1), X) # [bs, T], [bs, T, d] -> [bs, d]
        return y
        
    def polynomial_kernel(self, v_t, v_t_prime):
        '''
        v_t: [bs, d, (H+1)]
        v_t_prime: [bs, d, (H+1)]
        '''
        # Initialize the kernel result
        kernel_result = 0.0
        
        # For each binary vector alpha, compute the product term
        for idx, S_alpha in enumerate(self.S_alpha_list):
            if len(S_alpha) == 0:
                continue
            product_term = torch.prod(torch.stack([torch.einsum('bd,bd->b',v_t[...,h], v_t_prime[..., h]) for h in S_alpha]))
            kernel_result += self.C_alpha_list[idx]**2 * product_term
        return kernel_result


def test_model():
    vocab_size = 10 # S
    seq_length = 20 # L
    num_heads = 3 # H
    window_length = 5 # M
    w_plus = 0.1
    w_minus = 0.05
    a_init = 0.01
    c_alpha_init = 1
    model = TwoLayerTransformer(vocab_size, seq_length, num_heads, window_length, w_plus, w_minus, a_init, c_alpha_init)
    # Test inputs
    bs = 2  # batch size
    X = torch.randn(bs, seq_length, vocab_size)
    y = model(X)
    
    


# test_model()