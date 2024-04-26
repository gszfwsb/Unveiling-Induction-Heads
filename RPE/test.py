import torch

def polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list):
    '''
    v_t: [bs, d, (H+1)]
    v_t_prime: [bs, d, (H+1)]
    '''
    # Compute the dot product between v_t and v_t_prime
    dot_product = torch.matmul(v_t.transpose(-1, -2), v_t_prime)
    
    # Compute the product term for each binary vector alpha
    product_terms = torch.prod(torch.gather(dot_product, -1, S_alpha_list.unsqueeze(-1)), dim=-1)
    
    # Compute the kernel result using matrix operations
    kernel_result = torch.matmul(C_alpha_list**2, product_terms)
        
    return kernel_result


def polynomial_kernel_2(v_t, v_t_prime, S_alpha_list, C_alpha_list):
    '''
    v_t: [bs, d, (H+1)]
    v_t_prime: [bs, d, (H+1)]
    '''
    # Initialize the kernel result
    kernel_result = 0.0
    
    # For each binary vector alpha, compute the product term
    for idx, (S_alpha, C_alpha) in enumerate(zip(S_alpha_list, C_alpha_list)):
        product_term = torch.prod(torch.stack([torch.dot(v_t[...,h], v_t_prime[..., h]) for h in S_alpha]))
        kernel_result += C_alpha**2 * product_term
        
    return kernel_result

# Generate test samples
bs = 2  # batch size
d = 4   # dimensionality
H = 3   # H value
v_t = torch.randn(bs, d, H+1)
v_t_prime = torch.randn(bs, d, H+1)
S_alpha_list = torch.randint(low=0, high=H+1, size=(2**(H+1), H))
C_alpha_list = torch.randn(2**(H+1))

# Test the functions
kernel_result_1 = polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list)
kernel_result_2 = polynomial_kernel_2(v_t, v_t_prime, S_alpha_list, C_alpha_list)

print(kernel_result_1.shape, kernel_result_2.shape)

