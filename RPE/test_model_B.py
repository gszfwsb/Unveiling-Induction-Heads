import torch
import numpy as np

def reference_polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list):
    '''
    Reference implementation of polynomial_kernel function
    '''
    dot_product = torch.matmul(v_t.transpose(-1, -2), v_t_prime)
    product_terms = torch.prod(torch.gather(dot_product, -1, S_alpha_list.unsqueeze(-1)), dim=-1)
    kernel_result = torch.matmul(C_alpha_list**2, product_terms)
    return kernel_result

def test_polynomial_kernel():
    # Test case 1
    v_t = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    v_t_prime = torch.tensor([[[7, 8, 9], [10, 11, 12]]])
    S_alpha_list = torch.tensor([[0, 1]])
    C_alpha_list = torch.tensor([2, 3])
    expected_output = reference_polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list)
    output = polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list)
    assert torch.allclose(output, expected_output), "Test case 1 failed"

    # Test case 2
    v_t = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    v_t_prime = torch.tensor([[[7, 8, 9], [10, 11, 12]]])
    S_alpha_list = torch.tensor([[0, 1]])
    C_alpha_list = torch.tensor([2, 3])
    expected_output = reference_polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list)
    output = polynomial_kernel(v_t, v_t_prime, S_alpha_list, C_alpha_list)
    assert torch.allclose(output, expected_output), "Test case 2 failed"

    # Add more test cases here...

test_polynomial_kernel()