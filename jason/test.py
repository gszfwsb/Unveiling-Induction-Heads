import torch

T = 5

A = (torch.diag(torch.ones(T-1), diagonal=-1)  
            + torch.diag(torch.ones(T-2), diagonal=-2))

print(A)


B = torch.diag(torch.ones(T-1), diagonal=-1)

print(B)