import torch
import torch.distributions as dist


T = 5
d = 3
n = 3
alpha = 0.3



a = torch.randn((T+1,T+1))
print(a)
print(torch.diagonal(a,-1))
print(torch.diagonal(a,-2))
                
