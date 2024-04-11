import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, H, L, w_plus):
        super(YourModel, self).__init__()
        self.H = H
        self.L = L
        # Initialize W as a tensor
        self.W = nn.Parameter(torch.zeros(self.H, self.L))
        print(self.W)
        
        # Fill the diagonal with the value w_plus
        self.W[torch.arange(self.H), torch.arange(self.H)] = nn.Parameter(torch.ones(self.H) * w_plus)
        print(self.W)

# Example usage
model = YourModel(H=10, L=20, w_plus=1.0)
print(model.W)
