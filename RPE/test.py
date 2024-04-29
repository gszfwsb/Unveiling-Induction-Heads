import torch
import torch.distributions as dist


T = 20
d = 3
n = 3
alpha = 0.3



pi = torch.zeros((d ** (n-1), d))
for i in range(d ** (n-1)):
    pi[i] = dist.Dirichlet(torch.full((d,), alpha)).sample()


context = dist.Categorical(probs=torch.ones(d) / d).sample((n-1,)) # uniformly sample

# Convert a sequence context to an index for accessing the transition matrix
# This assumes that context is a tensor of indices
context_idx = 0
for i, state in enumerate(context.flip(0)):
    context_idx += state * (d ** i)

y = dist.Categorical(probs=pi[context_idx]).sample()


# print('pi',pi)
print('context',context)
print(context.flip(0))
print('context_id',context_idx)
print('y', y)
