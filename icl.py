import torch


def sample_function(S):
    # Sample a new function f: [S] -> [S] by creating a random permutation
    return torch.randperm(S)

def generate_prompt(f, n, S, batch_size=1):
    # Generate a batch of sequences [x1, f(x1), x2, f(x2), ..., xn, f(xn), x_test]
    xs = torch.randint(0, S, (batch_size, 2*n))  # Sample xs and x_test for each sequence in the batch
    # x_test = torch.randint(0, S, (batch_size, 1))  # Randomly sample x_test
    for i in range(batch_size):
        xs[i, 1::2] = f[xs[i, ::2]]  # Apply f to every second element of each sequence (2k-1)
    # prompt = torch.cat([xs, x_test],dim=-1)  # Append x_test at the end of the sequence
    return xs