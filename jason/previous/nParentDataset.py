import torch
import torch.distributions as dist

import torch
import torch.nn.functional as F
import torch.distributions as dist

# Function to generate Pπ over irreducible and aperiodic Markov chains π on [S]
def generate_distribution_over_markov_chains(S, alpha):
    # Ensure alpha is greater than zero to avoid zero probabilities
    assert alpha > 0, "Alpha must be greater than 0 to ensure irreducibility and aperiodicity"
    
    # Define the Dirichlet distribution for a single row with concentration parameter alpha
    # Since the Dirichlet distribution with alpha > 0 will produce only positive probabilities,
    # the generated transition matrix will be irreducible and aperiodic
    alpha_tensor = torch.full((S,), alpha)
    dirichlet_distribution = dist.Dirichlet(alpha_tensor)

    return dirichlet_distribution

# Extend the original MarkovDataset class
class ParentDataset(torch.utils.data.Dataset):
    def __init__(self, S, T, alpha, n_sample, k):
        super().__init__()
        self.S = S
        self.T = T
        self.n_sample = n_sample
        self.k = k
        self.dirichlet_distribution = generate_distribution_over_markov_chains(S, alpha)
        self.transition_tensors = self.sample_transition_tensors(k, S)
        
    def sample_transition_tensors(self, k, S):
        # Generate k-parent transition tensors
        tensors = []
        for i in range(k):
            tensor = torch.stack([self.dirichlet_distribution.sample() for _ in range(S)])
            tensors.append(tensor)
        return tensors

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        # Generate a sequence based on k-parent tensors
        sequence = torch.empty(self.T + 1, dtype=torch.long)
        for i in range(self.T + 1):
            if i < self.k:  # If there are not enough parents, sample uniformly
                sequence[i] = torch.randint(0, self.S, (1,)).item()
            else:
                # Otherwise, sample based on the k-parent tensor
                context = sequence[i-self.k:i]
                probs = self.transition_tensors[self.k-1][context[-1]]
                for j in range(self.k-2, -1, -1):
                    # print(probs.shape, self.transition_tensors[j][context[j]].shape)
                    probs = probs * self.transition_tensors[j][context[j]]
                sequence[i] = dist.Categorical(probs=probs).sample()

        # Input x is the sequence excluding the last element
        # Target y is the last element of the sequence
        x = F.one_hot(sequence[:-1], num_classes=self.S).float()
        y = F.one_hot(sequence[-1], num_classes=self.S)
        return x, y

