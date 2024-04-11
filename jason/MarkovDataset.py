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

# Function to sample a Markov chain π from the distribution Pπ
def sample_markov_chain_from_distribution(dirichlet_distribution, S):
    # Sample each row of the transition matrix from the Dirichlet distribution
    transition_matrix = torch.stack([dirichlet_distribution.sample() for _ in range(S)])
    return transition_matrix


# Function to find the stationary distribution µπ of π
def stationary_distribution(transition_matrix):
    stationary_dist = torch.full((transition_matrix.size(0),), fill_value=1/transition_matrix.size(0))
    previous_dist = torch.zeros_like(stationary_dist)
    while not torch.allclose(stationary_dist, previous_dist):
        previous_dist = stationary_dist.clone()
        stationary_dist = torch.mv(transition_matrix, stationary_dist)
    return stationary_dist

class MarkovDataset(torch.utils.data.Dataset):
    def __init__(self, S, T, alpha, n_sample):
        self.S = S
        self.T = T
        self.n_sample = n_sample
        self.dirichlet_distribution = generate_distribution_over_markov_chains(S, alpha)
        self.generate_dist()
    def __len__(self):
        return self.n_sample
    
    def generate_dist(self):
        # Sample a Markov chain transition matrix π from the prior Pπ
        self.pi = sample_markov_chain_from_distribution(self.dirichlet_distribution, self.S)
        # Compute the stationary distribution µπ of π
        self.mu_pi = stationary_distribution(self.pi)
    
    def __getitem__(self, idx):
        self.generate_dist() # regenerate pi and mu_pi
        sequence = torch.empty(self.T, dtype=torch.long)
        sequence[0] = dist.Categorical(probs=self.mu_pi).sample()
        for i in range(0, self.T-1):
            sequence[i+1] = dist.Categorical(probs=self.pi[sequence[i]]).sample()
        # Draw s_T uniformly from [S] and then s_{T+1} from π conditioned on s_T
        s_T = torch.randint(0, self.S, (1,)).item()
        sequence[self.T-1] = s_T
        s_T_plus_1 = dist.Categorical(probs=self.pi[s_T]).sample()
        
        x = F.one_hot(sequence, num_classes=self.S).float()
        y = F.one_hot(s_T_plus_1, num_classes=self.S)
        # print(x.shape, y.shape)
        return x, y