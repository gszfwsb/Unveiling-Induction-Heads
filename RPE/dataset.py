import torch
import torch.distributions as dist
import torch.nn.functional as F

def generate_distribution_over_markov_chains(S, alpha):
    # Assuming the distribution is a Dirichlet distribution for each state
    return dist.Dirichlet(torch.full((S,), alpha))

def sample_markov_chain_from_distribution(distribution, S):
    # Sample a stochastic matrix for Markov chain transitions
    return distribution.sample((S,))


class MarkovDataset(torch.utils.data.Dataset):
    def __init__(self, S, L, alpha, n_sample):
        self.S = S
        self.L = L
        self.n_sample = n_sample
        self.dirichlet_distribution = generate_distribution_over_markov_chains(S, alpha)
        self.mu_pi = torch.ones(self.S) / self.S
        self.samples = []
        print('generating datasets...')
        for _ in range(n_sample):
            self.generate_dist() # regenerate pi
            sequence = torch.empty(self.L, dtype=torch.long)
            sequence[0] = dist.Categorical(probs=self.mu_pi).sample()
            for i in range(0, self.L-2):
                sequence[i+1] = dist.Categorical(probs=self.pi[sequence[i]]).sample()
            # Draw s_L uniformly from [S] and then s_{L+1} from π conditioned on s_L
            s_L = torch.randint(0, self.S, (1,)).item()
            sequence[self.L-1] = s_L
            y = dist.Categorical(probs=self.pi[s_L]).sample()
            x = F.one_hot(sequence, num_classes=self.S).float()
            # y = F.one_hot(y, num_classes=self.S)
            self.samples.append((x,y))

    def __len__(self):
        return self.n_sample
    
    def generate_dist(self):
        # Sample a Markov chain transition matrix π from the prior Pπ
        self.pi = sample_markov_chain_from_distribution(self.dirichlet_distribution, self.S)
        # Compute the stationary distribution µπ of π
    
    def __getitem__(self, idx):
        return self.samples[idx]