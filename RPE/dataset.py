import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.utils.data import Dataset



class MarkovDataset(Dataset):
    def __init__(self, S, L, alpha, n_sample):
        self.S = S
        self.L = L
        self.n_sample = n_sample
        self.dirichlet_distribution = self.generate_distribution(S, alpha)
        self.mu_pi = torch.ones(self.S) / self.S
        self.samples = []
        print('generating datasets...')
        for _ in range(n_sample):
            self.sample() # regenerate pi
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

    def generate_distribution(self, S, alpha):
        # Assuming the distribution is a Dirichlet distribution for each state
        return dist.Dirichlet(torch.full((S,), alpha))

    def sample_transition(self, distribution, S):
        # Sample a stochastic matrix for Markov chain transitions
        return distribution.sample((S,))

    def __len__(self):
        return self.n_sample
    
    def sample(self):
        # Sample a Markov chain transition matrix π from the prior Pπ
        self.pi = self.sample_transition(self.dirichlet_distribution, self.S)
        # Compute the stationary distribution µπ of π
    
    def __getitem__(self, idx):
        return self.samples[idx]



class NGramDataset(Dataset):
    def __init__(self, S, L, n, alpha, n_sample, output=False):
        self.S = S  # Number of states
        self.L = L  # Length of sequence
        self.n = n  # The n in n-gram
        self.alpha = alpha
        self.n_sample = n_sample
        self.dirichlet_distribution = dist.Dirichlet(torch.full((S,), alpha))
        self.mu_pi = torch.ones(self.S) / self.S # uniformly
        self.samples = [] 
        if output:
            print('Generating datasets...')
        for _ in range(n_sample):
            self.sample()  # regenerate pi
            sequence = torch.empty(self.L, dtype=torch.long)
            sequence[:self.n-1] = dist.Categorical(probs=self.mu_pi).sample((self.n-1,)) # uniformly sample
            for i in range(self.n-1, self.L):
                # Condition on the previous n-1 states
                context = sequence[i-self.n+1:i]
                context_idx = self.context_to_index(context)
                sequence[i] = dist.Categorical(probs=self.pi[context_idx]).sample()
            # The input sequence x is the sequence up to L, and the target y is the next state
            x = F.one_hot(sequence, num_classes=self.S).float() # length L
            y_context = dist.Categorical(probs=self.mu_pi).sample((self.n-1,)) # uniformly sample
            y_context_idx = self.context_to_index(y_context)
            y = dist.Categorical(probs=self.pi[y_context_idx]).sample()
            self.samples.append((x, y))

    def context_to_index(self, context):
        # Convert a sequence context to an index for accessing the transition matrix
        # This assumes that context is a tensor of indices
        index = 0
        for i, state in enumerate(context.flip(0)):
            index += state * (self.S ** i)
        return index

    def sample_transition(self):
        # Sample a stochastic matrix for n-gram transitions
        # Each row in pi corresponds to a unique context of n-1 states
        pi = torch.zeros((self.S ** (self.n-1), self.S))
        for i in range(self.S ** (self.n-1)):
            pi[i] = self.dirichlet_distribution.sample()
        return pi

    def __len__(self):
        return self.n_sample

    def sample(self):
        # Sample a Markov chain transition matrix π from the prior Pπ
        # for the n-gram model
        self.pi = self.sample_transition()

    def __getitem__(self, idx):
        return self.samples[idx]