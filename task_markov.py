import torch
import torch.distributions as dist
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to sample a Markov chain transition matrix π where each row is sampled i.i.d from Dir(α · 1_S)
def sample_markov_chain(S, alpha):
    # Sample from the Dirichlet distribution for each row
    alpha_tensor = torch.full((S,), alpha)
    transition_matrix = dist.Dirichlet(alpha_tensor).sample((S,))
    return transition_matrix

# Function to find the stationary distribution µπ of π
def stationary_distribution(A):
    # Solve (πP = π) for π with a uniform initial distribution
    n = A.shape[0]
    b = torch.zeros(n)
    x = torch.linalg.solve(A-torch.eye(n), b)
    assert torch.allclose(A @ x, x)
    print(x)
    return x

# Function to generate a random sequence with causal structure
def generate_sequence_with_causal_structure(S, T, alpha, size=1):  
    # Initialize the sequence
    sequences = torch.empty(size, T, dtype=torch.long)
    s_T_plus_1 = torch.empty(size, dtype=torch.long)
    pis, mu_pis = [],[]
    pbar = tqdm(range(size),ncols=100,mininterval=1)
    pbar.set_description('generating data...')
    for b in pbar:
        # Sample a Markov chain transition matrix π from the prior Pπ
        pi = sample_markov_chain(S, alpha)
        # Compute the stationary distribution µπ of π
        mu_pi = stationary_distribution(pi)
        # Sample the first element s1 from the stationary distribution µπ if p(1) is empty
        sequences[b,0] = dist.Categorical(probs=mu_pi).sample()
        
        # For each position i from 2 to T-1, sample si conditioned on the previous state si-1
        for i in range(1, T-1):
            sequences[b,i] = dist.Categorical(probs=pi[sequences[b,i-1]]).sample()
        
        # Draw s_T uniformly from [S] and then s_{T+1} from π conditioned on s_T
        s_T = torch.randint(0, S, (1,)).item()
        s_T_plus_1[b] = dist.Categorical(probs=pi[s_T]).sample().item()
        sequences[b,T-1] = s_T
        # Return the input sequence x = s_{1:T} and the target y = s_{T+1}
        pis.append(pi)
        mu_pis.append(mu_pi)
    return sequences, s_T_plus_1, pis, mu_pis
