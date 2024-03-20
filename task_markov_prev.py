import torch
import torch.distributions as dist
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
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

# Function to generate a random sequence with causal structure
def generate_sequence_with_causal_structure(S, T, alpha, size=1):  
    # Initialize the sequence
    sequences = torch.empty(size, T, dtype=torch.long)
    s_T_plus_1 = torch.empty(size, dtype=torch.long)
    pis, mu_pis = [],[]
    pbar = tqdm(range(size),ncols=100,mininterval=1)
    pbar.set_description('generating data...')
    dirichlet_distribution = generate_distribution_over_markov_chains(S, alpha)

    for b in pbar:
        # Sample a Markov chain transition matrix π from the prior Pπ
        pi = sample_markov_chain_from_distribution(dirichlet_distribution, S)
        # Compute the stationary distribution µπ of π
        mu_pi = stationary_distribution(pi)
        # Sample the first element s1 from the stationary distribution µπ if p(1) is empty
        sequences[b,0] = dist.Categorical(probs=mu_pi).sample()
        
        # For each position i from 2 to T-1, sample si conditioned on the previous state si-1
        for i in range(1, T-1):
            if i % 2 == 1:  # For odd-indexed nodes, sample from mu_pi
                sequences[b, i] = dist.Categorical(probs=mu_pi).sample()
            else:  # For even-indexed nodes, sample conditioned on the previous state
                prev_state = sequences[b, i-1]
                sequences[b, i] = dist.Categorical(probs=pi[prev_state]).sample()
      
        
        # Draw s_T uniformly from [S] and then s_{T+1} from π conditioned on s_T
        s_T = torch.randint(0, S, (1,)).item()
        sequences[b,T-1] = s_T
        s_T_plus_1[b] = dist.Categorical(probs=pi[s_T]).sample().item()
        # Return the input sequence x = s_{1:T} and the target y = s_{T+1}
        pis.append(pi)
        mu_pis.append(mu_pi)
    return sequences, s_T_plus_1, pis, mu_pis
