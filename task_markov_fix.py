import torch
import torch.distributions as dist
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
    # Initialize the transition matrix
    transition_matrix = torch.stack([dirichlet_distribution.sample() for _ in range(S)])
    # Normalize the rows to sum to 1 (if necessary)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
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
    sequences = torch.ones(size, T, dtype=torch.long) * (-100)
    targets = torch.ones(size, dtype=torch.long)* (-100)
    pis, mu_pis = [],[]
    pbar = tqdm(range(size),ncols=100,mininterval=1)
    pbar.set_description('generating data...')
    dirichlet_distribution = generate_distribution_over_markov_chains(S, alpha)
   
    # Sample a Markov chain transition matrix π from the prior Pπ
    pi = sample_markov_chain_from_distribution(dirichlet_distribution, S)
    # Compute the stationary distribution µπ of π
    mu_pi = stationary_distribution(pi) 
    
    for b in pbar:
        # Sample the first element s1 from the stationary distribution µπ if p(1) is empty
        sequences[b,0] = dist.Categorical(probs=mu_pi).sample()
        # For each position i from 2 to T-1, sample si conditioned on the previous state si-1
        for i in range(0, T-1, 2):
            sequences[b, i] = dist.Categorical(probs=mu_pi).sample()
            sequences[b, i+1] = dist.Categorical(probs=pi[sequences[b, i]]).sample()
        
        # Draw s_T uniformly from [S] and then s_{T+1} from π conditioned on s_T
        s_T = torch.randint(0, S, (1,)).item()
        sequences[b,T-1] = s_T
        targets[b] = dist.Categorical(probs=pi[s_T]).sample().item()

        # Return the input sequence x = s_{1:T} and the target y = s_{T+1}
        pis.append(pi)
        mu_pis.append(mu_pi)
    assert (sequences>=0).all(), 'negative values!'
    assert (sequences<S).all(), 'large values!'
    return sequences, targets, pis, mu_pis
