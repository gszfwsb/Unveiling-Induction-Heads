import torch
import torch.distributions as dist
import networkx as nx
import matplotlib.pyplot as plt


# Function to sample a Markov chain transition matrix π where each row is sampled i.i.d from Dir(α · 1_S)
def sample_markov_chain(S, alpha):
    # Sample from the Dirichlet distribution for each row
    alpha_tensor = torch.full((S,), alpha)
    transition_matrix = dist.Dirichlet(alpha_tensor).sample((S,))
    return transition_matrix

# Function to find the stationary distribution µπ of π
def stationary_distribution(transition_matrix):
    # Solve (πP = π) for π with a uniform initial distribution
    stationary_dist = torch.full((transition_matrix.size(0),), fill_value=1/transition_matrix.size(0))
    previous_dist = torch.zeros_like(stationary_dist)
    while not torch.allclose(stationary_dist, previous_dist):
        previous_dist = stationary_dist.clone()
        stationary_dist = torch.mv(transition_matrix, stationary_dist)
    return stationary_dist

# Function to generate a random sequence with causal structure
def generate_sequence_with_causal_structure(S, T, alpha, batch_size=1):
    # Sample a Markov chain transition matrix π from the prior Pπ
    pi = sample_markov_chain(S, alpha)
    # Compute the stationary distribution µπ of π
    mu_pi = stationary_distribution(pi)
    
    # Initialize the sequence
    sequences = torch.empty(batch_size, T, dtype=torch.long)
    s_T_plus_1 = torch.empty(batch_size, dtype=torch.long)
    for b in range(batch_size):
        # Sample the first element s1 from the stationary distribution µπ if p(1) is empty
        sequences[b,0] = dist.Categorical(probs=mu_pi).sample()
        
        # For each position i from 2 to T-1, sample si conditioned on the previous state si-1
        for i in range(1, T-1):
            sequences[b,i] = dist.Categorical(probs=pi[sequences[b,i-1]]).sample()
        
        # Draw s_T uniformly from [S] and then s_{T+1} from π conditioned on s_T
        s_T = dist.Uniform(0, S).sample().long().item()
        s_T_plus_1[b] = dist.Categorical(probs=pi[s_T]).sample().item()
        sequences[b,T-1] = s_T
        # Return the input sequence x = s_{1:T} and the target y = s_{T+1}
    return sequences, s_T_plus_1, pi, mu_pi

# Set parameters for the sequence generation
# S = 3  # Size of the state space
# T = 6  # Length of the sequence
# alpha = 0.1  # Dirichlet parameter

# Generate the sequence with causal structure
# x, y = generate_sequence_with_causal_structure(S, T, alpha)
# print(f"Input sequence x: {x}")
# print(f"Target y: {y}")
# print('----')
