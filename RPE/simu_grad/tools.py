import torch
import numpy as np

def ind2code(ind, S, n):
    """Decode the index into a base-S list of length n"""
    assert ind < S**n
    code = [0 for _ in range(n)]
    for j in range(n):
        code[n - j - 1] = ind % S
        ind //= S
    return code
def code2ind(code, S):
    """Encode the base-S list into an index"""
    n = len(code)
    return sum([code[j] * S**(n-j-1) for j in range(n)])
def next_state(parent, son, S, n):
    """Return the next state of the Markov chain given the current state i and the next symbol j
    
    Args:
    parent: Union[int, list], the parent context
    son: int, the next symbol
    S: int, the number of symbols in the alphabet
    n: int, the length of parent context

    Returns:
    state: Union[int, list], the next state of the Markov chain, the form is determined by the input parent
    """
    if isinstance(parent, int):
        return parent // S + son * S**(n-1)
    else:
        return [son] + parent[0:-1]
    

def get_stationary(pi, S, n, max_iter=100, seed_index=0, output=False):
    """Get the stationary distribution of the Markov chain
    
    Args:
    pi: torch.tensor, the transition matrix
    S: int, the number of symbols in the alphabet
    n: int, the length of the context
    max_iter: int, the maximum number of iterations
    seed_index: int, the index of the seed distribution

    Returns:
    x: torch.tensor, the stationary distribution
    """
    # initialize x
    x = torch.rand(S**n, 1)
    x /= x.sum()

    # get the joint distribution of the parent and the son
    for i in range(max_iter):
        y = x * pi
        # transpose and reshape
        y = y.transpose(0, 1).reshape(-1).reshape(-1, S).sum(axis=-1, keepdims=True)

        # take the TV distance
        d = torch.abs(x - y).sum()
        # print(f'Iteration {i+1}, TV distance: {d}')

        # update x
        x = y
    if output:
        print(f'Final TV distance after {i+1} iterations: {d}')
    return x, d

# Let's histogram the stationary distribution
import matplotlib.pyplot as plt

def plot_hist(y, S, n, title='Stationary distribution'):
    """Plot the stationary distribution"""
    plt.bar(range(S**n), y.reshape(-1))
    plt.xlabel('State')
    plt.title(title)
    # add the state's code to the x-axis
    plt.xticks(range(S**n), [f'{i}:{ind2code(i, S, n)}' for i in range(S**n)], rotation=-90)
    plt.show()

# get the stationary distribution for window size 1
def get_stationary_single_symbol(mu, n):
    S = int(np.exp(np.log(len(mu.squeeze())) / n))
    return mu.reshape(S, -1).sum(axis=-1, keepdims=False)
    
def get_stationary_multi_support(mu_prod_pi, support, S, n):
    """Get the stationary distribution for multiple parents
    
    Args:
    mu: torch.tensor, the stationary distribution
    pi: torch.tensor, the transition matrix
    support: Union[list, int], the list of support of the parents or the binary representation of the support
    
    Returns:
    torch.tensor, the stationary distribution"""

    if tuple(mu_prod_pi.shape) == (S**n, S):
        mu_prod_pi = mu_prod_pi.transpose(0, 1).reshape(-1, S)
    mu_extended = mu_prod_pi.view(tuple(
            [S for _ in range(n+1)]
        ))
    
    if isinstance(support, int):
        support = ind2code(support, 2, n)
    assert isinstance(support, list)
    # include the current state
    support_extended = torch.tensor([1] + support, dtype=torch.bool)
    
    # marginalize out the unsupported parents
    # check if all the parents are supported
    if all(support_extended):
        return mu_extended
    else:
        return torch.sum(mu_extended, dim=tuple([i for i in range(n+1) if not support_extended[i]]), keepdim=True)
        
# # calculate the chi-square mutual information
# def chi_square_mutual_info(parent, pi, mu):
#     S = pi.shape[1]
#     # n = log(S, pi.shape[0])
#     n = int(np.log(pi.shape[0]) / np.log(S))

#     # get the stationary distribution for one symbol
#     mu_single = get_stationary_single_symbol(mu, n)
    
#     if parent == -1:
#         # the average chi-square mutual information
#         return mu.reshape(1, -1) @ ((pi ** 2 / mu_single.reshape(1, -1)).sum(axis=-1, keepdims=True) - 1)
#     elif parent == -2:
#         # the inner product of the squared stationary distribution and the mutual information
#         return mu.reshape(1, -1)**2 @ ((pi ** 2 / mu_single.reshape(1, -1)).sum(axis=-1, keepdims=True) - 1)
#     else: 
#         if isinstance(parent, list):
#             parent = code2ind(parent, S)
#         p = pi[parent]
#         # the chi-square mutual information between mu_single and p
#         return (p ** 2 / mu_single).sum() - 1
#     # take the average over all the parents
    
def chi_square_mutual_info(joint_dist, power=1):
    """Calculate the chi-square mutual information
    
    Args:
    joint_dist: torch.tensor, the joint distribution of the parent and the son
    
    Returns:
    torch.tensor, the chi-square mutual information"""
    marginal_dist = joint_dist.sum(dim=tuple(range(1, joint_dist.ndim)), keepdim=True)
    joint_dist_parent = joint_dist.sum(dim=0, keepdim=True)
    return ((((joint_dist / joint_dist_parent) ** 2 / marginal_dist).sum(dim=0, keepdim=True) - 1) * joint_dist_parent**power).sum()

def chi_square_mutual_info_support(support, mu_prod_pi, S, n, power=1):
    mu_extended = get_stationary_multi_support(mu_prod_pi, support, S, n)
    return chi_square_mutual_info(mu_extended, power)