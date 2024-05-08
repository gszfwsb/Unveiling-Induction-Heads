from tqdm import tqdm, trange
from dataset import NGramDataset
from utils import *
from tools import *
# Initialize the dataset
S, L, n_language, alpha, n_sample = 3, 100, 4, 0.3, 1
n = n_language - 1
num_test = 1000
power = 2

pbar = tqdm(range(num_test),ncols=100,mininterval=1)
mu = torch.zeros(num_test, S**n)
MI = torch.zeros(num_test, 2**n)
for j in pbar:
    dataset = NGramDataset(S, L, n_language, alpha, n_sample)
    pi = dataset.pi
    if j % 20 == 0:
        pbar.set_description(f'Iteration {j+1}/{num_test}')
        x, _ = get_stationary(pi, S, n, output=False)
    else: 
        x, _ = get_stationary(pi, S, n)

    mu[j] = x.reshape(-1)

    # test get_stationary_multi_parent
    mu_prod_pi = (x.reshape(-1, 1) * pi).transpose(0, 1).view(
        tuple([S for _ in range(n+1)])
        )
    for i, support in enumerate(range(0, 2**n)):
        MI[j, i] = chi_square_mutual_info_support(support, mu_prod_pi, S, n, power)
        # print(f'Power {power} chi-square MI for support {ind2code(support, 2, n)}: {chi}')

    # compute the mutual information
    for ind in range(2**n):
        support = ind2code(ind, 2, n)
        

mean = mu.mean(axis=0)
std = mu.std(axis=0)

hist_path = f'Dataset_Info/S={S}_n={n}_L={L}_alpha={alpha}_num_test={num_test}'
makedirs(hist_path)
# Let's plot the mean and standard deviation of the stationary distribution
plot_hist(mean.numpy(), S, n, f'{hist_path}/stationary_mean.png', title=f'Mean of the stationary distribution: S={S},n={n},L={L},alpha={alpha},num_test={num_test}')
plot_hist(std.numpy(), S, n, f'{hist_path}/stationary_std.png', title=f'Standard deviation of the stationary distribution: S={S},n={n},L={L},alpha={alpha},num_test={num_test}')

# Let's plot the mutual information
mean_MI = MI.mean(axis=0)
std_MI = MI.std(axis=0)

plot_hist(mean_MI.numpy(), 2, n, f'{hist_path}/MI_mean.png', title=f'Mean of the mutual information: S={S},n={n},L={L},alpha={alpha},num_test={num_test}')
plot_hist(std_MI.numpy(), 2, n, f'{hist_path}/MI_std.png', title=f'Standard deviation of the mutual information: S={S},n={n},L={L},alpha={alpha},num_test={num_test}')

