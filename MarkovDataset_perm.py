import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F

def generate_permutation_matrix(S):
    """
    生成一个S×S的置换矩阵。
    """
    # 生成一个随机的排列
    permutation = torch.randperm(S)
    # 创建一个S×S的置换矩阵
    permutation_matrix = torch.zeros((S, S))
    # 根据排列设置置换矩阵的元素
    permutation_matrix[torch.arange(S), permutation] = 1
    return permutation_matrix

def sample_markov_chain_from_permutation(S):
    """
    从置换矩阵集合中采样Markov链π。
    """
    # 生成置换矩阵作为转移矩阵
    transition_matrix = generate_permutation_matrix(S)
    return transition_matrix

class MarkovDataset_perm(torch.utils.data.Dataset):
    def __init__(self, S, T, n_sample):
        self.S = S
        self.T = T
        self.n_sample = n_sample
        self.generate_dist()
        
    def __len__(self):
        return self.n_sample
    
    def generate_dist(self):
        # 从置换矩阵集合中采样Markov链转移矩阵π
        self.pi = sample_markov_chain_from_permutation(self.S)
        # 计算π的平稳分布µπ（对于置换矩阵，每个状态平稳分布相同）
        self.mu_pi = torch.full((self.S,), fill_value=1/self.S)
    
    def __getitem__(self, idx):
        self.generate_dist() # 重新生成pi和mu_pi
        sequence = torch.empty(self.T, dtype=torch.long)
        sequence[0] = torch.multinomial(self.mu_pi, 1).item()
        for i in range(0, self.T-1):
            sequence[i+1] = torch.multinomial(self.pi[sequence[i]], 1).item()
        # 在[S]中均匀抽取s_T，然后根据s_T从π中抽取s_{T+1}
        s_T = torch.randint(0, self.S, (1,)).item()
        sequence[self.T-1] = s_T
        s_T_plus_1 = torch.multinomial(self.pi[s_T], 1).item()
        
        x = F.one_hot(sequence, num_classes=self.S).float()
        y = F.one_hot(torch.tensor(s_T_plus_1), num_classes=self.S)
        return x, y
