import numpy as np

def find_stationary_distribution(transition_matrix):
    # 转移矩阵的大小
    n = transition_matrix.shape[0]
    
    # 构建线性方程组
    # A*x = b
    A = np.transpose(transition_matrix) - np.identity(n)
    A[-1, :] = np.ones(n)  # 替换最后一行为和为1的约束
    b = np.zeros(n)
    b[-1] = 1  # 最后一个元素设为1，对应概率和为1的约束
    
    # 解线性方程组找到稳定分布
    stationary_distribution = np.linalg.solve(A, b)
    
    return stationary_distribution

# 示例转移矩阵
transition_matrix = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 1]
])

# 找到稳定分布
stationary_distribution = find_stationary_distribution(transition_matrix)
print("Stationary Distribution:", stationary_distribution)
