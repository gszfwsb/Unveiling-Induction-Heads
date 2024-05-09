from itertools import combinations


train_cmd = ['CW','a']



n_params = len(train_cmd)
for i in range(1, n_params + 1):
    for subset in combinations(train_cmd, i):
        train_C = 'C' in subset
        train_W = 'W' in subset
        train_a = 'a' in subset
        print(train_C, train_W, train_a)