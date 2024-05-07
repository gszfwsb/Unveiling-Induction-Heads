# train with 2 time-scale
python3 train_independently_model_B_fix_a_first.py --w-plus 1 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 --enable-wandb True # works!!!
python3 train_independently_model_B_fix_a_first.py --w-plus 0.05 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 --enable-wandb True # works!!!
python3 train_independently_model_B_fix_a_first.py --w-plus 1 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 --enable-wandb True --low-degree 3 # works!!!
python3 train_independently_model_B_fix_a_first.py --w-plus 0.05 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 --enable-wandb True --low-degree 3 # works!!!
python3 train_independently_model_B_fix_a_first.py --w-plus 1 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 --enable-wandb True --low-degree 2 # works!!!
python3 train_independently_model_B_fix_a_first.py --w-plus 0.05 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 --enable-wandb True --low-degree 2 # works!!!



# train with 1 time-scale
python3 train.py --w-plus 1 --device cuda:2 --n-epochs 5000 --lr 1e2 --enable-wandb True # works!!!
python3 train.py --w-plus 1 --device cuda:2 --n-epochs 5000 --lr 1e2 --enable-wandb True --low-degree 3 # works!!!
python3 train.py --w-plus 1 --device cuda:2 --n-epochs 5000 --lr 1e2 --enable-wandb True --low-degree 2 # works!!!



