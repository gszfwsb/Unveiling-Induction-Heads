# python3 train.py --batch-size 1024 --n-epoch 100 --device cuda:0
# python3 train.py --batch-size 512 --n-epoch 100 --device cuda:0
# python3 train.py --batch-size 256 --n-epoch 1000 --device cuda:0
# python3 train.py --batch-size 128 --n-epoch 1000 --device cuda:0
# python3 train.py --batch-size 64 --n-epoch 100 --device cuda:0



# python3 train_once.py --batch-size 256 --lr 1 --n-epoch 1000 --device cuda:0 # more than 2**17 steps, bs no bigger than 762



# python3 train_once.py --batch-size 256 --lr 1 --n-epoch 1000 --device cuda:0 --enable-wandb True # more than 2**17 steps, bs no bigger than 762

# python3 train_once.py --batch-size 10000 --lr 1 --n-epoch 1000 --device cuda:2


# # true settings
# python3 train_once.py --batch-size 1024 --lr 1 --n-epoch 1024 --n-sample 100 --device cuda:0 --alpha 0.1 #--enable-wandb True


# python3 train.py --lr 0.01 --device cuda:0 --enable-wandb True --data-type 'Two grams' --optim adam --init realistic --n-epochs 1000  --n-heads 2 1
# python3 train.py --lr 0.01 --device cuda:0 --enable-wandb True --data-type 'Two grams' --optim adam --init paper --n-epochs 1000  --n-heads 2 1
# python3 train.py --lr 0.3 --device cuda:0 --enable-wandb True --data-type 'Two grams' --optim sgd --init realistic --n-epochs 1000  --n-heads 2 1
# python3 train.py --lr 0.3 --device cuda:0 --enable-wandb True --data-type 'Two grams' --optim sgd --init paper --n-epochs 1000  --n-heads 2 1

# python3 train.py --lr 0.01 --device cuda:0 --enable-wandb True --data-type 'Markov chain' --optim adam --init realistic --n-epochs 1000 --n-heads 1 1
# python3 train.py --lr 0.01 --device cuda:0 --enable-wandb True --data-type 'Markov chain' --optim adam --init paper --n-epochs 1000  --n-heads 1 1
# python3 train.py --lr 0.3 --device cuda:0 --enable-wandb True --data-type 'Markov chain' --optim sgd --init realistic --n-epochs 1000  --n-heads 1 1
# python3 train.py --lr 0.3 --device cuda:0 --enable-wandb True --data-type 'Markov chain' --optim sgd --init paper --n-epochs 1000  --n-heads 1 1

# python3 train.py --lr 0.01 --device cuda:0 --enable-wandb True --data-type 'Markov chain' --optim adam --init random --n-epochs 1000  --n-heads 1 1
# python3 train.py --lr 0.3 --device cuda:0 --enable-wandb True --data-type 'Markov chain' --optim sgd --init random --n-epochs 1000  --n-heads 1 1


python3 train.py --lr 0.01 --device cuda:0 --enable-wandb True --data-type 'Two grams' --optim adam --init random --n-epochs 1000  --n-heads 2 1
python3 train.py --lr 0.3 --device cuda:0 --enable-wandb True --data-type 'Two grams' --optim sgd --init random --n-epochs 1000  --n-heads 2 1
