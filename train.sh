python3 train.py --batch-size 1024 --n-epoch 100 --device cuda:0
python3 train.py --batch-size 512 --n-epoch 100 --device cuda:0
python3 train.py --batch-size 256 --n-epoch 1000 --device cuda:0
python3 train.py --batch-size 128 --n-epoch 1000 --device cuda:0
python3 train.py --batch-size 64 --n-epoch 100 --device cuda:0



python3 train_once.py --batch-size 256 --lr=1 --n-epoch 1000 --device cuda:0 --enable-wandb True # more than 2**17 steps, bs no bigger than 762
