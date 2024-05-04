# python3 train_independently_model_B_fix_a_W_first.py \
#     --n-sample 10000 \
#     --vocab-size 3 \
#     --seq-length 100 \
#     --window-length 5 \
#     --n-heads 3 \
#     --alpha 0.1 \
#     --n-epochs 2000 \
#     --optim adam \
#     --device cuda:0 \
#     --enable-wandb True



# python3 train_independently_model_B_fix_a_W_first.py \
#     --n-sample 100 \
#     --vocab-size 3 \
#     --seq-length 100 \
#     --window-length 5 \
#     --n-heads 3 \
#     --alpha 0.1 \
#     --a 0.01 \
#     --c-alpha 1 \
#     --w-plus 100 \
#     --w-minus 0.01 \
#     --lr1 0.1 \
#     --lr2 0.1 \
#     --n-epochs 500 \
#     --optim adam \
#     --device cuda:2 \
#     --enable-wandb True


python3 train.py --w-plus 100 --device cuda:5 --n-epochs 10000 
python3 train.py --w-plus 50 --device cuda:5 --n-epochs 10000 
python3 train.py --w-plus 50 --device cuda:4 --n-epochs 10000 --enable-wandb True
python3 train.py --w-plus 40 --device cuda:3 --n-epochs 10000 --enable-wandb True
python3 train.py --w-plus 30 --device cuda:2 --n-epochs 10000 --enable-wandb True
python3 train.py --w-plus 20 --device cuda:1 --n-epochs 10000 --enable-wandb True
python3 train.py --w-plus 1 --device cuda:1 --n-epochs 10000 --lr 1e1
python3 train.py --w-plus 1 --device cuda:2 --n-epochs 10000 --lr 1e2
python3 train_independently_model_B_fix_W_first.py --w-plus 1 --device cuda:4 --n-epochs 10000 --lr1 1e5 --lr2 1e6 # works!!!
python3 train_independently_model_B_fix_a_first.py --w-plus 100 --device cuda:5 --n-epochs 5000 --lr1 1e1 --lr2 1e2


