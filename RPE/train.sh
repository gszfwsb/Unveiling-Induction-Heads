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



python3 train_independently_model_B_fix_a_W_first.py \
    --n-sample 10000 \
    --vocab-size 3 \
    --seq-length 100 \
    --window-length 5 \
    --n-heads 3 \
    --alpha 2 \
    --a 0.01 \
    --c-alpha 1 \
    --w-plus 100 \
    --w-minus 0.01 \
    --lr1 0.1 \
    --lr2 0.1 \
    --n-epochs 500 \
    --optim adam \
    --device cuda:2 \
    --enable-wandb True