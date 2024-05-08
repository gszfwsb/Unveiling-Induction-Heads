python3 train_independently_model_B_fix_a_first.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 2 --c-alpha 0.01 --device cuda:2 --n-epochs 2000 --lr1 1 --lr2 1  # works!!!
python3 train_independently_model_B_fix_a_first.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 1 --c-alpha 0.01 --device cuda:1 --n-epochs 2000 --lr1 1 --lr2 1  # works!!!


#### low degree
python3 train_independently_model_B_fix_a_first.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:0 --n-epochs 2000 --lr1 1 --lr2 1  # works!!!
python3 train_independently_model_B_fix_a_first.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:1 --n-epochs 2000 --lr1 1 --lr2 1  --low-degree 3 # works!!!
python3 train_independently_model_B_fix_a_first.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:2 --n-epochs 2000 --lr1 1 --lr2 1  --low-degree 2 # works!!!
