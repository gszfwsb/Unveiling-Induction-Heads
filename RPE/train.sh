python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda:0 \
--n-epochs 2000 --n-epochs 5000 --n-epochs 5000 --lr 1 --lr 1e5 --lr 1 --train-cmd C --train-cmd W --train-cmd a --low-degree 3


python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda:1 \
--n-epochs 2000 --n-epochs 5000 --n-epochs 5000 --lr 1 --lr 1e5 --lr 1 --train-cmd C --train-cmd W --train-cmd a --low-degree 2

python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda:0 \
--n-epochs 2000 --n-epochs 50000 --n-epochs 5000 --lr 1 --lr 1 --lr 1 --train-cmd C --train-cmd W --train-cmd a --low-degree 2


python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda:1 \
--n-epochs 500000 --n-epochs 5000 --lr 1  --lr 1 --train-cmd CW --train-cmd a --low-degree 2


python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda:4 \
--n-epochs 50000   --lr 1 --train-cmd CWa --low-degree 2




python3 train.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 10 --c-alpha 0.01 --device cuda:1 --n-epochs 50000  --lr 1 --low-degree 2 

python3 train.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda:2 --n-epochs 50000  --lr 1 --low-degree 2
