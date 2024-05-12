python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:0 --n-epochs 5000 --lr 1 --lr 1 --lr 1 --train-cmd C --train-cmd W --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:0 --n-epochs 5000 --lr 1 --lr 1 --train-cmd CW --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:0 --n-epochs 5000 --lr 1  --train-cmd CWa


python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 10 --c-alpha 0.01 --device cuda:0 --n-epochs 5000 --lr 1 --lr 1 --lr 1 --train-cmd C --train-cmd W --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 10 --c-alpha 0.01 --device cuda:0 --n-epochs 5000 --lr 1 --lr 1 --train-cmd CW --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 10 --c-alpha 0.01 --device cuda:0 --n-epochs 5000 --lr 1  --train-cmd CWa


python3 train.py --vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 100 --device cuda:0 --n-epochs 2000 --lr 1 --n-sample 10000
