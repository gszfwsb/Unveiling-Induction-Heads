cd ../

python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:2 --n-epochs 5000 --lr 1 --lr 1 --lr 1 --train-cmd C --train-cmd W --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:2 --n-epochs 5000 --lr 1 --lr 1 --train-cmd CW --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 2 --c-alpha 0.01 --device cuda:2 --n-epochs 5000 --lr 1  --train-cmd CWa




python3 train_simplified.py --vocab-size 10 --n-sample 100 --seq-length 20 --n-heads 2 --n-gram 3 --w-plus 2 --c-alpha 0.01 --device cuda:2 --n-epochs 2000 --lr 1 --lr 1 --lr 1 --train-cmd C --train-cmd W --train-cmd a
