cd ../


python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 10 --c-alpha 0.01 --device cuda:1 --n-epochs 5000 --lr 1 --lr 1 --lr 1 --train-cmd C --train-cmd W --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 10 --c-alpha 0.01 --device cuda:1 --n-epochs 5000 --lr 1 --lr 1 --train-cmd CW --train-cmd a
python3 train_simplified.py --vocab-size 3 --seq-length 100 --n-heads 5 --n-gram 4 --w-plus 10 --c-alpha 0.01 --device cuda:1 --n-epochs 5000 --lr 1  --train-cmd CWa