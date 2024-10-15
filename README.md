# Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers
This repository includes codes for paper [Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers](https://arxiv.org/abs/2409.10559), NeurIPS 2024

## Run the simulation
```bash
cd ./RPE
# To implement the results, try
python3 train_independently_model_B_fix_a_first.py --w-plus 1 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 
python3 train_independently_model_B_fix_a_first.py --w-plus 0.05 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 
python3 train.py --w-plus 1 --device cuda:2 --n-epochs 5000 --lr 1e2 
