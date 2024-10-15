# Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers
This repository includes codes for paper [Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers](https://arxiv.org/abs/2409.10559), NeurIPS 2024

## Run the simulation
The following parameters can be configured to customize the training of the 2-layer disentangled Transformer:
- vocab-size: Integer, default=3. the size of the vocabulary.
- seq-length: Integer, default=20. the length of the input sequences.
- window-length: Integer, default=5. the window length for relative positional embeddings.
- n-heads: Integer, default=3. the number of attention heads.
- lr1: Float, default=0.8. the learning rate for the first layer.
- lr2: Float, default=0.8. the learning rate for the second layer.
- batch-size: Integer, default=100000. the batch size for training.
- seed: Integer, default=2024. the seed for random number generation.
- n-sample: Integer, default=10000. the number of samples.
- device: String, default='cuda:0'. the device for computation (e.g., 'cuda:0', 'cpu').
- dataset: String, default='NGram'. the dataset to use for training.
- optim: String, default='adam'. the optimizer to use (e.g., 'adam', 'sgd').
- w-plus: Float, default=1. the weight for positive samples.
- w-minus: Float, default=0.01. the weight for negative samples.
- a: Float, default=0.01. the value for parameter 'a'.
- c-alpha: Float, default=1. the value for parameter 'c_alpha'.
- alpha: Float, default=0.3. the value for parameter 'alpha'.
- n-epochs: Integer, default=10000. the number of epochs for training.
- n-gram: Integer, default=3. the value for parameter 'n_gram'.

```bash
cd ./RPE
# To implement the results, try
python3 train_independently_model_B_fix_a_first.py --w-plus 1 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 
python3 train_independently_model_B_fix_a_first.py --w-plus 0.05 --device cuda:1 --n-epochs 5000 --lr1 1e2 --lr2 1e3 
python3 train.py --w-plus 1 --device cuda:2 --n-epochs 5000 --lr 1e2 
