# Communication-Efficient Laplace Mechanism for Differential Privacy via Random Quantization

This repository contains the implementation of the *dyadic quantized Laplace(DQL)* mechanism introduced in the following paper:
- Ali Moradi Shahmiri, Chih Wei Ling, and Cheuk Ting Li, “[Communication-efficient Laplace mechanism for differential privacy via random quantization](https://arxiv.org/abs/2309.06982)”, arXiv pre-print, 2023.

The encoding and decoding algorithms are implemented in the `DQLMechanism` class in `privacy.py` and the auxiliary functions used by the mechanism are in `dql_utils.py`.

## Experiment

In the experiment described in the paper, the input of the mechanism is assumed to be x ~ Unif(-1, 1). `preprocess_dql.py` runs the experiment for different values of the mechanism's parameters and computes the MSE and the average communication cost incurred by the mechanism. `plot_dme_scalar.py` then loads these values and plots the mechanism's database and decoder privacy budgets against the MSE for a given communication budget. 

### Preprocess
```
python preprocess_dql.py --eps_from 1 --eps_to 11 --eps_subdiv 20 --num_clients 30000
```

### Plot
```
python plot_dme_scalar.py --eps_from 1 --eps_to 11 --eps_subdiv 20 --num_clients 30000 --comm_budget 5
```
