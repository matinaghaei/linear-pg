<!-- # Towards Principled, Practical Policy Gradient for Bandits and Tabular MDPs
Code for: [Towards Principled, Practical Policy Gradient for Bandits and Tabular MDPs](https://arxiv.org/abs/2405.13136). -->

## Installation
```
conda create -n ppg python=3.8
conda install scipy tqdm matplotlib pandas
pip install jax flax absl-py
```

##  Running Experiments

To reproduce linear bandit experiments in the exact setting, run: 
```
python linear_pg_experiments.py
```
To reproduce linear bandit experiments in the stochastic setting, run:
```
python linear_spg_experiments.py
```


For each linear bandit experiment, the corresponding plot can be generated in `plot_linear_experiment.ipynb`.
