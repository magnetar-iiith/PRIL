# Frozen Lake Experiments

## Setup and installation instructions
All experiments performed are performed within a conda environment with python=3.8
We use the following external libraries:
- numpy
- pandas
- TensorFlow v2.5.0
- TensorFlow Privacy v0.5.2
- PyTorch v1.9.0
- gym
- cvxopt
- plotly

## How to run experiments
### 1. DQN
```python
python dqn.py
```
### Arguments
- `--with_privacy` : if set to true, the code will run private DQN
- `--type` : optimizer type (either SGD or Adam)
- `--activation` : activation function of model (either relu or shoe)
- `--env_size` : grid size of FrozenLake environment (either 5x5 or 10x10)
- `--with_testing` : if set to true, the code will run tests too

### 2. PPO
```python
python ppo.py
```
### Arguments
- `--with_privacy` : if set to true, the code will run private PPO
- `--type` : optimizer type (either SGD or Adam)
- `--activation` : activation function of model (either relu or shoe)
- `--env_size` : grid size of FrozenLake environment (either 5x5 or 10x10)
- `--with_testing` : if set to true, the code will run tests too

### 3. VI
```python
python vi.py
```
### Arguments
- `--with_privacy` : if set to true, the code will run private value iteration
- `--env_size` : grid size of FrozenLake environment (either 5x5 or 10x10)
- `--with_testing` : if set to true, the code will run tests too

### 4. DQNFN
```python
python dqnfn.py
```
### Arguments
- `--with_privacy` : if set to true, the code will run private DQNFN
- `--env_size` : grid size of FrozenLake environment (either 5x5 or 10x10)
- `--with_testing` : if set to true, the code will run tests too