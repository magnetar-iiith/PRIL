# Mountain Car Experiments

## Setup and installation instructions
All experiments performed are performed within a conda environment with python=3.8
We use the following external libraries:
- numpy
- pandas
- TensorFlow v2.5.0
- TensorFlow Privacy v0.5.2
- gym
- cvxopt
- scipy

## How to run experiments
```python
python main.py
```
### Arguments
- `--algo` : name of the algorithm (either DQN or PPO) 
- `--with_privacy` : if set to true, the code will run private algorithm
- `--optim` : optimizer type (either SGD or Adam)
- `--activation` : activation function of model (either relu or shoe)
