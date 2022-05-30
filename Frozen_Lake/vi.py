import numpy as np
import gym
import random
import tools
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy.random as rn
import copy
import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import os 
import argparse
from irl import irl
from mdp import FrozenLakeEnv
from helper import get_state_rewards, get_transition_prob_matrix, to_s, pretty

parser = argparse.ArgumentParser()
parser.add_argument('--with_privacy', type=str, default='true')
parser.add_argument('--env_size', type=str, default='5x5')
parser.add_argument('--with_testing', type=str, default='true')
config = parser.parse_args()


wind = 0.001
arg_list = []

if config.env_size == '10x10':
    grid_sizes = [10] * 12
    for i in range(97, 109):
        arg_list.append([None, "10x10_" + chr(i), True, wind, -1., -1., 10., None, 100.])
else:
    grid_sizes = [5] * 12
    for i in range(97, 100):
        arg_list.append([None, "5x5_" + chr(i), True, wind, -1., -1., 10., None, 100.])
        arg_list.append([None, "5x5_" + chr(i), True, wind, -1., -1., 10., None, 100.])
    for i in range(100, 106):
        arg_list.append([None, "5x5_" + chr(i), True, wind, -1., -1., 10., None, 100.])

if config.with_testing == 'true':
    testing = True
else:
    testing = False

if config.with_privacy == 'true':
    is_priv = True
else:
    is_priv = False

def compute_sigma(eps, sens):
    return (2.*sens*sens*np.log(1.25e4)/(eps*eps))

eps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 10]
sigmas = [compute_sigma(e, 1.05) for e in eps]

def compute_value_iteration(env, gamma=.99, theta=.0000000001, verbose=True, sigma=1):
    env.reset()
    nb_actions = env.action_space.n
    nb_states = env.observation_space.n
    V = np.zeros([nb_states])
    newV = V.copy()
    P = np.zeros([nb_states], dtype=int)
    iteration = 0
    while True:
        delta = 0
        for s in range (0, nb_states):
            action_vals = []      
            for action in range(nb_actions):
                temp_val = 0
                for i in range(len(env.P[s][action])):
                    prob, next_state, reward, done = env.P[s][action][i]
                    sa_value = prob * (reward + gamma * V[next_state]) + is_priv * np.random.normal(0,sigma,1)
                    temp_val += sa_value
                action_vals.append(temp_val)      #the value of each action
                bestA = np.argmax(np.asarray(action_vals))   # choose the action which gives the maximum value
                P[s] = bestA
                newV[s] = action_vals[bestA] 
            delta = max(delta, np.abs(newV[s] - V[s]))
        V = newV.copy()
        iteration += 1
        if delta < theta or iteration > 10000:
            if verbose:
                print (iteration,' iterations done')
            break
    return V, P

discount = 0.999
theta = 0.15
lambd = 0.99

os.mkdir(config.env_size)
os.chdir(config.env_size)

for sigma_iter in range(len(sigmas)):
    path = str(eps[sigma_iter])
    os.mkdir(path)
    for i in range(len(arg_list)):
        grid_size = grid_sizes[i]
        li = arg_list[i]
        env = FrozenLakeEnv(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7])
        env.reset()
        rews = get_state_rewards(env)
        ground_r = np.array(rews)
        
        with open(path + "/" + str(i + 1) +  '_ground_r.txt', 'w') as f:
            for j in ground_r.flatten():
                f.write('%f ' % j)
            f.write('\n')
        
        for avg_iter in range(10):
            env.reset()
            V, P = compute_value_iteration(env, gamma=discount, theta=theta, verbose=True, sigma=sigmas[sigma_iter])

            tns_prob = get_transition_prob_matrix(env)
            r = irl(env.nS, env.nA, tns_prob, P, discount, 1, 5)
            
            with open(path + "/" + str(i + 1) + "_" + str(avg_iter + 1) + '_r.txt', 'a') as f:
                for j in r.flatten():
                    f.write('%.15f ' % j)
                f.write('\n')
            
            if testing:
                all_episode_reward = []
                num_test_episodes = 5
                max_steps = 200
                for i in range(num_test_episodes):
                    s = env.reset()
                    rAll = 0
                    for j in range(max_steps):  
                        a = P[s]
                        s1, r, d, _ = env.step(a)
                        rAll += r * pow(lambd, j)
                        s = s1

                        if d: break

                    all_episode_reward.append(rAll)

                with open(path + '/' + str(avg_iter + 1) + '_r_test.txt', 'a') as f:
                        for j in all_episode_reward:
                            f.write('%.15f ' % j)
                        f.write('\n')
