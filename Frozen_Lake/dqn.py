import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer, DPKerasSGDOptimizer
import gym
import os
import time
import argparse
from irl import irl
from mdp import FrozenLakeEnv
from helper import get_state_rewards, get_transition_prob_matrix, to_s, pretty

parser = argparse.ArgumentParser()
parser.add_argument('--with_privacy', type=str, default='true')
parser.add_argument('--type', type=str, default='SGD')
parser.add_argument('--activation', type=str, default='relu')
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
        
def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


if config.activation == 'shoe':
    activation = 'tanh'
else:
    activation = 'relu'

if config.with_testing == 'true':
    testing = True
else:
    testing = False
    
if config.with_privacy == 'true':
    is_priv = True
else:
    is_priv = False

def get_model(inputs_shape):
    state_input = Input(inputs_shape, name='observation')
    x = Dense(64, activation=activation, kernel_initializer=tf.random_uniform_initializer(0, 0.01), bias_initializer=None)(state_input)
    x = Dense(16, activation=activation, kernel_initializer=tf.random_uniform_initializer(0, 0.01), bias_initializer=None)(x)
    out_value = Dense(4, activation=None, kernel_initializer=tf.random_uniform_initializer(0, 0.01), bias_initializer=None)(x)
    
    model = Model(inputs=state_input, outputs=out_value)

    return model

def save_ckpt(model, path):  
    model.save(path)

def load_ckpt(mode, path):
    return keras.models.load_model(path)

if is_priv:
    sigmas = [0]
    epsilons = ['inf']
else:
    sigmas = [94229, 150, 22.75, 9.893, 5.38, 3.031, 1.549, 1.0024]
    epsilons = [0.1, 0.105, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

lambd = .99  
e = 0.1  
max_steps = 200

l2_norm_clip = 1.0
noise_multiplier = 0.001
batch_size = 50
num_episodes = epochs = 15
num_test_episodes = 5
microbatches = 5
learning_rate = 0.15

if config.type == 'SGD':
    optimizer = DPKerasSGDOptimizer(l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
                        num_microbatches=microbatches, learning_rate=learning_rate)
else:
    optimizer = DPKerasAdamOptimizer(l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier,
                        num_microbatches=microbatches, learning_rate=learning_rate)

os.mkdir(config.env_size)
os.chdir(config.env_size)

for sigma_iter in range(len(sigmas)):
    path = str(epsilons[sigma_iter])
    os.mkdir(path)
    for k_iter in range(len(arg_list)):
        path = str(epsilons[sigma_iter]) + '/' + str(k_iter)
        os.mkdir(path)
        grid_size = grid_sizes[k_iter]
        li = arg_list[k_iter]
        env = FrozenLakeEnv(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7])
        rews = get_state_rewards(env)
        ground_r = np.array(rews)
        
        with open(path + '/' + 'ground_r.txt', 'w') as f:
            for j in ground_r.flatten():
                f.write('%f ' % j)
            f.write('\n')

        for avg_iter in range(10):
            env.reset()
            iter_start_time = time.time()
            noise_multiplier = sigmas[sigma_iter] / l2_norm_clip
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
            qnetwork = get_model([None, env.nS])
            qnetwork.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

            train_weights = qnetwork.trainable_weights
            all_episode_reward = []
            
            for i in range(num_episodes):
                s = env.reset()  
                rAll = 0

                states = []
                q_values = []
                targetQs = []

                for j in range(1, max_steps + 1):  

                    states.append(s)
                    allQ = qnetwork(np.asarray([to_one_hot(s, env.nS)], dtype=np.float32)).numpy()
                    a = np.argmax(allQ, 1)

                    if np.random.rand(1) < e:
                        a[0] = env.action_space.sample()
                        
                    s1, r, d, _ = env.step(a[0])

                    Q1 = qnetwork(np.asarray([to_one_hot(s1, env.nS)], dtype=np.float32)).numpy()
                    maxQ1 = np.max(Q1)  
                    targetQ = allQ
                    targetQ[0, a[0]] = r + lambd * maxQ1

                    targetQs.append(targetQ)

                    if not j%(batch_size):
                        with tf.GradientTape() as tape:
                            q_values = [qnetwork(np.asarray([to_one_hot(s, env.nS)], dtype=np.float32)) for s in states]
                            _losses = loss(q_values, targetQs)

                        grads_and_var = optimizer._compute_gradients(_losses, train_weights, tape=tape)
                        optimizer.apply_gradients(grads_and_var)

                        states = []
                        q_values = []
                        targetQs = []

                    rAll += r
                    s = s1
                    if d ==True:
                        e = 1. / ((i / 50) + 10)  
                        break

            discount = 0.999 
            P = [0] * env.nS
            for s in range(env.nS):
                allQ = qnetwork(np.asarray([to_one_hot(s, env.nS)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)
                P[s] = a[0]

            tns_prob = get_transition_prob_matrix(env)
            r = irl(env.nS, env.nA, tns_prob, P, discount, 1, 5)

            with open(path + '/' + str(avg_iter + 1) + '_r.txt', 'a') as f:
                    for j in r.flatten():
                        f.write('%.15f ' % j)
                    f.write('\n')

            iter_end_time = time.time()
            iter_time_taken = iter_end_time - iter_start_time
            with open(path + '/' + str(avg_iter + 1) + '_time.txt', 'w') as f:
                f.write('%.15f ' % iter_time_taken)

            if testing:
                all_episode_reward = []
                
                for i in range(num_test_episodes):
                    s = env.reset()
                    rAll = 0
                    for j in range(max_steps):  
                        allQ = qnetwork(np.asarray([to_one_hot(s, env.nS)], dtype=np.float32)).numpy()
                        a = np.argmax(allQ, 1)
                        s1, r, d, _ = env.step(a[0])
                        rAll += r * pow(lambd, j)
                        s = s1

                        if d: break

                    all_episode_reward.append(rAll)

                with open(path + '/' + str(avg_iter + 1) + '_r_test.txt', 'a') as f:
                        for j in all_episode_reward:
                            f.write('%.15f ' % j)
                        f.write('\n')

                


