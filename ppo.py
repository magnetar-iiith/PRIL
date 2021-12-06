import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import gym
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

def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a    
    
# The Actor model performs the task of learning what action to take under a particular observed state of the environment 
def get_actor_model(inputs, num_states):
    state_input = Input(inputs)
    x = Dense(num_states, activation=activation)(state_input)
    x = Dense(16, activation=activation)(x)
    x = Dense(8, activation=activation)(x)
    out_actions = Dense(4, activation='softmax')(x)
    model = Model(inputs=state_input, outputs=out_actions)
    return model

# The output of Critic model is a real number indicating a rating i.e. Q-value of the action taken in the previous state
def get_critic_model(inputs, num_states):
    state_input = Input(inputs)
    x = Dense(num_states, activation=activation)(state_input)
    x = Dense(16, activation=activation)(x)
    x = Dense(8, activation=activation)(x)
    x = Dense(4, activation=activation)(x)
    out_value = Dense(1, activation=activation)(x)
    model = Model(inputs=state_input, outputs=out_value)
    return model


clipping_val = 0.2 # epsilon
critic_discount = 0.5 # to bring both losses to the same order of magnitude
gamma = 0.99 # discount factor
lmbda = 0.95 # smoothing parameter

# Advantage is defined as a way to measure how much better off we can be by taking a particular action when we are in a particular state
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def test_model(ppo_steps=128, max_test_iters=5):
    state = env.reset()
    state_dims = env.observation_space.n
    num_actions = env.action_space.n
    iters = 0
    all_episode_reward = []
    while iters < max_test_iters:
        states, actions, values, masks, rewards, actions_probs, actions_onehot, actions_log_prob, next_states = ([] for i in range(9))
        state_input = None
        rAll = 0
        for itr in range(ppo_steps):
            state_input = np.asarray([to_one_hot(int(state), state_dims)], dtype=np.float32)
            action_dist = actor_model(state_input)
            q_value = critic_model(state_input)
            action = np.argmax(action_dist[0, :])
            action_onehot = to_one_hot(action, num_actions)
            action_log_prob = np.log(action_dist[0, :][action])
            observation, reward, done, info = env.step(action)
            rAll += reward * pow(gamma, itr)

        all_episode_reward.append(rAll)

        iters += 1
    return all_episode_reward

def train_model(ppo_steps=128, batch_size=64, max_iters=30):
    state = env.reset()
    state_dims = env.observation_space.n
    num_actions = env.action_space.n
    iters = 0
    while iters < max_iters:
        states, actions, values, masks, rewards, actions_probs, actions_onehot, actions_log_prob, next_states = ([] for i in range(9))
        state_input = None
        for itr in range(ppo_steps):
            state_input = np.asarray([to_one_hot(int(state), state_dims)], dtype=np.float32)
            action_dist = actor_model(state_input)
            q_value = critic_model(state_input)
            action = np.argmax(action_dist[0, :])
            action_onehot = to_one_hot(action, num_actions)
            action_log_prob = np.log(action_dist[0, :][action])
            observation, reward, done, info = env.step(action)
            mask = not done
            states.append(state)
            actions.append(action)
            actions_onehot.append(action_onehot)
            values.append(q_value)
            masks.append(mask)
            rewards.append(reward)
            actions_probs.append(action_dist)
            actions_log_prob.append(action_log_prob)
            next_states.append(observation)
            state = observation
            if done: env.reset()

        with tf.GradientTape() as tape:
            q_value = critic_model(state_input)
            values.append(q_value)
            returns, advantages = get_advantages(values, masks, rewards)
            index = np.random.choice(ppo_steps, batch_size, replace=False)
            old_actions_log_prob = np.array(actions_log_prob)[index]
            new_states = np.array(states)[index]
            new_states_inp = list(map(lambda x: np.asarray([to_one_hot(int(x), env.nS)], dtype=np.float32), new_states))
            new_actions_dist = tf.vectorized_map(actor_model, elems=np.array(new_states_inp))
            new_actions = tf.vectorized_map(lambda x: tf.math.argmax(x[0, :]), new_actions_dist)
            new_actions_log_prob = tf.vectorized_map(lambda x: tf.math.log(x[0][0, :][x[1]]), (new_actions_dist, new_actions))
            new_advantages = advantages[index]
            new_advantages = tf.reshape(new_advantages, shape=(batch_size,))
            old_actions_log_prob = tf.convert_to_tensor(old_actions_log_prob, dtype=tf.float32)
            new_actions_log_prob = tf.convert_to_tensor(new_actions_log_prob, dtype=tf.float32)
            ratio = tf.math.exp(new_actions_log_prob - old_actions_log_prob)
            L1 = ratio * new_advantages
            L2 = tf.clip_by_value(ratio, 1 - clipping_val, 1 + clipping_val) * new_advantages
            
            def _loss_fn(val0, val1):
                return -tf.minimum(val0, val1)

            grads_and_vars = actor_optimizer.compute_gradients(lambda: _loss_fn(L1, L2), actor_train_weights, gradient_tape=tape)
            actor_optimizer.apply_gradients(grads_and_vars)
        with tf.GradientTape() as tape:
            q_values = tf.vectorized_map(critic_model, elems=np.array(new_states_inp))
            new_rewards = np.array(rewards)[index]
            h = tf.keras.losses.Huber()
            critic_loss = h(q_values, new_rewards)
        grad = critic_optimizer._compute_gradients(critic_loss, critic_train_weights, tape=tape)
        critic_optimizer.apply_gradients(grad)
        env.reset()
  
        iters += 1

    
l2_norm_clip = 1.0
batch_size = 64
microbatches = 4
learning_rate = 0.15

if is_priv:
    sigmas = [0]
    epsilons = ['inf']
else:
    sigmas = [94229, 150, 22.75, 9.893, 5.38, 3.031, 1.549, 1.0024]
    epsilons = [0.1, 0.105, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

os.mkdir(config.env_size)
os.chdir(config.env_size)

# Actor and Critic model interacts with the environement for a fixed number of steps and collect the experiences

for sigma_iter in range(len(sigmas)):
    noise_multiplier = sigmas[sigma_iter] / l2_norm_clip
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
        
        with open(path + '/' +  'ground_r.txt', 'w') as f:
            for j in ground_r.flatten():
                f.write('%f ' % j)
            f.write('\n')
        
        for avg_iter in range(10):
            actor_model = get_actor_model([None, env.nS], env.nS )           
            critic_model = get_critic_model([None, env.nS], env.nS)

            if config.type == 'SGD':
                actor_optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(l2_norm_clip=l2_norm_clip,
                                            noise_multiplier=noise_multiplier, num_microbatches=microbatches, learning_rate=learning_rate)
                critic_optimizer = SGD(learning_rate=0.01) 
            else:
                actor_optimizer = dp_optimizer.DPAdamOptimizer(l2_norm_clip=l2_norm_clip,
                                            noise_multiplier=noise_multiplier, num_microbatches=microbatches, learning_rate=learning_rate)
                critic_optimizer = Adam(learning_rate=0.01) 

            actor_train_weights = actor_model.trainable_weights
            critic_train_weights = critic_model.trainable_weights  

            train_model()

            discount = 0.999 
            P = [0] * env.nS
            for s in range(env.nS):
                state_input = np.asarray([to_one_hot(int(s), env.nS)], dtype=np.float32)
                action_dist = actor_model(state_input)
                a = np.argmax(action_dist[0, :])
                P[s] = a
            tns_prob = get_transition_prob_matrix(env)
            r = irl(env.nS, env.nA, tns_prob, P, discount, 1, 5)

            with open(path + '/' + str(avg_iter + 1) + '_r.txt', 'a') as f:
                for j in r.flatten():
                    f.write('%.15f ' % j)
                f.write('\n')

            if testing:
                all_episode_reward = test_model()
                with open(path + '/' + str(avg_iter + 1) + '_r_test.txt', 'a') as f:
                    for j in all_episode_reward:
                        f.write('%.15f ' % j)
                    f.write('\n')
