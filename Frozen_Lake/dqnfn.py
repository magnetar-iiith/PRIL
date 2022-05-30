import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import bisect
import os
import argparse
import numpy.random as rn
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
    
class hitorstandcontinuous:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_envs = 1
        self.cnt = 0
        self.length = 128

    def step2(self, action):
        self.cnt += 1
        observation, reward, done, info = self.env.step(action)
        self.state = observation
        if done:
            self.cnt = 0

        return self.state, reward, done, None

    def reset(self):
        self.cnt = 0
        self.state = self.env.reset()
        return self.state

    def render(self):
        raise NotImplementedError

    def seed(self, seed_value):
        np.random.seed(seed)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")

def kk(x, y):
    return np.exp(-abs(x-y))

def rho(x, y):
    return np.exp(abs(x-y)) - np.exp(-abs(x-y))

class noisebuffer:
    def __init__(self, m, sigma):
        self.buffer = []
        self.base = {}
        self.m = m
        self.sigma = sigma

    def sample(self, s):
        buffer = self.buffer
        sigma = self.sigma
            
        if len(buffer) == 0:
            v0 = np.random.normal(0, sigma)
            v1 = np.random.normal(0, sigma)
            v2 = np.random.normal(0, sigma)
            v3 = np.random.normal(0, sigma)
            self.buffer.append((s, v0, v1, v2, v3))
            return (v0, v1, v2, v3)
        else:
            idx = bisect.bisect(buffer, (s, 0, 0, 0, 0))
            if len(buffer) == 1:
                if buffer[0][0] == s:
                    return (buffer[0][1], buffer[0][2], buffer[0][3], buffer[0][4])
            else:
                if (idx <= len(buffer)-1) and (buffer[idx][0] == s):
                    return (buffer[idx][1], buffer[idx][2], buffer[idx][3], buffer[idx][4])
                elif (idx >= 1) and (buffer[idx-1][0] == s):
                    return (buffer[idx-1][1], buffer[idx-1][2], buffer[idx-1][3], buffer[idx-1][4])
                elif (idx <= len(buffer)-2) and (buffer[idx+1][0] == s):
                    return (buffer[idx+1][1], buffer[idx+1][2], buffer[idx+1][3], buffer[idx+1][4])
            
        if s < buffer[0][0]:
            mean0 = kk(s, buffer[0][0]) * buffer[0][1]
            mean1 = kk(s, buffer[0][0]) * buffer[0][2]
            mean2 = kk(s, buffer[0][0]) * buffer[0][3]
            mean3 = kk(s, buffer[0][0]) * buffer[0][4]
            var0 = 1 - kk(s, buffer[0][0]) ** 2
            var1 = 1 - kk(s, buffer[0][0]) ** 2
            var2 = 1 - kk(s, buffer[0][0]) ** 2
            var3 = 1 - kk(s, buffer[0][0]) ** 2
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            v2 = np.random.normal(mean2, np.sqrt(var2) * sigma)
            v3 = np.random.normal(mean3, np.sqrt(var3) * sigma)
            self.buffer.insert(0, (s, v0, v1, v2, v3))
        elif s > buffer[-1][0]:
            mean0 = kk(s, buffer[-1][0]) * buffer[0][1]
            mean1 = kk(s, buffer[-1][0]) * buffer[0][2]
            mean2 = kk(s, buffer[-1][0]) * buffer[0][3]
            mean3 = kk(s, buffer[-1][0]) * buffer[0][4]
            var0 = 1 - kk(s, buffer[-1][0]) ** 2
            var1 = var0
            var2 = var0
            var3 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            v2 = np.random.normal(mean2, np.sqrt(var2) * sigma)
            v3 = np.random.normal(mean3, np.sqrt(var3) * sigma)
            self.buffer.insert(len(buffer), (s, v0, v1, v2, v3))
        else:
            idx = bisect.bisect(buffer, (s, None, None, None, None))
            sminus, eminus0, eminus1, eminus2, eminus3 = buffer[idx-1]
            splus, eplus0, eplus1, eplus2, eplus3 = buffer[idx]
            mean0 = (rho(splus, s)*eminus0 + rho(sminus, s)*eplus0) / rho(sminus, splus)
            mean1 = (rho(splus, s)*eminus1 + rho(sminus, s)*eplus1) / rho(sminus, splus)
            mean2 = (rho(splus, s)*eminus2 + rho(sminus, s)*eplus2) / rho(sminus, splus)
            mean3 = (rho(splus, s)*eminus3 + rho(sminus, s)*eplus3) / rho(sminus, splus)
            var0 = 1 - (kk(sminus, s)*rho(splus, s) + kk(splus, s)*rho(sminus, s)) / rho(sminus, splus)
            var1 = var0
            var2 = var0
            var3 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            v2 = np.random.normal(mean2, np.sqrt(var2) * sigma)
            v3 = np.random.normal(mean3, np.sqrt(var3) * sigma)
            self.buffer.insert(idx, (s, v0, v1, v2, v3))
        return (v0, v1, v2, v3)

    def reset(self):
        self.buffer = []

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, sigma=0.4, hidden=16):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)
        self.sigma = sigma
        self.nb = noisebuffer(m, sigma)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.head(x)
        if sigma > 0:
            eps = [self.nb.sample(int(state)) for state in s]
            eps = torch.Tensor(eps)
            eps = eps.to(device)
            return x + eps
        else:
            return x

def select_action(state, test=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or test:
        with torch.no_grad():
            a = policy_net(torch.tensor(state, dtype=torch.float32, device=device))
            return a.max(1)[1].view(1, 1).long()
    else:
        return torch.tensor([[random.randrange(m)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.uint8)
    non_final_mask = torch.reshape(non_final_mask, (BATCH_SIZE, 1))
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).type(torch.float32)
    shap_nfs = non_final_next_states.shape[0]                                        
    non_final_next_states = torch.reshape(non_final_next_states, (shap_nfs, 1))
    
    state_batch = torch.cat(batch.state).type(torch.float32)
    state_batch = torch.reshape(state_batch, (BATCH_SIZE, 1))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device).type(torch.float32)
    next_state_values = torch.reshape(next_state_values, (BATCH_SIZE, 1))
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

if is_priv:
    sigmas = [0]
    epsilons = ['inf']
else:
    sigmas = [94229, 150, 22.75, 9.893, 5.38, 3.031, 1.549, 1.0024]
    epsilons = [0.1, 0.105, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

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
            _ = env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            env2 = hitorstandcontinuous(env)
            m = env2.action_space.n
            seed = 0
            np.random.seed(seed)
            torch.manual_seed(seed)
            sigma = sigmas[sigma_iter]
            policy_net = DQN(sigma=sigma).to(device)
            target_net = DQN(sigma=sigma).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

            optimizer = optim.RMSprop(policy_net.parameters())
            memory = ReplayMemory(1000)

            steps_done = 0
            episode_durations = []
            num_episodes = 21
            episodic_rewards = []
            for i_episode in range(num_episodes):
                state = int(env2.reset())
                state = torch.tensor([state], dtype=torch.long, device=device)
                total_reward = 0
                for t in count():
                    action = select_action(state)
                    next_state, reward, done, info = env2.step2(int(action.item()))
                    reward = torch.tensor([reward], device=device)

                    if not done:
                        next_state = int(next_state)
                        next_state = torch.tensor([next_state], dtype=torch.long, device=device)
                    else:
                        next_state = None

                    memory.push(state, action, next_state, reward)
                    total_reward += float(reward.squeeze(0).data)
                    state = next_state
                    optimize_model()
                    
                    if done:
                        episode_durations.append(t + 1)
                        break

                if i_episode % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                episodic_rewards.append(total_reward)
                policy_net.nb.reset()
                target_net.nb.reset()

            P = []

            for s in range(env.nS):
                state = int(s)
                state = torch.tensor([state], dtype=torch.long, device=device)
                a = select_action(state, test=True)
                a = a.max(1)[1].view(1, 1).long()
                P.append(a.item())

            discount = 0.999 
            
            tns_prob = get_transition_prob_matrix(env)
            r = irl(env.nS, env.nA, tns_prob, P, discount, 1, 5)

            with open(path + '/' + str(avg_iter + 1) + '_r.txt', 'a') as f:
                for j in r.flatten():
                    f.write('%.15f ' % j)
                f.write('\n')

            if testing:
                num_test_episodes = 5
                all_episode_reward = []
                for i_episode in range(num_test_episodes):
                    state = int(env2.reset())
                    state = torch.tensor([state], dtype=torch.long, device=device)
                    total_reward = 0
                    for t in count():
                        action = select_action(state)
                        next_state, reward, done, info = env2.step2(int(action.item()))
                        reward = torch.tensor([reward], device=device)
                        if not done:
                            next_state = int(next_state)
                            next_state = torch.tensor([next_state], dtype=torch.long, device=device)
                        else:
                            next_state = None                                                                                                             

                        total_reward += float(reward.squeeze(0).data) * pow(discount, t)

                        if done:
                            break

                    all_episode_reward.append(total_reward)

                with open(path + '/' + str(avg_iter + 1) + '_r_test.txt', 'a') as f:
                    for j in all_episode_reward:
                        f.write('%.15f ' % j)
                    f.write('\n')

