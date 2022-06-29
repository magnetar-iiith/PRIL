from collections import deque
import tensorlayer as tl
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import random
import copy
from make_optimizer import make_keras_optimizer_class

class MountainCarTrain:
    def __init__(self, env, reward_type, alphas, std, means, sigma, optimizer='SGD', activation='relu', dp_flag=0):
        self.clip = 0.2
        self.lmbda = 0.95
        self.l2_norm_clip = 1.0
        self.noise_multiplier = sigma
        self.microbatches = 4
        self.reward_type = reward_type
        self.alphas = alphas
        self.std = std
        self.means = means
        self.env = env
        self.gamma = 0.99
        self.epsilon = 0.3
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.02
        self.learningRate = 0.01
        self.replayBuffer = deque(maxlen=20000)
        self.optimizer = optimizer
        self.activation = activation
        self.dp_flag = dp_flag

        self.trainActorNetwork = self.createActorNetwork()
        self.trainCriticNetwork = self.createCriticNetwork()
        
        self.episodeNum = 10
        self.iterationNum = 100
        self.numPickFromBuffer = 32

        self.targetActorNetwork = self.createActorNetwork()
        self.targetCriticNetwork = self.createCriticNetwork()

        self.targetActorNetwork.set_weights(self.trainActorNetwork.get_weights())
        self.targetCriticNetwork.set_weights(self.trainCriticNetwork.get_weights())

    def createActorNetwork(self):
        state_shape = self.env.observation_space.shape
        state_input = Input(state_shape)
        if self.activation == 'shoe':
            x = Dense(16, activation='tanh')(state_input)
            x = Dense(8, activation='tanh')(x)
        else:
            x = Dense(16, activation='relu')(state_input)
            x = Dense(8, activation='relu')(x)
        out_actions = Dense(self.env.action_space.n, activation='softmax')(x)
        actor_model = Model(inputs=state_input, outputs=out_actions)
        
        return actor_model
    
    def createCriticNetwork(self):
        state_shape = self.env.observation_space.shape
        state_input = Input(state_shape)
        if self.optimizer == 'Shoe':
            x = Dense(16, activation='tanh')(state_input)
            x = Dense(8, activation='tanh')(x)
            out_value = Dense(1, activation='tanh')(x)
        else:
            x = Dense(16, activation='relu')(state_input)
            x = Dense(8, activation='relu')(x)
            out_value = Dense(1, activation='relu')(x)

        critic_model = Model(inputs=state_input, outputs=out_value)
        h = tf.keras.losses.Huber()
        if self.optimizer == 'Adam':
            critic_model.compile(loss=h, optimizer=Adam(learning_rate=self.learningRate))
        else:
            critic_model.compile(loss=h, optimizer=SGD(learning_rate=self.learningRate))    
        return critic_model
    
    def getBestAction(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        action_dist = self.trainActorNetwork.predict(state)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action=np.argmax(action_dist[0])

        log_prob = np.log(action_dist[0][action])
        return action, log_prob
    
    def getBestValue(self, state):
        value = self.trainCriticNetwork.predict(state)
        return value
    
    def getBestActionTestTime(self, state):
        action_dist = self.trainActorNetwork.predict(state)
        action=np.argmax(action_dist[0])
        log_prob = np.log(action_dist[0][action])
        return action, log_prob
    
    def getRandomAction(self, state):
        action = np.random.randint(0, 3)
        return action
    
    def runTestLoop(self, mode="random"):
        currentState = self.env.reset().reshape(1,2)
        rewards = []
        positions = []
        positions.append(currentState[0][0])
        max_position=-99
        for i in range(self.iterationNum):
            if mode == "random":
                bestAction = self.getRandomAction(currentState)
            else:
                bestAction, _ = self.getBestActionTestTime(currentState)
            new_state, reward, done, _ = self.env.step(bestAction, self.reward_type, self.alphas, self.std, self.means)
            new_state = new_state.reshape(1, 2)
            positions.append(new_state[0][0])
            # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]
            rewards.append(reward)
            # rewardSum = rewardSum + (reward * (self.gamma ** (i)))
            currentState = new_state
        return rewards, positions
    
    def get_advantages(self, values, masks, rewards, last_val):
        N = len(rewards)
        values = np.append(values, last_val)
        returns = []
        gae = 0
        
        for i in reversed(range(N)):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        advantages = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, advantages
    
    def trainFromBuffer_Boost(self, actor_optimizer):
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return
        
        samples = random.sample(self.replayBuffer,self.numPickFromBuffer)
        npsamples = np.array(samples)

        states_temp, actions_temp, rewards_temp, \
        newstates_temp, dones_temp, values_temp, \
        log_probs_temp = np.hsplit(npsamples, 7)

        states = np.concatenate((np.squeeze(states_temp[:])), axis = 0)
        actions = actions_temp.reshape(self.numPickFromBuffer,)
        rewards = rewards_temp.reshape(self.numPickFromBuffer,).astype(float)
        targets, target_log_probs = self.getBestActionTestTime(states)

        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~(copy.deepcopy(dones))
        notdones = notdones.astype(float)
        dones = dones.astype(float)

        values = values_temp.reshape(self.numPickFromBuffer,).astype(float)
        log_probs = log_probs_temp.reshape(self.numPickFromBuffer,).astype(float)

        # Q_futures = self.targetNetwork.predict(newstates).max(axis = 1)
        # targets[(np.arange(self.numPickFromBuffer), actions_temp.reshape(self.numPickFromBuffer,)
        # .astype(int))] = rewards * dones + (rewards + Q_futures * self.gamma)*notdones
        # self.trainNetwork.fit(states, targets, epochs=1, verbose=0)
        values_future = self.getBestValue(newstates)
        last_val = values_future[-1]
        returns, advantages = self.get_advantages(values_future, notdones, rewards, last_val)
        actions_future, log_probs_future = self.getBestActionTestTime(newstates)
        ratio = tf.math.exp(log_probs_future - log_probs)
        
        L1_ = tf.math.multiply(ratio, advantages)
        clip_ = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
        L2_ = tf.math.multiply(clip_, advantages)
        
        def _loss_fn(val0, val1):
                return -tf.minimum(val0, val1)

        with tf.GradientTape() as tape:
            _losses = lambda: _loss_fn(L1_, L2_)
            actor_weights = self.trainActorNetwork.trainable_weights
        grads_and_var = actor_optimizer._compute_gradients(_losses, actor_weights)
        actor_optimizer.apply_gradients(grads_and_var)

        self.trainCriticNetwork.fit(newstates, rewards, batch_size=self.numPickFromBuffer, verbose=False)

    def orginalTry(self, currentState, eps, actor_optimizer):
        rewardSum = 0
        max_position=-99
        for i in range(self.iterationNum):
            bestAction, log_prob = self.getBestAction(currentState)
            bestValue = self.getBestValue(currentState)
            new_state, reward, done, _ = self.env.step(bestAction, self.reward_type, self.alphas, self.std, self.means)
            new_state = new_state.reshape(1, 2)
            mask = not done
            # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]
            self.replayBuffer.append([currentState, bestAction, 
                                      reward, new_state, done, 
                                      bestValue, log_prob])
            self.trainFromBuffer_Boost(actor_optimizer)
            rewardSum += reward
            currentState = new_state
        self.targetActorNetwork.set_weights(self.trainActorNetwork.get_weights())
        self.targetCriticNetwork.set_weights(self.trainCriticNetwork.get_weights())
        print("Epsilon = {}, RewardSum = {}, MaxPosition = {}".format(max(self.epsilon_min, self.epsilon), rewardSum, max_position))
        self.epsilon -= self.epsilon_decay
    
    def start(self):
        if self.optimizer == 'Adam':
            DPKerasAdamOptimizer = make_keras_optimizer_class(tf.keras.optimizers.Adam)
            actor_optimizer = DPKerasAdamOptimizer(l2_norm_clip=self.l2_norm_clip,
                                                noise_multiplier=self.noise_multiplier, 
                                                num_microbatches=self.microbatches, 
                                                learning_rate=self.learningRate,
                                                dp_flag=self.dp_flag)
        else:
            DPKerasSGDOptimizer = make_keras_optimizer_class(tf.keras.optimizers.SGD)
            actor_optimizer = DPKerasSGDOptimizer(l2_norm_clip=self.l2_norm_clip,
                                                noise_multiplier=self.noise_multiplier, 
                                                num_microbatches=self.microbatches, 
                                                learning_rate=self.learningRate,
                                                dp_flag=self.dp_flag)
            
        for eps in range(self.episodeNum):
            currentState = self.env.reset().reshape(1,2)
            self.orginalTry(currentState, eps, actor_optimizer)