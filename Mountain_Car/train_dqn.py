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
        self.l2_norm_clip = 1.0
        self.noise_multiplier = sigma
        self.microbatches = 4
        self.reward_type = reward_type
        self.alphas = alphas
        self.std = std
        self.means = means
        self.env = env
        self.optimizer = optimizer
        self.activation = activation
        self.dp_flag = dp_flag
        self.gamma = 0.99
        self.epsilon = 0.3
        self.epsilon_decay = 0.02
        self.epsilon_min = 0.02
        self.learningRate = 0.01
        self.replayBuffer = deque(maxlen=20000)
        self.trainNetwork = self.createNetwork()
        self.episodeNum = 10
        self.iterationNum = 100
        self.numPickFromBuffer = 32
        self.targetNetwork = self.createNetwork()
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())
    def get_model(self, inputs_shape):
        state_input = Input(inputs_shape, name='observation')
        # print("Inputs shape: ", inputs_shape)
        x = Dense(16, activation='relu', kernel_initializer=tf.random_uniform_initializer(0, 0.01), 
                  bias_initializer=None)(state_input)
        x = Dense(8, activation='relu', kernel_initializer=tf.random_uniform_initializer(0, 0.01), 
                  bias_initializer=None)(x)
        out_value = Dense(self.env.action_space.n, activation=None, 
                          kernel_initializer=tf.random_uniform_initializer(0, 0.01), 
                          bias_initializer=None)(x)
        model = Model(inputs=state_input, outputs=out_value)
        return model
    def createNetwork(self):
        # model = models.Sequential()
        state_shape = self.env.observation_space.shape
        print("States shape: ", state_shape)
        state_input = Input(state_shape, name='observation')
        # state_input = Input([None, 2], name='observation')
        if self.activation == 'shoe':
            x = Dense(16, activation='tanh')(state_input)
            x = Dense(8, activation='tanh')(x)
        else:
            x = Dense(16, activation='relu')(state_input)
            x = Dense(8, activation='relu')(x)
        
        out_actions = Dense(self.env.action_space.n, activation='linear')(x)
        # out_actions = Dense(self.env.action_space.n, activation=None)(x)
        model = Model(inputs=state_input, outputs=out_actions)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        # dp_sum_query = gaussian_query.GaussianSumQuery(1.0e9, 0.0)
        if self.optimizer == 'Adam':
            DPKerasAdamOptimizer = make_keras_optimizer_class(tf.keras.optimizers.Adam)
            optimizer = DPKerasAdamOptimizer(l2_norm_clip=self.l2_norm_clip, noise_multiplier=self.noise_multiplier, 
                                            num_microbatches=self.microbatches, learning_rate=self.learningRate,
                                            dp_flag=1) #  dp_sum_query=dp_sum_query,
        else:
            DPKerasSGDOptimizer = make_keras_optimizer_class(tf.keras.optimizers.SGD)
            optimizer = DPKerasSGDOptimizer(l2_norm_clip=self.l2_norm_clip, noise_multiplier=self.noise_multiplier, 
                                         num_microbatches=self.microbatches, learning_rate=self.learningRate,
                                         dp_flag=1) #  dp_sum_query=dp_sum_query,
        
        # optimizer = VectorizedDPAdam(l2_norm_clip=self.l2_norm_clip, noise_multiplier=self.noise_multiplier, 
        #                                  num_microbatches=self.microbatches, learning_rate=self.learningRate) #  dp_sum_query=dp_sum_query,

        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        return model
    def getBestAction(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action=np.argmax(self.trainNetwork.predict(state)[0])
        return action
    def getBestActionTestTime(self, state):
        action=np.argmax(self.trainNetwork.predict(state)[0])
        return action
    def getRandomAction(self, state):
        action = np.random.randint(0, 3)
        return action
    def runTestLoop(self, mode="random"):
        currentState = self.env.reset().reshape(1,2)
        rewards = []
        positions = []
        positions.append(currentState[0][0])
        # rewardSum = 0
        max_position=-99
        for i in range(self.iterationNum):
            if mode == "random":
                bestAction = self.getRandomAction(currentState)
            else:
                bestAction = self.getBestActionTestTime(currentState)
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
    def trainFromBuffer_Boost(self):
        import copy
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return
        samples = random.sample(self.replayBuffer,self.numPickFromBuffer)
        npsamples = np.array(samples)
        states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
        states = np.concatenate((np.squeeze(states_temp[:])), axis = 0)
        rewards = rewards_temp.reshape(self.numPickFromBuffer,).astype(float)
        targets = self.trainNetwork.predict(states)
        trains = self.trainNetwork.predict(states)
        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~(copy.deepcopy(dones))
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.targetNetwork.predict(newstates).max(axis = 1)
        targets[(np.arange(self.numPickFromBuffer), actions_temp.reshape(self.numPickFromBuffer,).astype(int))] = rewards * dones + (rewards + Q_futures * self.gamma)*notdones
        # print("targets: ", targets)
        # self.trainNetwork.fit(states, targets, epochs=1, verbose=0)
        train_qs = self.trainNetwork.predict(states).max(axis = 1)
        trains[(np.arange(self.numPickFromBuffer), actions_temp.reshape(self.numPickFromBuffer,).astype(int))] = train_qs
        ####################################################################################################
        with tf.GradientTape() as tape:
            _losses = self.trainNetwork.loss(trains, targets)
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
            _losses.numpy()
            _losses = _losses.reshape(1, -1)
            # print("Losses: ", _losses, type(_losses))
            train_weights = self.trainNetwork.trainable_weights
            # print("Trainable weights: ",train_weights, type(train_weights[0]))
        for val in train_weights:
          val.numpy()
        grads_and_var = self.trainNetwork.optimizer._compute_gradients(_losses, train_weights, tape=tape)
        # grads_and_var = self.trainNetwork.optimizer.get_gradients(_losses, train_weights)
        name="temp"
        # grads_and_var = _compute_gradients(_losses, train_weights, self.microbatches, name, self.l2_norm_clip, self.noise_multiplier, tape=tape)
        self.trainNetwork.optimizer.apply_gradients(grads_and_var)
        ####################################################################################################
    def trainFromBuffer(self):
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return
        samples = random.sample(self.replayBuffer,self.numPickFromBuffer)
        states = []
        newStates=[]
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            newStates.append(new_state)
        newArray = np.array(states)
        states = newArray.reshape(self.numPickFromBuffer, 2)
        newArray2 = np.array(newStates)
        newStates = newArray2.reshape(self.numPickFromBuffer, 2)
        targets = self.trainNetwork.predict(states)
        new_state_targets=self.targetNetwork.predict(newStates)
        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i += 1
        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)
    def orginalTry(self, currentState, eps):
        rewardSum = 0
        max_position=-99
        for i in range(self.iterationNum):
            bestAction = self.getBestAction(currentState)
            # print("currentState: ", currentState)
            new_state, reward, done, _ = self.env.step(bestAction, self.reward_type, self.alphas, self.std, self.means)
            new_state = new_state.reshape(1, 2)
            # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]
            self.replayBuffer.append([currentState, bestAction, reward, new_state, done])
            # Or you can use self.trainFromBuffer_Boost(), it is a matrix wise version for boosting 
            # self.trainFromBuffer()
            # if done:
            #     print("Step no = {}, observation = {}, reward = {}, done = {}".format(i, new_state, reward, done))
            self.trainFromBuffer_Boost()
            rewardSum += reward
            currentState = new_state
        #Sync
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())
        # print("Step no = {}, observation = {}, reward = {}, done = {}".format(i, new_state, reward, done))
        print("Epsilon = {}, RewardSum = {}, MaxPosition = {}".format(max(self.epsilon_min, self.epsilon), rewardSum, max_position))
        self.epsilon -= self.epsilon_decay
    def start(self):
        for eps in range(self.episodeNum):
            # print("Episode:", eps)
            currentState = self.env.reset().reshape(1,2)
            self.orginalTry(currentState, eps)
