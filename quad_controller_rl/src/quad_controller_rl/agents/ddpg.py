"""DDPG agent."""
import pdb

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Add
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import layers, models, optimizers
from keras import backend as K


import os
import pandas as pd
from quad_controller_rl import util

'''----------------------------------------------
        class DDPG 在160行
   ----------------------------------------------'''

class CSVSaver:
    def __init__(self):
        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'height', 'target_distance', 'total_reward']  # specify columns to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]
    
    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        series = [self.episode_num, *stats]
        df_stats = pd.DataFrame([series], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only
        self.episode_num += 1

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
        

class Exp:
    '''经验池'''
    def __init__(self, size=int(1e6)):
        self.size=size
        self.memory = [(0,0,0,0,0,-np.inf)]*size
        self.is_full = False
        self.idx = 0
        
    def add(self, s, a, r, s_, d, ddrw):
        if self.idx == self.size:
            self.is_full = True
        
        if self.is_full:
            self.idx = int(np.random.choice(self.size, 1))
        
        self.memory[self.idx] = (s, a, r, s_, int(d), ddrw)
        self.idx += 1
    
    def simple_add(self, s,a,r,s_,d,ddrw):
        if self.idx >= self.size:
            if not self.is_full:
                self.is_full = True
            self.idx = 0
        self.memory[self.idx] = (s, a, r, s_, int(d), ddrw)
        self.idx += 1
    
    def random_sample(self, batchsize=64):
        if self.idx == 0:
            return None
        
        if self.is_full:
            indeces = np.random.choice(self.size, batchsize)
        else:
            indeces = np.random.choice(self.idx, min(self.idx, batchsize))
        
        return [self.memory[idx] for idx in indeces]

    def sample(self, batchsize=64):
        if self.idx == 0:
            return None

        self.memory.sort(key=lambda item: item[-1], reverse=True)
        if not self.is_full:
            # 还没填满，随机选择
            if self.idx > batchsize:
                rand_indeces = np.random.choice(self.idx, batchsize//2)
                good_indeces = range(batchsize//2)
                indeces = [*good_indeces, * rand_indeces]
            else:
                indeces = range(self.idx)
        else:
            # 已经填满
            # 选择前20%里的batchsize/4个作为优秀案例学习
            good_indeces = np.random.choice(int(self.size*0.2), batchsize//4)

            # 选择最后20%的batchsize/4个作为反面教材学习
            bad_indeces = np.random.choice(int(self.size*0.2), batchsize//4) + int(self.size*0.8)

            # 中间样本（更广阔的探索）
            rand_indeces = np.random.choice(int(self.size*0.6), batchsize//2) + int(self.size*0.2)
            
            indeces = [*good_indeces, *rand_indeces, *bad_indeces]

        return [self.memory[idx] for idx in indeces]
        
    def _sample(self, batchsize=64):
        if self.is_full:
            choose_range = self.size
            choose_batch = batchsize
        else:
            choose_range = self.idx
            choose_batch = min(self.idx, batchsize)
        
        if choose_range == 0:
            return None
            
        rand_indeces = np.random.choice(choose_range, int(choose_batch * 0.8 // 2))
        good_indeces = range(int(choose_batch * 0.2 // 2 + 1))
        indeces = [*good_indeces, *rand_indeces]
        return [self.memory[idx] for idx in indeces]

    def get_size(self):
        return self.size if self.is_full else self.idx


'''先使用论文中的参数尝试下'''
H1_UNITS = 400
H2_UNITS = 300
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
   
class Actor:
    def __init__(self, state_size, action_size, action_max):
        states_in = Input(shape=[state_size])
        h1 = Dense(units=H1_UNITS, activation='linear')(states_in)
        h1 = BatchNormalization()(h1)
        h1 = Activation('relu')(h1)

        h2 = Dense(units=H2_UNITS, activation='linear')(h1)
        h2 = BatchNormalization()(h2)
        h2 = Activation('relu')(h2)

        raw_actions = Dense(units=action_size, activation='tanh')(h2)
        actions = Lambda(lambda ra: ra*action_max)(raw_actions)

        self.model = Model(inputs=states_in, outputs=actions)
        
        # TODO：以下梯度策略算法没搞明白
        action_gradients = Input(shape=[action_size])
        loss = K.mean(-action_gradients * actions)
        # Incorporate any additional losses here (e.g. from regularizers)
        optimizer = Adam(lr=ACTOR_LR)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    def __init__(self, state_size, action_size):
        '''根据论文（P11），actions在第二隐层引入'''
        states_in = Input(shape=[state_size])
        actions_in = Input(shape=[action_size])

        s1 = Dense(units=H1_UNITS, activation='relu')(states_in)

        s_merge = Dense(units=H2_UNITS, activation='linear')(s1)
        a_merge = Dense(units=H2_UNITS, activation='linear')(actions_in)
        sa = Add()([s_merge, a_merge])
        sa = BatchNormalization()(sa)
        h1 = Activation('relu')(sa)

        h2 = Dense(units=H2_UNITS, activation='relu')(h1)
        h2 = BatchNormalization()(h2)

        q_values = Dense(units=1, activation='linear')(h2)

        self.model = Model(inputs=[states_in, actions_in], outputs=q_values)
        optimizer = Adam(lr=CRITIC_LR)
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(q_values, actions_in)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class Actor2:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic2:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


class DDPG(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        self.TAU = 1e-3
        self.GAMMA = 0.99
        self.EXP_SIZE = int(5e3)
        self.BATCH_SIZE = 64

        '''接受任务------------------------------------------------------------------------------------'''
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low
        
        '''限制动作和状态空间------------------------------------------------------------------------------------'''
        self.state_size = 3  # position only 不考虑方向
        self.action_size = 3  # force only 不考虑力矩
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))

        '''创建Actor和Critic俩基友------------------------------------------------------------------------------------'''
        self.actor_local = Actor2(self.state_size, self.action_size, self.task.action_space.low[:3], self.task.action_space.high[:3])
        self.critic_local = Critic2(self.state_size, self.action_size)
        self.actor_target = Actor2(self.state_size, self.action_size, self.task.action_space.low[:3], self.task.action_space.high[:3])
        self.critic_target = Critic2(self.state_size, self.action_size)

        # 复制local的weights到target上
        self.actor_target.model.set_weights(
            self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(
            self.critic_local.model.get_weights())

        '''初始化ounoise------------------------------------------------------------------------------------'''
        # 给输出网络输出action增加探索性噪音，考虑到重力，升力均值要为正
        mu = np.zeros(shape=self.action_size)
        mu[2] = 18.0
        self.noise_maker = OUNoise(size=self.action_size, mu=mu)
        self.noise_maker.reset()

        '''初始化Experience---------------------------------------------------------------------------------'''
        self.exp = Exp(self.EXP_SIZE)
        self.last_reward = 0
        self.last_delta_reward = 0
        self.last_dd_reward = 0

        '''初始化csv记录器-----------------------------------------------------------------------------------------'''
        self.csv_saver = CSVSaver()

        # Score tracker
        self.best_score = -np.inf

        self.t = 0

        # Episode variables
        self.reset_episode_vars()
    
    def preprocess_states(self, states):
        """Reduce state vector to relevant dimensions."""
        return np.array(
            [state[0:self.state_size] for state in states])

    def postprocess_actions(self, actions):
        """Return complete action vector."""
        def process(action):
            complete_action = np.zeros(self.task.action_space.shape)
            complete_action[0:self.action_size] = action
            return complete_action      

        return np.array([process(action) for action in actions])

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.noise_maker.reset()

        # each episode should update noise

        # print('agent reset_episode\n'*5)

    def step(self, state, reward, done):
        # 修正重力影响
        # curr_z = state[2]
        # target_z = self.task.target_point[2]
        # vec_z = target_z - curr_z
        # mu_z = min(vec_z**3*0.05 - 3, 25)
        # self.noise_maker.mu[2] = mu_z

        raw_state = np.array(state)

        # Transform state vector
        state = (state - self.task.observation_space.low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector

        state = self.preprocess_states(state)
        # Choose an action
        action = self.act(state)
            
        self.t += 1
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1

            delta_reward = reward - self.last_reward
            dd_reward = delta_reward - self.last_delta_reward
            if done:
                dd_reward = self.last_dd_reward

            self.exp.simple_add(self.last_state, self.last_action, reward, state, done, dd_reward)
            self.last_reward = reward
            self.last_delta_reward = delta_reward
            self.last_dd_reward = dd_reward

        exps = self.exp.random_sample(self.BATCH_SIZE)
        if (exps is not None):
            self.learn(exps)        

        if done:
            self.reset_episode_vars()

            # 存储csv
            height = raw_state[2]
            dist = np.linalg.norm(raw_state[:3] - self.task.target_point)
            self.csv_saver.write_stats([height, dist, reward])

        self.last_state = state
        self.last_action = action

        action = self.postprocess_actions(action)
        return action

    def act(self, states):
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)

        return actions + self.noise_maker.sample()

    def learn(self, exps):
        # Learn by random policy search, using a reward-based score
        score = self.total_reward / float(self.count) if self.count else 0.0
        if score > self.best_score:
            self.best_score = score
            # TODO: save weights
        
        '''用经验值batch更新策略和权重'''
        states = np.vstack([e[0] for e in exps if e is not None])
        actions = np.array([e[1] for e in exps if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e[2] for e in exps if e is not None]).astype(np.float32).reshape(-1, 1)
        next_states = np.vstack([e[3] for e in exps if e is not None])
        dones = np.array([e[4] for e in exps if e is not None]).astype(np.uint8).reshape(-1, 1)

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        Q_targets = rewards + self.GAMMA * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        if self.t % 50 == 0:
            print("Exp size: {:8d} t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                    self.exp.get_size(), self.t, score, self.best_score, self.noise_maker.state))  # [debug]
        #print(self.w)  # [debug: policy parameters]

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.TAU * local_weights + (1 - self.TAU) * target_weights
        target_model.set_weights(new_weights)