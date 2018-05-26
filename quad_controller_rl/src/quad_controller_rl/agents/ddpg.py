"""DDPG agent."""
import pdb

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Add
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.3, sigma=0.3):
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
        self.memory = [None]*size
        self.is_full = False
        self.idx = 0
        
    def add(self, s, a, r, s_, d):
        if self.idx == self.size:
            self.idx = 0
            if not self.is_full:
                self.is_full = True
        
        self.memory[self.idx] = (s, a, r, s_, int(d))
        self.idx += 1
        
    def sample(self, batchsize=64):
        if self.is_full:
            choose_range = self.size
            choose_batch = batchsize
        else:
            choose_range = self.idx
            choose_batch = min(self.idx, batchsize)
        
        if choose_range == 0:
            return None
        
        indeces = np.random.choice(choose_range, choose_batch)
        return [self.memory[idx] for idx in indeces]


'''先使用论文中的参数尝试下'''
H1_UNITS = 64*7
H2_UNITS = 64*5
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
   
class Actor:
    def __init__(self, state_size, action_size, action_max):
        states_in = Input(shape=[state_size])
        h1 = Dense(units=H1_UNITS, activation='relu')(states_in)
        h1 = BatchNormalization()(h1)
        h2 = Dense(units=H2_UNITS, activation='relu')(h1)
        h2 = BatchNormalization()(h2)
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


class DDPG(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        self.TAU = 1e-3
        self.GAMMA = 0.99
        self.EXP_SIZE = int(1e6)
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
        self.actor_local = Actor(self.state_size, self.action_size, self.task.action_max)
        self.critic_local = Critic(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size, self.task.action_max)
        self.critic_target = Critic(self.state_size, self.action_size)

        # 复制local的weights到target上
        self.actor_target.model.set_weights(
            self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(
            self.critic_local.model.get_weights())

        '''初始化ounoise------------------------------------------------------------------------------------'''
        # 给输出网络输出action增加探索性噪音，考虑到重力，升力均值要为正
        mu = np.zeros(shape=self.action_size)
        mu[2] = 20.0
        self.noise_maker = OUNoise(size=self.action_size, mu=mu)
        self.noise_maker.reset()

        '''初始化Experience---------------------------------------------------------------------------------'''
        self.exp = Exp(self.EXP_SIZE)

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
            self.exp.add(self.last_state, self.last_action, reward, state, done)

        exps = self.exp.sample(self.BATCH_SIZE)
        if (exps is not None):
            self.learn(exps)        

        if done:
            # self.learn()
            self.reset_episode_vars()

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
            print("RandomPolicySearch.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                    self.t, score, self.best_score, self.noise_maker.state))  # [debug]
        #print(self.w)  # [debug: policy parameters]

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.TAU * local_weights + (1 - self.TAU) * target_weights
        target_model.set_weights(new_weights)