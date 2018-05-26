"""DDPG agent."""
import pdb

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent

class Exp:
    def __init__(self, size=1000):
        self.size=size
        self.memory = np.zeros([self.size, 5])
        self.is_full = False
        self.idx = 0
        
    def add(self, s, a, r, s_, d):
        if self.idx == self.size:
            self.idx = 0
            if not self.is_full:
                self.is_full = True
        
        self.memory[self.idx] = np.array([s, a, r, s_, int(d)])
        self.idx += 1
        
    def sample(self, batchsize=64):
        if self.is_full:
            choose_range = self.size
            choose_batch = batchsize
        else:
            choose_range = self.idx
            choose_batch = min(self.idx, batchsize)
        
        choose_idx = np.random.choice(choose_range, choose_batch)
        return self.memory[choose_idx]
        

class DDPG(BaseAgent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low
        
        
        '''------------------------------------------------------------------------------------'''
        # 新加入
        # Constrain state and action spaces
#         self.state_size = 3  # position only
#         self.action_size = 3  # force only
#         print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
#             self.task.observation_space.shape, self.task.action_space.shape,
#             self.state_size, self.action_size))
        
        '''------------------------------------------------------------------------------------'''
        # Policy parameters
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range[:self.action_size] / (2 * self.state_size)).reshape(1, -1))  # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode_vars()
    
    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[0:self.state_size]  # position only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[0:self.action_size] = action  # linear force only
        return complete_action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def step(self, state, reward, done):
        # Transform state vector
        state = (state - self.task.observation_space.low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector

        state = np.array([self.preprocess_state(s) for s in state])
        # Choose an action
        action = self.act(state)
        action = np.array([self.postprocess_action(a) for a in action])
            
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        return action

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        #print(action)  # [debug: action vector]
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        score = self.total_reward / float(self.count) if self.count else 0.0
        if score > self.best_score:
            self.best_score = score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)

        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        print("RandomPolicySearch.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                self.count, score, self.best_score, self.noise_scale))  # [debug]
        #print(self.w)  # [debug: policy parameters]
