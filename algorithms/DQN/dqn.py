import gymnasium as gym
import torch
import torch.nn as nn
from network import DQNet
from gymnasium.spaces import Discrete
import torch.optim as optim

from collections import namedtuple, deque
import random
import numpy as np
import math

import matplotlib.pyplot as plt
import pprint
from itertools import zip_longest

class ReplayMemory(object):
    def __init__(self, replay_size):
        self.memory = deque([], maxlen = replay_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))
        self.data_set = 0

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        transitions = self.Transition(*zip(*random.sample(self.memory, batch_size)))

        return transitions

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, env, seed=42, lr=0.001, gamma=0.99, tau=0.005, epsilon=0.9, epsilon_end=0.05, epsilon_decay=1000,replay_size=10000, batch_size=128, num_episodes=600, target_update_freq=100):
        super(DQN, self).__init__()

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        # print('env ', env, env.action_space)
        if isinstance(env.action_space, Discrete):
            self.act_dim = env.action_space.n
            self.discrete_agent = True
        else:
            # print('here ')
            # print(env.action_space.shape[0])
            self.act_dim = env.action_space.shape[0]
            self.discrete_agent = False

        #parameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_size = replay_size
        self.num_ep = num_episodes
        self.batch_size = batch_size
        self.seed = seed
        self.tau = tau
        # self.target_update_freq = target_update_freq

        self.policy_net = DQNet(self.state_dim, self.act_dim)
        self.target_net = DQNet(self.state_dim, self.act_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.replay_memory = ReplayMemory(self.replay_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)

        self.total_ep_rewards = []
        self.ep_duration = []
        self.loss = [[] for i in range(self.num_ep)]

        self.logger = {'ep_rewards': self.total_ep_rewards,
                       'ep_duration': self.ep_duration,
                       'loss': self.loss
                       }

    def choose_action(self, state, time_sofar):
        '''
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        '''
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * math.exp(-1. * time_sofar/self.epsilon_decay)
        if sample <= eps_threshold:
            # print('choose action ', state)
            return self.env.action_space.sample()
        # print('choose action using policy net ', state)
        q_val = self.policy_net(torch.tensor(state))
        return torch.argmax(q_val).item()

    def learn(self):
        print('starting learn() for total epsidoes ', self.num_ep)
        for ep in range(self.num_ep):
            # print('ep ', ep+1)
            state = self.env.reset(seed=self.seed)[0]
            # print('state resetted ', state)
            ep_reward = 0
            done = False
            timestep_sofar = 1

            while not done:
                action = self.choose_action(state, timestep_sofar)
                # print('action ', action, ' choose at timestep ', timestep_sofar, ' with state ', state, type(state))
                # print('ep ', ep, ' time step ', timestep_sofar, ' action ', action.item())
                next_state, reward,terminated, truncated, _ = self.env.step(action)
                # print('stepping ', type(state), state)
                self.replay_memory.push(state, action, next_state, reward, max(terminated,truncated))
                timestep_sofar += 1
                # print('next state ', next_state, ' reward ', reward)
                ep_reward += reward
                done = terminated or truncated
                state = next_state
                # print('optimizing model ')
                self.optimize_model(ep)

                # if timestep_sofar % self.target_update_freq == 0:
                #     # print('target updated ')
                #     self.target_net.load_state_dict(self.policy_net.state_dict())
                '''
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
                '''
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)


            # self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)
            self.total_ep_rewards.append(ep_reward)
            self.ep_duration.append(timestep_sofar)
        # print('learning done ', self.total_ep_rewards, self.ep_duration)

    def optimize_model(self, ep):
        if len(self.replay_memory) < self.batch_size:
            # print('did not optimize ')
            self.loss[ep].append(0)
            return
        # print(self.replay_memory.memory)
        #sample random minibatch of transitions from memory
        batch = self.replay_memory.sample(self.batch_size) #
        # print('optimizing and batch sampled ')
        batch_state = torch.tensor(batch.state)
        batch_action = torch.tensor(batch.action).unsqueeze(1)
        # print('batched exp ', batch_state, batch_action)
        # print('batched action ', batch_action)
        batch_reward = torch.tensor(batch.reward)
        batch_next_state = torch.tensor(batch.next_state)
        batch_done = torch.FloatTensor(batch.done) #converting to tensor of 1s and 0s, instead of True and False

        #Compute target q vals
        with torch.no_grad():
            max_next_q = self.target_net(batch_next_state).max(1)[0] #selectiong best reard with max(1)[0]

        #compute y_j = r_j if terminated or truncated, r_j + max_next_q if not.
        target_q_vals = batch_reward + self.gamma * max_next_q * (1 - batch_done)

        #compute policy_q_vals
        policy_q_vals = self.policy_net(batch_state).gather(1, batch_action).squeeze()
        # print('both q vals ', policy_q_vals, target_q_vals)
        # print('policy q vals ', type(policy_q_vals), policy_q_vals, self.policy_net(batch_state))
        #perform a gradient descent step on (y_j - policy_q_vals)^2 Mean Squared Loss
        criterion = nn.MSELoss() #default use reduction = mean
        loss = criterion(policy_q_vals, target_q_vals)
        self.loss[ep].append(float(loss))
        #gradients are reset(zero_grad), and computed using backpropagation(backward, computes the derivative of the loss wrt to the parameters), and Model parameters are updated
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
        self.optimizer.step()

    def plot(self):
        eps = np.arange(0, self.num_ep, 1, dtype=int)
        data1 = self.total_ep_rewards
        data2 = [pi[-1] for pi in self.loss]
        print('data 2 ', data2)
        # max_len = len(max(data2, key=len))
        # padded_list = [list(zip_longest(sublist, range(max_len), fillvalue=np.nan))[0] for sublist in data2]
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(eps, data1)

        axs[1].plot(eps, data2)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    print('starting evn1')
    env1 = gym.make('CartPole-v1')
    dqn_agent1 = DQN(env1)
    dqn_agent1.learn()
    print('dqn loss ', dqn_agent1.loss)
    dqn_agent1.plot()
    # print(dqn_agent1.total_ep_rewards)
    # print('starting evn2')
    # env2 = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)
    # dqn_agent2 = DQN(env2)
    # dqn_agent2.learn()
    # print(dqn_agent2.total_ep_rewards)




