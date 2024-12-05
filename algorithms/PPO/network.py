import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Tanh
class CriticNet(nn.Module):
    '''
    critic network
    input:
    state dimensions,
    a list of neurons in hidden layer

    '''
    def __init__(self, state_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = state_dim

        for h in hidden_sizes:
            layer = nn.Linear(prev, h)
            self.layers.append(layer)
            prev = h
            # self.layers.append(nn.Tanh())
        self.final = nn.Linear(prev, 1)

    def forward(self, state):
        '''
        :param state: a pytorch tensor
        :return: an estimate of the expected return form the current state.
        '''
        activation = state
        for layer in self.layers:
            activation = F.tanh(layer(activation))
        output = self.final(activation)
        return output

class ActorNet(nn.Module):
    '''
    actor network
    input:
    state dim
    action dim
    a list of neurons in hidden layer
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes = HIDDEN_SIZES):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = state_dim

        for h in hidden_sizes:
            layer = nn.Linear(prev, h)
            self.layers.append(layer)
            prev = h
            # self.layers.append(nn.Tanh())
        self.final = nn.Linear(prev, action_dim)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        activation = state
        for layer in self.layers:
            activation = F.tanh(layer(activation))
        output = self.final(activation)
        return output


if __name__ == '__main__':

    critic = CriticNet(10)
    print(critic.layers)
    state = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
    print(critic.forward(state))
    actor = ActorNet(10, 2)
    print(actor.layers)
    print(actor.forward(state), actor.final)
