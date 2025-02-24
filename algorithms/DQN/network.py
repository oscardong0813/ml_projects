import torch
import torch.nn as nn

class DQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        action_prob = self.linear3(x)

        return action_prob


