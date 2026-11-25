import torch 
import torch.nn as nn


class DQNetwork(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self , states):
        features = self.conv(states)
        features = torch.flatten(features, 1)
        q_values = self.fc(features)
        return q_values
