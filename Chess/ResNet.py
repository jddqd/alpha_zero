import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
import random


# Adapté à 22 channels, tout est encodé 

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super(ResNet, self).__init__()

        self.device = device
        self.to(device)

        self.startBlock = nn.Sequential(
            nn.Conv2d(12, num_hidden, kernel_size = 3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
            )

        self.backBone = nn.ModuleList([ResBlock(num_hidden) for i in range(num_resBlocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*game.row_count * game.column_count, game.action_size)
            )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size = 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*game.row_count*game.column_count, 1),
            nn.Tanh()
            )
    
    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size = 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + x)