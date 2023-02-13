import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__ (self, lr, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5184, 84)
        self.fc2 = nn.Linear(84, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0') 
        self.to(self.device)

    def forward(self, x):
        x = T.tensor(x, dtype=T.float)
        x = x.to('cuda:0')
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = T.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
