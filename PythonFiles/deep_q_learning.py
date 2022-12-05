import os
import torch as T
import torch.cuda as Tc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__ (self, lr, n_actions, name, input_dims, checkpoint_dir):
    #     super(DeepQNetwork, self).__init__()
    #     # checkpoint directory
        self.checkpoint_dir = checkpoint_dir 
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

    #     self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)  # input dims[0] to liczba kanałów
    #                                                             # 32 - filtry, 8 - kernel size, stride - liczba pixeli do przesuniecia kernela
    #     self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    #     self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    #     self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

    #     # obliczanie rozmiaru fully connected layer
    #     fc_input_dims = self.calculate_conv_output_dims(input_dims)
    #     self.fc1 = nn.Linear(fc_input_dims, 512)                # 512 outputow
    #     self.fc2 = nn.Linear(512, n_actions)

    #     self.optimizer = optim.RMSprop(self.parameters(0), lr=lr)
    #     self.loss = nn.MSELoss()
    #     self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') 
    #     self.to(self.device)

    # def calculate_conv_output_dims(self, input_dims):
    #     state = T.zeros(1, *input_dims)
    #     dims = self.conv1(state)
    #     dims = self.conv2(dims)
    #     dims = self.conv3(dims)
    #     return int(np.prod(dims.size()))
    
    # #obliczanie feedforward
    # def forward(self, state):
    #     conv1 = F.relu(self.conv1(state))
    #     conv2 = F.relu(self.conv1(conv1))
    #     conv3 = F.relu(self.conv1(conv2))

    #     #rozmiar conv3 -  batch_size * filter_num * width * height (wyjsciowego obrazu)
    #     conv_state = conv3.view(conv3.size()[0], -1)
    #     flat1 = F.relu(self.fc1(conv_state))
    #     actions = self.fc2(flat1)

    #     return actions

    #def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5184, 84)
        self.fc2 = nn.Linear(84, n_actions)


        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0') 
        self.to(self.device)

    def forward(self, x):
        x = T.tensor(x, dtype=T.float)
        x = x.to('cuda:0')
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = T.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    
    def save_checkpoint(self):
        print("saving chechpoint")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("loading checkpoint")
        self.load_state_dict(T.load(self.checkpoint_file))