import random
import numpy as np
import torch
import torch.nn.functional as F
from model import DeepQNetwork
from replay_memory import ReplayMemory

class DQNAgent:
    #funkcje do trenoqwania agenta
    def __init__(self, device, state_size, action_size,
                    discount = 0.99,
                    eps_max = 1.0,
                    eps_min = 0.01,
                    eps_dec = 0.995,
                    mem_cap = 5000,
                    lr = 1e-3,
                    train_mode = True):

        self.device = device

        # for epsilon-greedy exploration strategy
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_dec

        # for defining how far-sighted or myopic the agent should be
        self.discount = discount

        # size of the state vectors and number of possible actions
        self.state_size = state_size
        self.action_size = action_size

        # instances of the network for current policy and its target
        self.policy_net = DeepQNetwork(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DeepQNetwork(self.state_size, self.action_size, lr).to(self.device)
        self.target_net.eval() # since no learning is performed on the target net
        if not train_mode:
            self.policy_net.eval()

        # instance of the replay buffer
        self.memory = ReplayMemory(capacity=mem_cap)

    def update_target_net(self):
        # copy weights of current policy into target net
        # parameters: none
        # returns: none
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        # reducing epsilon val
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state):
        #wybiera akcje na podstawie epsilona
        #parameters: 
        #   state: vector or tensor -> current state
        #returns: none
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
        
        #wybierz akcje z najwieksza wartoscia q
        with torch.no_grad():
            action = self.policy_net.forward(state)
        
        return torch.argmax(action).item()

    def learn(self, batch_size):
        # funkcja updatuje siec neuronową
        #parameters:
        #   batch_size: int
        #       liczba wydarzen to losowego samplowania z pamieci, dla agenta do uczenia
        #returns: none

        # select n samples picked uniformly at random from the experience replay memory, such that n=batchsize
        if len(self.memory) < batch_size:
            return
            
        states, actions, next_states, rewards, dones = self.memory.sample(batch_size, self.device)
        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        # get q values of the actions that were taken, i.e calculate qpred; 
        # actions vector has to be explicitly reshaped to nx1-vector
        q_pred = self.policy_net.forward(states).gather(1, actions) 
        
        #calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        target_action = torch.argmax(self.policy_net.forward(next_states), dim=1)#.numpy()

        q_target = self.target_net.forward(next_states).gather(1, target_action.view(-1, 1)) # because max returns data structure with values and indices

        q_target[dones] = 0.0 # setting Q(s',a') to 0 when the current state is a terminal state

        y_j = rewards + (self.discount * q_target)
        
        # calculate the loss as the mean-squared error of yj and qpred
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
        

    def save_model(self, filename):
        self.policy_net.save_model(filename)

    def load_model(self, filename):
        self.policy_net.load_model(filename)