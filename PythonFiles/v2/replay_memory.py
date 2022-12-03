import random
import numpy as np
import torch

class ReplayMemory:
    #replay memory is used for storing experiences for off policy learning
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done):
        #dodawanie doswiadczen do pamieci.
        #
        #   state: numpy.ndarray
        #       current state vector
        #   action: int
        #       action performed on current state
        #   next_state: numpy.ndarray
        #       state vector observed as result of a action
        #   reward: float
        #   
        #   done: bool
        #
        #   returns: none

        if  len(self.buffer_state) < self.capacity:         #jeśli replay mem nie został zapełniony
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        
        else:                                               #jeśli replay mem został zapełniony
            self.buffer_state[self.idx]     = state
            self.buffer_action[self.idx]    = action
            self.buffer_next_state[self.idx]= next_state
            self.buffer_reward[self.idx]    = reward
            self.buffer_done[self.idx]      = done
        
        self.idx - (self.idx+1) % self.capacity              #dzieki temu pamiec zapelnia sie w kółko

    def sample(self, batch_size, device):
        #funkcja wybiera n sampli z pamieci w sposob losowy tak ze n = batch_size
        # 
        # batch_size: int
        #   liczba elementow do losowego samplowania w jednym batchu
        # device: str
        #   nazwa urządzenia cuda albo cpu
        # Reurns: tensory reprezentujace batcha z przejściami pomiędzy stanami, z pamieci
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states      = torch.from_numpy(np.array(self.buffer_state)[indices_to_sample]).float().to(device)
        actions     = torch.from_numpy(np.array(self.buffer_action)[indices_to_sample]).to(device)
        next_states = torch.from_numpy(np.array(self.buffer_next_state)[indices_to_sample]).float().to(device)
        rewards     = torch.from_numpy(np.array(self.buffer_reward)[indices_to_sample]).float().to(device)
        dones       = torch.from_numpy(np.array(self.buffer_done)[indices_to_sample]).to(device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        #funkcja mowi o liczbie elementow w replay memory
        # parameters: none
        # returns: int
        #       liczba elementow w replay bufferze
        return len(self.buffer.state)