import numpy as np

#replay buffer samplowanie state, aciton reward, state_transition z wielu epizodow
#niesekwencyjne samplowanie opzwala na samplowanie szerokiego zakresu parametrow,
#zeby agent nie zacial sie w jednym miejscu 

class ReplayBuffer():
    def __init__ (self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uitn8)

        

    def store_transision(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size   # zapisz w pamieci na pierwszej wolnej pozycji
        
        self.state_memory[index]    = state     # przechodze do arraya i wpisuje wartosci state 
        self.action_memory[index]   = action
        self.reward_memory[index]   = reward
        self.new_state_memory[index]= state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1                      # po zapisaniu przejdz do nast. miejsca w pamieci

    def sample_buffer(self, batch_size):        # sample buffer uniformally
        max_mem = min(self.mem_cntr, self.mem_size)
                                                # pozycja ostatniego zapisu w pamieci,
                                                # jesli zapelnilem cala - sampluj do konca pamieci
                                                # jesli nie - sampluj do mem_cntr
        batch = np.random.choice(max_mem, batch_size, replace=False)
                                                # uniformally sample the memory
                                                # replace=False - jezeli index zostal zsamplowany zostaje usuniety
        
        states  = self.state_memory[batch]    
        actions = self.action_memory[batch]   
        rewards = self.reward_memory[batch]   
        states_ = self.new_state_memory[batch]
        dones   = self.terminal_memory[batch] 

        return states, actions, rewards, states_, dones