# input:    80 by 60 1 channel image
# stack 4 frames
#
#   online and target networks, replay memory
#   choose action copy target net, decrement epsilon
#   save and load models

import numpy as np
import torch as T
from deep_q_learning import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                replace=1000, algo=None, env_name=None, checkpoint_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)] # do epsilon greedy action selection
        self.learn_step_counter = 0                            # kiedy bede wpisywac wartosci z policy network do eval network
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # tworzenie sieci do ewaluacji ruchow
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims, name = self.env_name+'_'+self.algo+'_q_eval',checkpoint_dir=self.checkpoint_dir)
        
        # tworzenie sieci do taktyki nie bede w niej robic gradient descent i propagacji wstecznej
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims, name = self.env_name+'_'+self.algo+'_q_next',checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transision(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states  = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones   = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self - self.eps_dec if self.epsilon > self.epsilon_min else self.eps_min
    
    def save_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    # funkcja uczenia powinna sie zaczac gdy bede mial jakies sample w replay memory wiec moge poczekac z uczeniem az cale replay memory sie zapelni
    # lub zrobic warunek: if mem ctr < batch size: nie aktywuj funcji uczenia
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()      # wyzeruj gradienty w optymizerze
        self.replace_target_network()          # zmieniam siec teraz zeby nie uczyc na starych parametrach
                                               # sample memory
        states, actions, rewards, states_, dones = self.sample_memory()
        
                                               # oblicz q_pred i q_target
        indecies = np.arange(self.batch_size)  # zmienna do wyboru wartosci z tabeli poniewaz q_pred bedzie 2 wymiarowa
        q_pred = self.q_eval.forward(states)[indecies, actions]   # to da wartosci dla akcji dla batcha stanow
        q_next = self.q_next.forward(states_).max(dim=1)[0]       # sprawdzam wartosci q z tablei z targetami dla batcha ze stanami ([0]bo funkcja zwraca tuple)
        # robie to zeby znalezc maksymalne akcje dla stanow w sieci next i nakierowac estymacje agenta w ich strone

        
        #calculation of target value  = rewards + gamma * q_next -> if next state is not terminal else, target value = rewards
        q_next[dones] = 0.0   #using done flag as mask -> if done = true set q_next[index] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        