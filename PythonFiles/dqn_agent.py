from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from collections import deque
from keras.applications.xception import Xception
from keras.optimizers import Adam
import random
import numpy as np
import tensorflow as tf
import time

REPLAY_MEMORY_SIZE = 5_000

MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAININIG_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

DISCOUNT = 0.99

EPISODES = 100
EPSILON = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

IMG_HEIGHT = 120
IMG_WIDTH = 160

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        
       # self.tensorboard = ModifiedTensorBoard()                 z tym sa problemy wersja tf1
        self.graph = tf.compat.v1.get_default_graph()
    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(4, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model
    
    def update_replay_memory(self, transition): pass
        #transition = (current_state, action, reward, new_state, done)
        #self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.array([transition[0] for tranisition in minibatch]) / 255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for tranisition in minibatch]) / 255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = [] 
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

######################tensorboard step 

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=None)


    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1 *state.shape)/255)[0]

    
    def train_in_loop(self):
        X = np.random.uniform(size=(1, IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 4)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1, verbose=0)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
