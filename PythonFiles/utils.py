import collections
from preprocess import preprocessing
import numpy as np
import gym


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)
    
    def observation(self, obs):
        new_frame = preprocessing(obs)
        new_obs = np.array(new_frame, dtype=np.int32).reshape(self.shape)  
        new_obs = new_obs / 255.
        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for i in range(self.stack.maxlen):
            self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.high.shape)

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.high.shape)


def make_env(env_name, shape=(84, 84, 1), repeat=4):
    env = gym.make(env_name)
    env = StackFrames(env, repeat)
    env = PreprocessFrame(shape, env)
    return env
