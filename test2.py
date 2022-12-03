import gym
import gym_donkeycar

env = gym.make("donkey-generated-track-v0")

obs = env.reset()

for i in range(10):
    action = env.action_space.sample()
    print(action)