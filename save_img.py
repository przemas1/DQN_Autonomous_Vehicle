import gym
import gym_donkeycar
import cv2
import time

env = gym.make("donkey-generated-track-v0")

obs = env.reset()
img = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
cv2.imshow("test2", img)
cv2.imwrite("testestest.jpg", img)
cv2.waitKey(0)