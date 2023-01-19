import gym_donkeycar
import gym
import pygame


pygame.init()
# sterowanie = axis 2;  hamulec = axis 4;  gaz = axis 5

env = gym.make("donkey-generated-roads-v0")

obs = env.reset()
def control(value):
    for event in pygame.event.get(): # User did something

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    str = round(joystick.get_axis(2), 1)

    env.step([value,1.0])