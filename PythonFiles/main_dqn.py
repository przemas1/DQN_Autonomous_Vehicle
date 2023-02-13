import numpy as np
from dqn_agent import DQNAgent
from utils import make_env
from csv import writer
import gym_donkeycar
import pygame
import serial
import time


CONTROLER = False
if __name__ == '__main__':
    env = make_env("donkey-generated-roads-v0")
    # możliwe mapy:
    # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|
    #        "donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")

    best_score = 0
    n_games = 1000000





    #                   przykładowe do sprawdzenia
    batch_size = 32     # 64, 32
    replace = 1         # 1, 100
    eps_dec = 1e-5      # 1e-5, 1e-3
    mem_size = 10000     # 100, 1000, 10000
    gamma = 1.2        # 1.2, 0.8
    lr = 0.0001         # 0.01, 0.001 v , 0.0001 v , 0.00001 v






    if CONTROLER:
        pygame.init()
        ser = serial.Serial(port='COM3', baudrate=115200, bytesize=8, timeout=3, stopbits=serial.STOPBITS_ONE)

    agent = DQNAgent(gamma=gamma, epsilon=0.95, lr=lr, input_dims=(env.observation_space.shape), n_actions=15,
                     mem_size=mem_size, eps_min=0.02, batch_size=batch_size, replace=replace, eps_dec=eps_dec)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    # tytuły do csv
    title_row = ["no. episodes", 'score', 'avg_score', 'best_score', 'epsilon', 'n_steps', 'batch', batch_size,
                 'replace', replace, 'eps_dec', eps_dec, 'mem size', mem_size, 'gamma', gamma, 'lr', lr]

    with open("DDQN_History.csv", "a", newline="") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(title_row)
        f_object.close()

    # rozpoczęcie epizodu
    for i in range(n_games):
        episode_start = time.time()
        done = False
        score = 0   
        obs = env.reset()
        # rozpoczęcie kroku
        while not done:

            # epsilon
            action = agent.choose_action(obs)

            if CONTROLER:
                # serial
                joy_action = int((action + 1) * 128)
                if joy_action > 255: joy_action = 255
                if joy_action < 0: joy_action = 0
                # print(joy_action)
                ser.write(bytearray([joy_action, 0]))

                # kontroler
                for event in pygame.event.get():  # User did something
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                str = round(joystick.get_axis(2), 1)
                if str > 255: str = 255
                if str < 0: str = 0

                # krok
                obs_, gym_reward, done, info = env.step([str, 0.7])
            else:
                obs_, gym_reward, done, info = env.step([action, 0.7])

            # alternatywna nagroda
            cte = info['cte']
            episode_length = time.time()-episode_start
            reward = (1 - (abs(cte) / 8)) * episode_length

            # print(reward, 'nagroda')
            # print(time.time()-episode_start, 'mnożnik')
            # nagroda gym
            # reward = gym_reward
            # print("reward", reward)

            score += reward
            agent.store_transition(obs, action, reward, obs_, int(done))
            agent.learn()

            obs = obs_
            n_steps += 1

            print("----------------------------------------------------------------------")

        #zapisywanie statystyk
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        best_score = max(scores)

        row = [i, score, avg_score, best_score, agent.epsilon, n_steps]
        with open("DDQN_History.csv", "a", newline="") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(row)
            f_object.close()

        print('ep', i, 'score', score, 'avgscore', avg_score, 'best',
              best_score, 'epsilon', agent.epsilon, 'steps', n_steps)