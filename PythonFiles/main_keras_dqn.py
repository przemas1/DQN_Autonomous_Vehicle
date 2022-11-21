import gym
from gym import wrappers
import numpy as np
from dqn import DDQNAgent
#from utils import plotLearningcd

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    ddqn_agent = DDQNAgent(alpha=0.000_5, gamma=0.99, n_actions=4, epsilon=1.0, batch_size=64, input_dims=8)

    n_games = 500

    ddqn_scores = []
    eps_history = []

    # env = wrappers.Monitor(env, 'tmp/lunar-lander', video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, done)
            ddqn_agent.learn()
        
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode', i , 'score %.2f'%score, 'average score%.2f'%avg_score)

        if i%10 ==0 and i>0:
            ddqn_agent.save_model()

        filename = 'lunarlander-ddqn.png'
        x = [i+1 for i in range(n_games)]
        #plotLearning(x, ddqn_scores, eps_history, filename)