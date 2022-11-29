import numpy as np
from dqn_agent import DQNAgent
from utils import make_env, plot_learning_curve

if __name__ == '__main__':
    env = make_env("donkey-mountain-track-v0")
    best_score = np.inf
    load_checkpoint = False
    n_games = 100
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape), n_actions=env.action_space.n, mem_size=25000,
    eps_min=0.1, batch_size=32, replace=1000, eps_dec=1e-5, checkpoint_dir='models/',algo='DQNAgent', env_name="donkey-mountain-track-v0")

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + "_lr" + str(agent.lr)
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(obs, action, reward, obs_, int(done))
                agent.learn()
            
            obs = obs_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-100:])
        print('ep', i, 'score', score, 'avgscore', avg_score, 'best', best_score, 'epsilon', agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)