import collections 
import cv2

import matplotlib.pyplot as plt
import numpy as np
import gym

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label= "1")
    ax2 = fig.add_subplot(111, label= "2", frame_on=False)

    ax.plot(x, epsilons, color='C0')
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])
    
    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

'''
ZMIANY WPROWADZONE W TUTORIALU ATARI:
- zmiana 3 kanałów na 1 szarości
- zmiana rozmiaru do 84 na 84
- łączy 2 screeny (problem tylko na atari)
- powtarzanie akcji 4 razy (inne niż repeat w make_env())
- przerzucenie liczby kanałów na pierwsze miejsce (pytorch tak oczekuje)
- stackowanie 4 ostatnich screenów
- skalowanie inputów


Z repo gym o wrappers:
    instancja env jako zmienna którą wrzucam jako input do konstruktora MyWrapper
        env = gym.make('Game')
        env = MyWrapper(env)

    żeby nadpisać __init__ wrappera musze wywołać:
        super(class_name, self).__init__(env)

    wtedy moge zmieniac step, restart, itd.

Używane klasy:
- gym.Wrapper -> użyty do funkcji step
- gym.ObservationWrapper -> zajmuje się obserwacjami ze środowiska


PSEUDOKOD:

    Class repeatactionandmaxframe:
        derived from: gymWrapper
        input: emvoironment, repeat
        init frme buffer as an array of zeros in shape 2 x the obs space

        funcition step:
            input: action
            set total reward to 0
            set done to false
            for i in range repeat
                call the env.step
                    recieve obs, reward,, done, info
                increment total reward
                insert obs in frame buffer
                if done
                    break
                    
                end for
                find the max frame
                return max frame, total reward, done, info

        function reset:
            input: none
            reset frame buffer
            store initial observation in buffer
            
            return: initial observation

    Class preprocessframe:
        derives from: gymObservationWrapper
        input: env, new shape
        set shape by swapping channels axis from posiiton 2 to 0
        set observation space to new shape using gym.spaces.Box(0 to 1.0)

        function observation:
            input: raw observation
            convert the observation to gray scale
            resize observation to new shape
            convert obs to numpy array
            move obs channel axis from position 2 to 0
            observation / 255

            return observation

    Class stackframes
        derive from gymObservationWrapper
        input: env stack size
        init the new obs space (gym.spaces.Box) low and high bouds as repeat of n_steps
        init empty frame stack

        reset function:
            clear the stack
            reset env
            for i in range(Stack size)
                append initial obs to stack
            convert stack to np array
            reshape stack array to obs space low shape
            return stack
    
        obs function:
            input obs
            append obs to end of stack
            convert stsck to np array
            reshape stack to obss space low shape
            return stack o frames

    function make_env
    input: env name, new shape, stack size
    init env with base gym.make funciton
    env = repeat action adn max frame
    env = preprocess frame
    env = stack frames

    return env
        
implementation tips:
- use nparray and deques    wszystko na koncu musi byc zaminenione ne nparray

'''


def make_env(env_name):
    env = gym.make(env_name)
    return env