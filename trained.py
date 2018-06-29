import gym
import numpy as np
import time
import pickle


def loadParams():
    with open('params.pickle', 'rb') as handle:
        return pickle.load(handle)


env = gym.make('CartPole-v1')
bestParams = loadParams()

observation = env.reset()
while True:
    env.render()
    action = 0 if np.matmul(bestParams, observation) < 0 else 1
    observation, reward, done, _ = env.step(action)
    print(observation, action, reward)
    time.sleep(0.05)
    if (done):
        print('Done!')
        observation = env.reset()
        # break
