import gym
import numpy as np
import time
import pickle

env = gym.make('CartPole-v1')
episodes = 50
highscore = 0
bestParams = None

for episode in range(episodes):
    observation = env.reset()
    parameters = np.random.rand(4) * 2 - 1
    points = 0
    for t in range(100):
        env.render()
        # action = env.action_space.sample()
        # action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        points += reward
        print(episode, t, parameters, observation, action, points, highscore)
        time.sleep(0.01)
        if (done):
            break
    if points > highscore:
        highscore = points
        bestParams = parameters

with open('params.pickle', 'wb') as handle:
    pickle.dump(bestParams, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')
exit(0)
