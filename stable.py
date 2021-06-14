from collections import deque
import random
import numpy as np
import math

import gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from stable_baselines3 import A2C, SAC, PPO, TD3


env = gym.make('stocks-v0',
            df = STOCKS_GOOGL,
            window_size = 10,
            frame_bound = (10, 1800))

test_env = gym.make('stocks-v0',
            df = STOCKS_GOOGL,
            window_size = 10,
            frame_bound = (1800, 2335))


model = A2C('MlpPolicy', env,verbose=1).learn(2000)

# Calculating mean and buying or selling according to that
t_obs = test_env.reset()
print("> max_possible_profit(test):", test_env.max_possible_profit())
while True:
    try:
        mean = np.mean(t_obs[:, 0])
        print(t_obs)
        print(mean)
        
        if t_obs[-1][0] > mean:
            a = 0
        else:
            a = 1
        
        # a = model.predict(t_obs)
        t_obs, t_reward, t_done, info = test_env.step(a)

    except Exception as e:
        print(e)
        t_obs, t_reward, t_done, _ = test_env.step(test_env.observation_space.sample)

    if t_done:
        print(info)
        plt.cla()
        test_env.render_all()
        plt.show()
        break

