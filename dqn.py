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

from utils import *

def select_action(policy_net, state, step, env):
    sample = random.random()
    epsilon = np.interp(step, [0, EPS_DECAY], [EPS_START, EPS_END])
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)

    if sample > epsilon:
        with torch.no_grad():
            return policy_net(torch.tensor(state).unsqueeze(0).float()).max(1)[1].view(1, 1).item()
    else:
        return env.action_space.sample()


def evaluate_model(model, env):
    t_obs = env.reset()
    while True:
        
        try:
            t_obs = np.reshape(t_obs, OBESERVATION_SPACE)
            a = model.predict(t_obs)
            t_obs, t_reward, t_done, t_info = env.step(a)

            # env.render()
        except Exception as e:
            print(e)
            t_obs, t_reward, t_done, _ = env.step(env.observation_space.sample)

        if t_done:
            print(t_info)
            plt.cla()
            env.render_all()
            plt.show()
            break
    env.reset()


if __name__ == '__main__':

    env = gym.make('stocks-v0',
               df = STOCKS_GOOGL,
               window_size = 10,
               frame_bound = (10, 1800))

    test_env = gym.make('stocks-v0',
               df = STOCKS_GOOGL,
               window_size = 10,
               frame_bound = (1800, 2335))

    print("env information:")
    print("> df.shape:", env.df.shape)
    print("> prices.shape:", env.prices.shape)
    print("> signal_features.shape:", env.signal_features.shape)
    print("> max_possible_profit:", env.max_possible_profit())


    BUFFER_SIZE = 10000
    MIN_BUFFER_SIZE = 1000

    OBESERVATION_SPACE = env.observation_space.shape[0] * env.observation_space.shape[1]
    ACTION_SPACE = 2


    print(OBESERVATION_SPACE)
    
    print(ACTION_SPACE)

    BATCH_SIZE = 64
    TRAIN_STEPS = 5000

    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TARGET_UPDATE = 100

    buffer = Buffer(BUFFER_SIZE)

    # Main network
    policy_net = NN(OBESERVATION_SPACE, 500, 3, ACTION_SPACE)


    # Target network
    target_policy_net = NN(OBESERVATION_SPACE, 500, 3, ACTION_SPACE)

    # Loss function (Huber Loss)
    loss_fn = nn.SmoothL1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-5)  

    # Initial buffer fill
    obs = env.reset()
    for i in range(MIN_BUFFER_SIZE):
        obs = np.reshape(obs, OBESERVATION_SPACE)
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)
        new_obs = np.reshape(new_obs, OBESERVATION_SPACE)
        buffer.push(obs, action, reward, new_obs, done)
        
        obs = new_obs
            
        if done:
            obs = env.reset()


    # Training
    obs = env.reset()
    for i in range(TRAIN_STEPS):
        if i%5000 == 0:
            evaluate_model(policy_net, env)

            evaluate_model(policy_net, test_env)


        obs = np.reshape(obs, OBESERVATION_SPACE)
        action = select_action(policy_net, obs, i, env)
        new_obs, reward, done, _ = env.step(action)
        new_obs = np.reshape(new_obs, OBESERVATION_SPACE)
        buffer.push(obs, action, reward, new_obs, done)
        
        obs = new_obs
            
        if done:
            # plt.cla()
            # env.render_all()
            # plt.show()
            obs = env.reset()

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(BATCH_SIZE)

        state_batch = torch.tensor(state_batch, dtype = torch.float32)
        action_batch = torch.tensor(action_batch, dtype = torch.int64).unsqueeze(-1)
        reward_batch = torch.tensor(reward_batch, dtype = torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype = torch.float32)
        done_batch = torch.tensor(done_batch, dtype = torch.float32).unsqueeze(-1)

        target_q_values = target_policy_net(next_state_batch)

        max_target_q_values = target_q_values.max(dim=1,keepdim=True)[0]

        targets = reward_batch + GAMMA * (1-done_batch) * max_target_q_values

        q_values = policy_net(state_batch)

        action_q_values = torch.gather(input=q_values, dim=1, index=action_batch)


        loss = loss_fn(action_q_values, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Network
        if i % TARGET_UPDATE == 0:
            target_policy_net.load_state_dict(policy_net.state_dict())

        if i % 100 == 0:
            print("Step: {}  Loss: {}  Avg Reward: {}".format(i, loss.item(), torch.mean(reward_batch).item()))


    evaluate_model(policy_net, env)
    evaluate_model(policy_net, test_env)

