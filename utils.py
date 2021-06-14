import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import matplotlib.pyplot as plt


class Buffer:
    def __init__(self, size):
        self.max_size = size
        self.buffer = deque(maxlen=size)
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.size = min(self.size+1,self.max_size)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        # np.random.seed(0)
        batch = np.random.randint(0, len(self.buffer), size=batch_size)
        for experience in batch:
            state, action, reward, next_state, done = self.buffer[experience]
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return self.size


class NN(nn.Module):

    def __init__(self, input_shape, hidden_shape, hidden_layers, output_shape, output_activation=None):
        super(NN, self).__init__()

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.hidden_layers = hidden_layers
        self.output_shape = output_shape
        self.output_activation = output_activation

        self.input_layer = nn.Linear(input_shape, hidden_shape)

        self.hidden = []

        self.input_layer.weight.data.normal_(0,1)

        for _ in range(hidden_layers):
            self.hidden.append(nn.Linear(hidden_shape, hidden_shape))
            self.hidden[-1].weight.data.normal_(0,1)

        self.output_layer = nn.Linear(hidden_shape, output_shape)

        self.output_layer.weight.data.normal_(0,1)

    def forward(self, x):

        y = self.input_layer(x)
        y = torch.relu(y)

        for h_layer in self.hidden:
            y = h_layer(y)
            y = torch.tanh(y)

        y = self.output_layer(y)

        if self.output_activation != None:
            y = self.output_activation(y)

        return y

    def predict(self, x):
        obs = torch.tensor(x, dtype=torch.float32)

        q_values = self(obs.unsqueeze(0))

        max_q_idx = torch.argmax(q_values, dim=1)[0]
        action = max_q_idx.detach().item()

        return action