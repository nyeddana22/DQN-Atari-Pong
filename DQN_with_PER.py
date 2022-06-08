## dqn.py (Modified as Random Agent)
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        # epsilon = 1.0  # just for testing random agent
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action
            Q_val = self.forward(state)
            _, action = Q_val.max(1)
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
#     state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state, action, reward, next_state, done, sample_indices, weights = replay_buffer.sample(batch_size, beta)
    state = Variable(torch.FloatTensor(np.float32(state)).squeeze(1))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    weights = Variable(torch.FloatTensor(weights))
    # implement the loss function here

    curr_qvals = model.forward(state)
    next_qvals = target_model.forward(next_state)
    curr_qval = curr_qvals.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_qval = next_qvals.max(1)[0]
    expected_qval = reward + gamma * next_qval * (1 - done)
#     loss = torch.nn.functional.mse_loss(expected_qval,curr_qval)

    #MSE Loss calculation for PER 
    loss  = (curr_qval - expected_qval.detach()).pow(2) * weights
    prios = loss + 1e-5
    replay_buffer.update_priorities(sample_indices, prios.data.cpu().numpy())
    return loss

# Prioritized Experience Replay buffer class
class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha = 0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        #PER
        self.alpha = alpha
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # TODO: Randomly sampling data with specific batch size from the buffer

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
    
        probs = prios ** self.alpha
        probs /= probs.sum()
        total = len(self.buffer)
        sample_indices = np.random.choice(len(self.buffer),batch_size, p=probs)

        weights = (total * probs[sample_indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for i in sample_indices:
          state.append(self.buffer[i][0])
          action.append(self.buffer[i][1])
          reward.append(self.buffer[i][2])
          next_state.append(self.buffer[i][3])
          done.append(self.buffer[i][4])

        return state, action, reward, next_state, done, sample_indices, weights
        # return None # just for testing random agent
        #return state, action, reward, next_state, done

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)
