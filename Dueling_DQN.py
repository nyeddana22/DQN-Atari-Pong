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
    #Dueling DQN Architecture Implementation
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.conv_nn = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU() 
        )
        cnn_output_shape = self.conv_nn(torch.zeros(1, *self.input_shape))
        cnn_output_shape = int(np.prod(cnn_output_shape.size()))
        
        self.linear_actions = nn.Sequential(
			nn.Linear(cnn_output_shape, 512),
			nn.ReLU(),
			nn.Linear(512, self.num_actions)
        )
        self.linear_value = nn.Sequential(
			nn.Linear(cnn_output_shape, 512),
			nn.ReLU(),
			nn.Linear(512, 1)
        )

    def forward(self, state):
        batch_size = state.size()[0]
        cnn_output = self.conv_nn(state).view(batch_size, -1)
        value = self.linear_value(cnn_output)
        actions = self.linear_actions(cnn_output)
        return value + actions  - torch.mean(actions, dim=1, keepdim=True)

    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    #Define function to choose actions given a state
    def act(self, state, epsilon):
        if random.random() > epsilon:
	    #If state is known, choose action corresponding to the max Q value from the Q table
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            Q_val = self.forward(state)
            _, action = Q_val.max(1)
        else:
	    #If state is not known, choose a random action from the env action space to discover new states
            action = random.randrange(self.env.action_space.n)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

#Define TD loss function
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
	
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = Variable(torch.FloatTensor(np.float32(state)).squeeze(1))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    #Acquire current and next state Q values from the Q table
    curr_qvals = model.forward(state)
    next_qvals = target_model.forward(next_state)
    curr_qval = curr_qvals.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_qval = next_qvals.max(1)[0]
    expected_qval = reward + gamma * next_qval * (1 - done)
    #MSE loss function provides best value; can also try huber loss
    loss = torch.nn.functional.mse_loss(expected_qval,curr_qval)

    return loss

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample data given a specific batch size from the buffer
        rand_data_indices = np.random.choice(len(self.buffer),batch_size)
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for i in rand_data_indices:
          state.append(self.buffer[i][0])
          action.append(self.buffer[i][1])
          reward.append(self.buffer[i][2])
          next_state.append(self.buffer[i][3])
          done.append(self.buffer[i][4])

        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)
