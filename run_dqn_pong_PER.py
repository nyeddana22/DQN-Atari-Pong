from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, PrioritizedReplayBuffer

#Import the Pong environment
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

#define number of epochs, sampling batch size, gamma, buffer size
num_frames = 2000000
batch_size = 32
gamma = 0.99
record_idx = 10000

replay_initial = 10000
replay_buffer = PrioritizedReplayBuffer(100000)

#initialize model from the Qlearner and load pretrained model
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load("model_pretrained.pth", map_location='cpu'))

#create target model for predicted q values
target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model.copy_from(model)

#defne optimizer to solve the optimization problem with learning rate
optimizer = optim.Adam(model.parameters(), lr=0.00001)

if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    print("Using cuda")

#define epsilon and corresponding decay function
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 50000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

#define beta and corresponding decay function for PER
beta_frames = 100000
beta_start = 0.4
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)


losses = []
all_rewards = []
episode_reward = 0

frames_plt = []
mean_losses_plt = []
mean_rewards_plt = []
best_reward = None
state = env.reset()

#run iteration for every frame/epoch
for frame_idx in range(1, num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    #get the action given state and epsilon
    action = model.act(state, epsilon)
    #based on the action, return the next state and reward function value
    next_state, reward, done, _ = env.step(action)
    #store these values in the replay buffer
    replay_buffer.push(state, action, reward, next_state, done)
    
    #move to the next state and increment the episode reward sinc Q learning is episodic learning
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    if len(replay_buffer) > replay_initial:

        beta = beta_by_frame(frame_idx)
        #calculate the loss
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer, beta)
        loss = loss.mean()
        #optimize and backpropagate through the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        
        mloss = np.mean(losses, 0)[1]
        mreward = np.mean(all_rewards[-100:], 0)[1]
        print('#Frame: %d, Loss: %f' % (frame_idx, mloss))
        print('Last-10 average reward: %f' % mreward)
        print('Epsilon Value: %f' % epsilon)
        print('Beta Value: %f' % beta)

        frames_plt.append(frame_idx)
        mean_losses_plt.append(mloss)
        mean_rewards_plt.append(mreward)
        
        #save the model only when the best reward is obtained
        if best_reward is None or best_reward < mreward:
            torch.save(model.state_dict(), "model_new.pth")
            best_reward = mreward
            if best_reward is not None:
                print("Best reward in these frames: %f" % best_reward)
                
    #copy to target_model every 10000 frames to ensure updated q values are stored
    if frame_idx % 10000 == 0:
        target_model.copy_from(model)

#plot the loss and reward over all epochs
fig = plt.figure()
loss_plot = fig.add_subplot(121)
reward_plot = fig.add_subplot(122)
loss_plot.plot(frames_plt, mean_losses_plt,label="Losses")
loss_plot.set_title("Losses")
reward_plot.plot(frames_plt, mean_rewards_plt,label="Rewards")
reward_plot.set_title("Rewards")
fig.tight_layout()
plt.savefig('plots_new.png')
