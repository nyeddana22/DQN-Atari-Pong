import sys

#command to test the model
pthname_lst = [x for x in sys.argv if x.endswith(".pth")]
if(len(sys.argv) < 2 or len(pthname_lst) != 1):
    print("python3 test_dqn_pong.py model.pth [-g]")
    exit()
pthname = pthname_lst[0]
use_gui = "-g" in sys.argv

from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer
from pyvirtualdisplay import Display
from gym import wrappers

#enable game play display
virtual_display = Display(visible=0,size=(1400,900))
virtual_display.start()
#load the Pong environment
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
env = wrappers.Monitor(env,'./Pong_Replay', force=True)

num_frames = 1000000
batch_size = 32
gamma = 0.99

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
model.eval()
if USE_CUDA:
    model = model.cuda()
    print("Using cuda")

#load the model
model.load_state_dict(torch.load(pthname,map_location='cpu'))

#can modify seeds to change initial states
env.seed(1)
state = env.reset()
done = False

games_won = 0

while not done:
    if use_gui:
        env.render()
    
    #obtain action from act function
    action = model.act(state, 0)
    
    #obtain corresponding reward and state
    state, reward, done, _ = env.step(action)
    
    #keep track of wins based on reward function
    if reward != 0:
        print(reward)
    if reward == 1:
        games_won += 1

print("Games Won: {}".format(games_won))
try:
    sys.exit(0)
except:
    pass
