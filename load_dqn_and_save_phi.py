# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
from env import get_env
from memory import ReplayMemory
from test import test

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, required=True, help='ATARI game')
parser.add_argument('--T-max',
                    type=int,
                    default=int(400e3),
                    metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length',
                    type=int,
                    default=int(108e3),
                    metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length',
                    type=int,
                    default=4,
                    metavar='T',
                    help='Number of consecutive states processed')
parser.add_argument('--architecture',
                    type=str,
                    default='data-efficient',
                    choices=['canonical', 'data-efficient', 'ari', 'ram', 'pretrained', 'online'],
                    metavar='ARCH',
                    help='Network architecture')
parser.add_argument('--hidden-size',
                    type=int,
                    default=256,
                    metavar='SIZE',
                    help='Network hidden size')
parser.add_argument('--noisy-std',
                    type=float,
                    default=0.1,
                    metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms',
                    type=int,
                    default=51,
                    metavar='C',
                    help='Discretised size of value distribution')
parser.add_argument('--V-min',
                    type=float,
                    default=-10,
                    metavar='V',
                    help='Minimum of value distribution support')
parser.add_argument('--V-max',
                    type=float,
                    default=10,
                    metavar='V',
                    help='Maximum of value distribution support')
parser.add_argument('--model',
                    type=str,
                    metavar='PARAMS',
                    help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity',
                    type=int,
                    default=int(400e3),
                    metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency',
                    type=int,
                    default=1,
                    metavar='k',
                    help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent',
                    type=float,
                    default=0.5,
                    metavar='ω',
                    help='Prioritised experience replay exponent \
            (originally denoted α)')
parser.add_argument('--priority-weight',
                    type=float,
                    default=0.4,
                    metavar='β',
                    help='Initial prioritised experience replay \
            importance sampling weight')
parser.add_argument('--multi-step',
                    type=int,
                    default=20,
                    metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--discount',
                    type=float,
                    default=0.99,
                    metavar='γ',
                    help='Discount factor')
parser.add_argument('--target-update',
                    type=int,
                    default=int(2000),
                    metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip',
                    type=int,
                    default=1,
                    metavar='VALUE',
                    help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate',
                    type=float,
                    default=0.0001,
                    metavar='η',
                    help='Learning rate')
parser.add_argument('--adam-eps',
                    type=float,
                    default=1.5e-4,
                    metavar='ε',
                    help='Adam epsilon')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    metavar='SIZE',
                    help='Batch size')
parser.add_argument('--learn-start',
                    type=int,
                    default=1600,
                    metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval',
                    type=int,
                    default=10000,
                    metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes',
                    type=int,
                    default=10,
                    metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size',
                    type=int,
                    default=500,
                    metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--render',
                    action='store_true',
                    help='Display screen (testing only)')
parser.add_argument('--enable-cudnn',
                    action='store_true',
                    help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval',
                    default=0,
                    type=int,
                    help='How often to checkpoint the model, \
            defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--phi-net-path',
                    type=str,
                    help='Path to save/load the phi network from')
parser.add_argument('--disable-bzip-memory',
                    action='store_true',
                    help='Don\'t zip the memory file. Not recommended \
            (zipping is a bit slower and much, much smaller)')
parser.add_argument('--uuid',
                    help="""UUID for the experient run.
                    `env` uses environment name,
                    `random` uses a random UUID
                        (i.e. 2b25d648-ce34-4523-b409-35247d481e32),
                    anything else is custom, example:
                    PongNoFrameskip_ari_5 (uuid = env)
                    PongNoFrameskip_64f9c4c9-bee5-44a6-9a61-d70267d9d623_ari_5 (uuid = random)
                    PongNoFrameskip_lr_1e-5_ari_5 (uuid = lr_1e-5 (custom))
                    """,
                    default='env',
                    type=str,
                    required=False)

# Setup
args = parser.parse_args()

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)

env, test_env, run_tag = get_env(args)
dqn = Agent(args, env)

model_filename = os.path.basename(args.model)
model_dirname = os.path.dirname(args.model)

phi_net_path = dqn.online_net.save_phi(model_dirname, 'phi_'+model_filename)
print('Saved phi network to {}'.format(phi_net_path))

print(torch.load(phi_net_path))
