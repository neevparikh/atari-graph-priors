# -*- coding: utf-8 -*-
from __future__ import division
import bz2
from datetime import datetime
import os
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
from utils import initialize_environment
from memory import ReplayMemory
from test import test
from parser import parser

# Setup
args = parser.parse_args()

print('Options')
for k, v in vars(args).items():
    print(k + ': ' + str(v))


results_dir = os.path.join('results', args.id)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


# Environment
env, test_env = initialize_environment(args)
n_actions = env.action_space.n

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError(
            'Could not find memory file at {path}. Aborting...'.format(path=args.memory))

    mem = load_memory(args.memory, args.disable_bzip_memory)

else:
    mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
    if done:
        state, done = env.reset(), False

    next_state, _, done, _ = env.step(np.random.randint(0, n_actions))
    val_mem.append(state, -1, 0.0, done)
    state = next_state
    T += 1

if args.evaluate:
    dqn.eval()  # Set DQN (online network) to evaluation mode
    avg_reward, avg_Q = test(args, test_env, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
    print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
    # Training loop
    dqn.train()
    T, done = 0, True
    for T in trange(1, args.T_max + 1):
        if done:
            state, done = env.reset(), False

        if T % args.replay_frequency == 0:
            dqn.reset_noise()  # Draw a new set of noisy weights

        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, _ = env.step(action)  # Step
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        mem.append(state, action, reward, done)  # Append transition to memory

        # Train and test
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase,
                                      1)  # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                dqn.learn(mem)  # Train with n-step distributional double-Q learning

            if T % args.evaluation_interval == 0:
                dqn.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' +
                    str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                dqn.train()  # Set DQN (online network) back to training mode

                # If memory path provided, save it
                if args.memory is not None:
                    save_memory(mem, args.memory, args.disable_bzip_memory)

            # Update target network
            if T % args.target_update == 0:
                dqn.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn.save(results_dir, 'checkpoint.pth')

        state = next_state

env.close()
