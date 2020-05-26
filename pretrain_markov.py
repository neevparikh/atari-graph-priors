import argparse
import json
import os
import sys
import random
import pickle
from collections import namedtuple

import gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from env import get_train_test_envs, get_run_tag
from model import build_phi_network, MarkovHead
from gym_wrappers import FrameStack, MaxAndSkipEnv, AtariPreprocess

Replay = namedtuple("Replay", ["state", "action", "next_state", "reward", "done"])

parser = argparse.ArgumentParser(description='Pretrain Markov Abstraction')
# yapf: disable
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, required=True, help='ATARI game')
parser.add_argument('--n-training-steps', type=int, default=int(15e3), metavar='STEPS',
                      help='Number of gradient steps')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T',
                    help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE',
                    help='Network hidden size')
parser.add_argument('--memory-capacity', type=int, default=int(20e3), metavar='CAPACITY',
                    help='Experience memory capacity')
parser.add_argument('--learning-rate', type=float, default=0.003, metavar='η',
                    help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
                    help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=2048, metavar='SIZE', help='Batch size')
parser.add_argument('--enable-cudnn', action='store_true',
                    help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--phi-net-path', type=str,
                    help='Path to save/load the phi network from')
parser.add_argument('--uuid', default='env', type=str, required=False,
                    help="""UUID for the experient run.
                    `env` uses environment name,
                    `random` uses a random UUID
                        (i.e. 2b25d648-ce34-4523-b409-35247d481e32),
                    anything else is custom, example:
                    PongNoFrameskip_ari_5 (uuid = env)
                    PongNoFrameskip_64f9c4c9-bee5-44a6-9a61-d70267d9d623_ari_5 (uuid = random)
                    PongNoFrameskip_lr_1e-5_ari_5 (uuid = lr_1e-5 (custom))
                    """)
# yapf: enable

# Setup
args = parser.parse_args()
args.overfit_one_batch = False
args.architecture = 'data-efficient'

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')
random.seed(args.seed)


class FeatureNet(nn.Module):

    def __init__(self, args, action_space):
        super(FeatureNet, self).__init__()
        self.phi, self.feature_size = build_phi_network(args)
        self.markov_head = MarkovHead(args, self.feature_size, action_space)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.learning_rate,
                                          eps=args.adam_eps)

    def forward(self, x):
        return self.phi(x)

    def loss(self, batch):
        states, actions, next_states, returns, dones = batch
        markov_loss = self.markov_head.compute_markov_loss(
            z0=self.phi(states.to(torch.float32)),
            z1=self.phi(next_states.to(torch.float32)),
            a=actions,
        )
        loss = markov_loss
        return loss

    def save_phi(self, path, name):
        full_path = os.path.join(path, name)
        torch.save((self.phi, self.feature_size), full_path)
        return full_path

    def train_one_batch(self, batch):
        loss = self.loss(batch)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


def generate_experiences(args, env):
    state = torch.zeros((args.memory_capacity,) + env.observation_space.shape, dtype=torch.uint8)
    next_state = torch.zeros((args.memory_capacity,) + env.observation_space.shape, dtype=torch.uint8)
    action = torch.zeros((args.memory_capacity,), dtype=torch.long)
    reward = torch.zeros((args.memory_capacity,))
    done = torch.zeros((args.memory_capacity,))
    mem = Replay(state, action, next_state, reward, done)

    state, done = env.reset(), False
    i = 0
    pbar = tqdm(total=args.memory_capacity)
    while i < args.memory_capacity:
        if done:
            state, done = env.reset(), False
        action = np.random.randint(0, env.action_space.n)
        next_state, reward, done, _ = env.step(action)
        if not done:
            mem.state[i] = state
            mem.next_state[i] = next_state
            mem.action[i] = action
            mem.done[i] = done
            mem.reward[i] = reward
            i += 1
            pbar.update(1)
        state = next_state
    return mem


def sample(mem, args):
    batch_idx = random.sample(range(args.memory_capacity), args.batch_size)
    return mem.state[batch_idx], mem.action[batch_idx], mem.next_state[batch_idx], \
            mem.reward[batch_idx], mem.done[batch_idx]


def train():
    run_tag = get_run_tag(args)
    results_dir = os.path.join('results/pretraining', run_tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/params.json", 'w') as fp:
        param_dict = vars(args)
        param_dict['device'] = str(args.device)
        json.dump(param_dict, fp)

    print('making env')
    env = gym.make(args.env)
    env = FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4), args.history_length, cast=torch.uint8, scale=False)
    network = FeatureNet(args, env.action_space.n)

    experiences_filepath = f"{results_dir}/experiences.mem"
    # if os.path.exists(experiences_filepath):
    #     print('loading experiences')
    #     # with open(experiences_filepath, 'rb') as fp:
    #     #     mem = pickle.load(fp)
    # else:
    print('generating experiences')
    mem = generate_experiences(args, env)
        # with open(experiences_filepath, 'wb') as fp:
        #     pickle.dump(mem, fp)

    print('training')
    with open(f"{results_dir}/loss.csv", 'w') as fp:
        fp.write('step,loss\n')  # write headers

        batch = sample(mem, args)
        for step in tqdm(range(args.n_training_steps)):
            if not args.overfit_one_batch and step > 0:
                batch = sample(mem, args)
            loss = network.train_one_batch(batch)
            fp.write(f"{step},{loss}\n")

    phi_net_path = network.save_phi(results_dir, 'phi_model.pth')
    print('Saved phi network to {}'.format(phi_net_path))


if __name__ == '__main__':
    train()
