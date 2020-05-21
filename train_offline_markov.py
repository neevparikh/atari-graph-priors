import argparse
import os

import gym
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Agent
from env import get_train_test_envs
from model import build_phi_network, MarkovHead
from memory import ReplayMemory, load_memory, save_memory

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, required=True, help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(400e3), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T',
                    help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='data-efficient', metavar='ARCH',
                    choices=['canonical', 'data-efficient', 'ari', 'ram', 'pretrained', 'online'],
                    help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE',
                    help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C',
                    help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V',
                    help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V',
                    help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS',
                    help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(400e3), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=1, metavar='k',
                    help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=20, metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ',
                    help='Discount factor')
parser.add_argument('--markov-loss-coef', type=float, default=1.0, metavar='λ',
                    help='Coefficient for Markov loss when training online')
parser.add_argument('--target-update', type=int, default=int(2000), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE',
                    help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η',
                    help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
                    help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=1600, metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true',
                    help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, type=int,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--phi-net-path', type=str,
                    help='Path to save/load the phi network from')
parser.add_argument('--disable-bzip-memory', action='store_true',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
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

# Setup
args = parser.parse_args()

args.overfit_one_batch = True
args.architecture = 'data-efficient'
args.n_training_steps = 1000
args.evaluation_interval = 10
args.hidden_size = 256
args.learning_rate = 0.003
args.batch_size = 32
args.model = 'results/ccv-2020-05-20-1910/QbertNoFrameskip-v4_learning_rate_0.0001_seed_2/model.pth'
args.memory_capacity = int(20e3)
args.dqn_policy_epsilon = 0.05
# args.memory = 'results/ccv-2020-05-20-1910/QbertNoFrameskip-v4_learning_rate_0.0001_seed_2/replay_memory.mem'
args.memory = 'quick20K.mem'

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')


class FeatureNet(nn.Module):
    def __init__(self, args, action_space):
        super(FeatureNet, self).__init__()
        self.phi, self.feature_size = build_phi_network(args)
        self.markov_head = MarkovHead(args, self.feature_size, action_space)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.learning_rate,
            eps=args.adam_eps
        )

    def forward(self, x):
        return self.phi(x)

    def loss(self, batch):
        idxs, states, actions, returns, next_states, nonterminals, weights = batch
        markov_loss = self.markov_head.compute_markov_loss(
            z0 = self.phi(states),
            z1 = self.phi(next_states),
            a = actions,
        )
        loss = markov_loss
        return loss

    def train_one_batch(self, batch):
        loss = self.loss(batch)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

def generate_experiences(args, dqn, env, mem):
    state, done = env.reset(), False
    for i in tqdm(range(args.memory_capacity)):
        if done:
            state, done = env.reset(), False
        if np.random.rand() < args.dqn_policy_epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = dqn.act(state)
        next_state, reward, done, _ = env.step(action)
        mem.append(state, action, reward, done)
        state = next_state

def train():
    print('making env')
    env = get_train_test_envs(args)[0]
    dqn = Agent(args, env)
    dqn.eval()
    network = FeatureNet(args, env.action_space.n)

    if os.path.exists(args.memory):
        print('loading experiences')
        mem = load_memory(args.memory, disable_bzip=args.disable_bzip_memory)
    else:
        print('generating experiences')
        mem = ReplayMemory(args, args.memory_capacity, env.env.observation_space.shape)
        generate_experiences(args, dqn, env, mem)
        save_memory(mem, 'quick20K.mem', disable_bzip=args.disable_bzip_memory)

    print('sampling')
    batch = mem.sample(args.batch_size)
    loss = 0
    print('training')
    for step in tqdm(range(args.n_training_steps)):
        if not args.overfit_one_batch and step > 0:
            batch = mem.sample(args.batch_size)
        loss += network.train_one_batch(batch)
        if (step + 1) % args.evaluation_interval == 0:
            print(loss / args.evaluation_interval)
            loss = 0

if __name__ == '__main__':
    train()
