import gym
import torch
from torch import nn

from env import get_env
from model import build_phi_network, MarkovHead
from memory import load_memory

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace(
    env = 'QbertNoFrameskip-v4',
    architecture = 'data-efficient',
    n_training_steps = 1000,
    evaluation_interval = 10,
    history_length = 4,
    hidden_size = 256,
    learning_rate = 0.003,
    adam_eps = 1.5e-4,
    batch_size = 32,
    memory = 'results/ccv-2020-05-20-1910/QbertNoFrameskip-v4_learning_rate_0.0001_seed_2/replay_memory.mem',
    disable_bzip_memory = False,
)

class FeatureNet(nn.Module):
    def __init__(self, args, action_space):
        super(FeatureNet, self).__init__()
        self.phi, self.feature_size = build_phi_network(args)
        self.markov_head = MarkovHead(args, self.feature_size, action_space)

    def forward(self, x):
        return self.phi(x)

    def loss(self, batch):
        idxs, states, actions, returns, next_states, nonterminals, weights = batch
        markov_loss = self.markov_head.compute_markov_loss(
            z0 = phi(states),
            z1 = phi(next_states),
            a = actions,
        )
        loss = markov_loss
        return loss

def train_one_batch(batch):
    loss = network.loss(batch)
    network.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.detach().cpu().numpy()

def train():
    print('making env')
    env = gym.make(args.env)
    print('building net')
    network = FeatureNet(args, env.action_space.n)
    print('loading mem')
    mem = load_memory(args.memory, args.disable_bzip_memory)
    print('done.')
    optimizer = optim.Adam(
        network.parameters(),
        lr=args.learning_rate,
        eps=args.adam_eps
    )

    print('sampling')
    batch = mem.sample(args.batch_size)
    loss = 0
    print('training')
    for step in tqdm(range(args.n_training_steps)):
        if not args.overfit_one_batch and step > 0:
            batch = mem.sample(args.batch_size)
        loss += train_one_batch(batch)
        if step % args.evaluation_interval + 1 == 0:
            print(loss / args.evaluation_interval)
            loss = 0

if __name__ == '__main__':
    train()
