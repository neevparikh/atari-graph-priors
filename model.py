# -*- coding: utf-8 -*-
from __future__ import division
import math
import os
import torch
from torch import nn
from torch.nn import functional as F

class Reshape(nn.Module):
    """Module that returns a view of the input which has a different size

    Parameters
    ----------
    args : int...
        The desired size
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    def __repr__(self):
        s = self.__class__.__name__
        s += '{}'.format(self.shape)
        return s
    def forward(self, input):
        return input.view(*self.shape)

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features,
                                                     in_features))
        self.register_buffer('weight_epsilon',
                             torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init /
                                     math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init /
                                   math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

def build_phi_network(args, input_shape=None):
    if args.architecture == 'canonical':
        output_size = 3136
        phi = nn.Sequential(
            nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            Reshape(-1, output_size))

    elif args.architecture in ['ari','ram']:
        assert input_shape is not None, 'Must specify input_shape for this architecture'
        shape_flat = torch.prod(torch.as_tensor(input_shape))
        output_size = 576
        layers = [Reshape(-1, shape_flat), nn.Linear(shape_flat, 128), nn.ReLU(),
                nn.Linear(128, output_size), nn.ReLU()]
        phi = nn.Sequential(*layers)

    elif args.architecture in ['data-efficient', 'online']:
        output_size = 576
        phi = nn.Sequential(
            nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=5, padding=0),
            nn.ReLU(),
            Reshape(-1, output_size))

    elif args.architecture == 'pretrained':
        assert args.phi_net_path is not None, 'Must specify path to feature network'
        # Always load tensors onto CPU by default, will shift to GPU if necessary
        phi, output_size = torch.load(args.phi_net_path, map_location='cpu')
        phi = phi.to(device=args.device)
        for param in phi.parameters():
            param.requires_grad = False

    else:
        raise ValueError('Invalid architecture: {}'.format(args.architecture))

    return phi, output_size

class DQN(nn.Module):
    def __init__(self, args, observation_space, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.state_space = observation_space
        self.action_space = action_space

        self.phi, self.feature_size = build_phi_network(args,
            input_shape=self.state_space.shape)
        if args.architecture == 'online':
            self.markov_head = MarkovHead(args, self.feature_size, action_space)
        self.fc_h_v = NoisyLinear(self.feature_size,
                                  args.hidden_size,
                                  std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.feature_size,
                                  args.hidden_size,
                                  std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size,
                                  self.atoms,
                                  std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size,
                                  action_space * self.atoms,
                                  std_init=args.noisy_std)

        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def save_phi(self, path, name):
        full_path = os.path.join(path, name)
        torch.save((self.phi, self.feature_size), full_path)
        return full_path

    def forward(self, x, log=False):
        x = self.phi(x)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space,
                                                 self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(
                q,
                dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(
                q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()

class InverseModel(nn.Module):
    def __init__(self, args, feature_size, action_space):
        super(InverseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size*2, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, action_space)
        )
    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)

class ContrastiveModel(nn.Module):
    def __init__(self, args, feature_size):
        super(ContrastiveModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_size*2, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )
    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)

class MarkovHead(nn.Module):
    def __init__(self, args, feature_size, action_space):
        super(MarkovHead, self).__init__()
        self.inverse_model = InverseModel(args, feature_size, action_space)
        self.discriminator = ContrastiveModel(args, feature_size)

        self.bce = torch.nn.BCEWithLogitsLoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_markov_loss(self, z0, z1, a):
        # Inverse loss
        log_pr_actions = self.inverse_model(z0, z1)
        l_inverse = self.ce(input=log_pr_actions, target=a)

        # Ratio loss
        with torch.no_grad():
            N = len(z1)
            idx = torch.randperm(N) # shuffle indices of next states
        z1_neg = z1.view(N,-1)[idx].view(z1.size())
        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_real_transition = torch.cat([torch.ones(N), torch.zeros(N)], dim=0)
        log_pr_real = self.discriminator(z0_extended, z1_pos_neg)
        l_ratio = self.bce(input=log_pr_real, target=is_real_transition.unsqueeze(-1).float())

        markov_loss = l_inverse + l_ratio
        return markov_loss
