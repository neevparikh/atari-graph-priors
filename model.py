# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

from modules import Reshape
from graph_modules import Node_Embed

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

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
            return F.linear(input,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space, observation_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.architecture = args.architecture
        self.action_space = action_space
        self.observation_space = observation_space

        if self.architecture == 'canonical':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 4, stride=2, padding=0),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, 3, stride=1, padding=0),
                                       nn.ReLU())
            self.conv_output_size = 3136
        elif self.architecture == 'data-efficient':
            self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 5, stride=5, padding=0),
                                       nn.ReLU())
            self.conv_output_size = 576

        elif self.architecture == 'mlp':

            self.input_shape = self.observation_space.shape[1] * args.history_length
            self.convs = nn.Sequential(Reshape(-1, self.input_shape),
                                       nn.Linear(self.input_shape, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 512),
                                       nn.ReLU())
            self.conv_output_size = args.hidden_size

        elif self.architecture == 'mlp_gcn':

            berzerk_entities_to_index = {
                    "player_x":19,
                    "player_y":11,
                    "player_direction":14,
                    "player_missile_x":22,
                    "player_missile_y":23,
                    "player_missile_direction":21,
                    "robot_missile_direction":26,
                    "robot_missile_x":29,
                    "robot_missile_y":30,
                    "num_lives":90,
                    "robots_killed_count":91,
                    "game_level":92,
                    "enemy_evilOtto_x":46,
                    "enemy_evilOtto_y":89
                    }

            berzerk_latent_entities = ["player","game","evilOtto","player_missile","robot_missile"]

            edge_type_0 = [
            ("player_x","player"),
            ("player_y","player"),
            ("player_direction","player"),
            ("num_lives","game"),
            ("game_level","game"),
            ("player_y","player"),
            ("robots_killed_count","game"),
            ("enemy_evilOtto_x","evilOtto"),
            ("enemy_evilOtto_y","evilOtto"),
            ("player_missile_x","player_missile"),
            ("player_missile_y","player_missile"),
            ("robot_missile_x","robot_missile"),
            ("robot_missile_y","robot_missile")]

            for i,value in enumerate(range(65,73)):
                berzerk_entities_to_index["enemy_robots_x_{}".format(i)]  = value
                berzerk_latent_entities.append("enemy_robot_{}".format(i))
                edge_type_0.append(("enemy_robots_x_{}".format(i),"enemy_robot_{}".format(i)))

            for i,value in enumerate(range(56, 64)):#Seems like bug in their berzerk code, y != x

                #for i,value in enumerate(range(56, 65)):#Seems like bug in their berzerk code, y != x

                berzerk_entities_to_index["enemy_robots_y_{}".format(i)]  = value
                edge_type_0.append(("enemy_robots_y_{}".format(i),"enemy_robot_{}".format(i)))


            for i,value in enumerate(range(93, 96)):
                berzerk_entities_to_index["player_score_{}".format(i)]  = value
                edge_type_0.append(("player_score_{}".format(i),"game"))


            embed_size = 32
            final_embed_size = 32
            self.node_embed = Node_Embed(berzerk_entities_to_index,latent_entities=berzerk_latent_entities,edge_list=[edge_type_0],embed_size=embed_size,out_embed_size=final_embed_size)
            print("GCN param:",sum([param.numel() for param in self.node_embed.parameters()]))

            # self.input_shape = args.history_length*len(list(berzerk_entities_to_index.keys())+berzerk_latent_entities)*embed_size #self.observation_space.shape[1] * args.history_length
            num_entities = len(list(berzerk_entities_to_index.keys())+berzerk_latent_entities)
            self.input_shape = args.history_length*num_entities*final_embed_size #self.observation_space.shape[1] * args.history_length

            self.convs = nn.Sequential(nn.Conv2d(num_entities, 64, (final_embed_size,2), stride=1, padding=0),
                                       nn.Conv2d(64,64, (1,2), stride=1, padding=0),

                                         # nn.Linear(self.input_shape, args.hidden_size),
                                       Reshape(-1, 128),
                                       nn.Linear(128, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, 512),
                                       nn.ReLU())

            self.conv_output_size = 512
        else:
            raise ValueError("architecture not recognized: {}".format(args.architecture))
        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size,
                                  action_space * self.atoms,
                                  std_init=args.noisy_std)


        print("All param:",sum([param.numel() for param in self.parameters()]))


    def forward(self, x, log=False):

        if self.architecture == 'mlp_gcn':
            x = F.relu(self.node_embed(x,extract_latent=False))
            x = x.permute(0,2,3,1)
            # x = x.view(x.shape[0],x.shape[1],-1) #Flatten everything. May need to rethink.

        x = self.convs(x)

        if self.architecture in ['canonical', 'data-efficient']:
            x = x.view(-1, self.conv_output_size)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
