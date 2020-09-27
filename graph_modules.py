import torch
from torch import nn
import sys
import numpy as np
import random
from collections import deque, namedtuple
import torch.nn.functional as F
from torch.nn.functional import relu
from torch.nn.functional import tanh


class Node_Embed(nn.Module):
    def __init__(self,
                 entities_to_index,
                 latent_entities,
                 edge_list,
                 embed_size,
                 out_embed_size,
                 device=0):
        super(Node_Embed, self).__init__()

        latent_entities = sorted(latent_entities)
        state_keys = sorted(entities_to_index.keys())
        self.state_keys = state_keys
        self.state_indices = [entities_to_index[e] for e in state_keys]

        self.one_hot_rep_template = torch.eye(len(state_keys)).unsqueeze(0).unsqueeze(0).to(device)

        all_nodes = state_keys + latent_entities

        self.one_hot_state_input_template = torch.eye(
            len(all_nodes)).unsqueeze(0).unsqueeze(0).to(device)
        self.lifting_layer = nn.Linear(len(all_nodes), embed_size, bias=False)
        self.lifting_layer.to(device)

        self.node_to_index = {}
        for node_index, node in enumerate(all_nodes):
            self.node_to_index[node] = node_index

        self.latent_node_indices = [self.node_to_index[node] for node in latent_entities]

        print(self.node_to_index)
        print(latent_entities)

        self.adjacency = torch.zeros(len(edge_list),
                                     len(all_nodes),
                                     len(all_nodes),
                                     dtype=torch.uint8).to(device)
        for edge_type in range(len(edge_list)):
            for i in range(len(all_nodes)):
                self.adjacency[edge_type][i][i] = 1
            for s, d in edge_list[edge_type]:
                self.adjacency[edge_type][self.node_to_index[s]][self.node_to_index[d]] = 1

        self.gcn = GCN(self.adjacency,
                       emb_size=embed_size,
                       use_layers=3,
                       activation="relu",
                       device=device)

        self.final_projection = torch.nn.Linear(embed_size, out_embed_size)

        self.final_projection.to(device)

    def forward(self, x, extract_latent=False):
        scaled_key_input = self.one_hot_rep_template.repeat(
            x.shape[0], x.shape[1], 1, 1) * x[:, :, self.state_indices].unsqueeze(-1)

        one_hot_state_input_all = self.one_hot_state_input_template.repeat(
            x.shape[0], x.shape[1], 1, 1)

        one_hot_state_input_all[:, :, :len(self.state_keys), :len(self.state_keys
                                                                 )] *= scaled_key_input

        embedded_nodes = self.lifting_layer(one_hot_state_input_all)

        output = self.gcn(embedded_nodes)

        if extract_latent:
            output = output[:, :, self.latent_node_indices]

        output = self.final_projection(output)

        return output

class Object_Embed(nn.Module):
    def __init__(self,
                 variables_to_index,
                 non_object_entities,
                 edge_list,
                 object_list,
                 embed_size,
                 out_embed_size,
                 device=0):
        super(Object_Embed, self).__init__()


        self.object_list = object_list
        self.x_indices = [variables_to_index[o+"_x"] if o+"_x" in variables_to_index else 128 for o in self.object_list]
        self.y_indices = [variables_to_index[o+"_y"] if o+"_y" in variables_to_index else 128 for o in self.object_list]

        all_nodes = self.object_list + non_object_entities

        print(all_nodes)
        print(self.object_list)
        print(edge_list)

        self.node_to_index = {}
        for node_index, node in enumerate(all_nodes):
            self.node_to_index[node] = node_index

        self.all_object_indices = torch.arange(len(all_nodes)).unsqueeze(0).unsqueeze(0).to(device)
        self.num_objects = len(self.object_list)

        self.node_embedding = nn.Embedding(len(all_nodes),embed_size-2).to(device)
        self.xy_placeholder = torch.zeros(1,1,len(all_nodes),2).to(device)

        self.adjacency = torch.zeros(len(edge_list),
                                     len(all_nodes),
                                     len(all_nodes),
                                     dtype=torch.uint8).to(device)
        for edge_type in range(len(edge_list)):
            for i in range(len(all_nodes)):
                self.adjacency[edge_type][i][i] = 1
            for s, d in edge_list[edge_type]:
                self.adjacency[edge_type][self.node_to_index[s]][self.node_to_index[d]] = 1

        self.gcn = GCN(self.adjacency,
                       emb_size=embed_size,
                       use_layers=3,
                       activation="relu",
                       device=device)

        self.final_projection = torch.nn.Linear(embed_size, out_embed_size).to(device)
        self.lifting_layer = nn.Linear(embed_size, embed_size).to(device)

        self.device = device
      

    def forward(self, x, extract_latent=False):

        x = torch.cat((x,torch.zeros(x.shape[0], x.shape[1],1).to(self.device)),-1)


        # print(self.all_object_indices.shape)

        embeddings = self.node_embedding(self.all_object_indices.repeat(
            x.shape[0], x.shape[1], 1))

        # print(embeddings.shape)

        xy_info = self.xy_placeholder.repeat(
            x.shape[0], x.shape[1], 1,1)

        # print(xy_info.shape)

        # print(x[:, :, self.x_indices].shape)

        # print(xy_info[:,:,:self.num_objects,0].shape,x[:, :, self.x_indices].shape)
        # exit()

        xy_info[:,:,:self.num_objects,0] += x[:, :, self.x_indices]
        xy_info[:,:,:self.num_objects,1] += x[:, :, self.y_indices]

        embedded_nodes = torch.cat((embeddings,xy_info),-1)

        # print(embedded_nodes[0][0][:,-5:])
        # print(self.x_indices)
        # print(xy_info[0][0])

        # exit()

        output = self.gcn(self.lifting_layer(embedded_nodes))

        if extract_latent:
            output = output[:, :, self.latent_node_indices]

        output = self.final_projection(output)

        # print("GCN Output:",output.shape)

        return output


# Adapted from https://github.com/tkipf/pygcn.
class GCN(nn.Module):
    def __init__(self, adj_matrices, emb_size=16, use_layers=3, activation="relu", device=0):
        super(GCN, self).__init__()

        print("starting init")

        self.emb_sz = emb_size

        if activation == "relu":
            self.activation = relu
        elif activation == "tanh":
            self.activation = tanh
        else:
            exit("Bad activation:{}".format(self.activation))

        print("Using activation:", self.activation)

        A_raw = adj_matrices

        self.A = [x.to(device) for x in A_raw]
        self.num_edges = len(A_raw)
        self.use_layers = use_layers
        self.layer_sizes = [(self.emb_sz, self.emb_sz // self.num_edges)] * self.use_layers

        self.num_layers = len(self.layer_sizes)
        self.weights = [[
            torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
            for (in_dim, out_dim) in self.layer_sizes
        ]
                        for e in range(self.num_edges)]

        for i in range(self.num_edges):
            for j in range(self.num_layers):
                self.add_module(str((i, j)), self.weights[i][j])

        # self.final_mapping = torch.nn.Linear(self.emb_sz, self.emb_sz)

        print("finished initializing")

    def ND_2D_mm(self, x, y):
        #t(x)= [bs,history,embeddings,num_nodes]
        #-> [bs*history*embeddings,num_nodes]
        #[46x46]
        #[bs*history*embeddings,num_nodes]
        #[bs,history,embeddings,num_nodes]
        #[bs,history,num_nodes,embeddings]

        x_new = x.reshape(-1, x.shape[-1])
        result = torch.mm(x_new, y)
        return result.view(list(x.shape[:-1]) + [y.shape[-1]])

    def forward(self, x):
        # print(x.shape)
        for l in range(self.num_layers):
            layer_out = []
            for e in range(self.num_edges):
                weighting = F.normalize(self.A[e].float(), dim=0)

                layer_out.append(self.ND_2D_mm(x.transpose(3, 2), weighting).transpose(3, 2))
                # print(layer_out[-1].shape)

            x = torch.cat([
                self.activation(self.weights[e][l](type_features)) for e,
                type_features in enumerate(layer_out)
            ],
                          axis=-1)
            # print(x.shape)
        # x = self.final_mapping(x)
        return x
