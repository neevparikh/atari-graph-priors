# -*- coding: utf-8 -*-
from __future__ import division
import os

import numpy as np
import torch
from cpprb import PrioritizedReplayBuffer

from utils import torchify


class ReplayMemory():
    def __init__(self, args, capacity, env):
        # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_weight = args.priority_weight
        self.n = args.multi_step
        self.device = args.device
        if args.mmap:
            os.makedirs('memories/', exist_ok=True)
            mmap_prefix = 'memories/mm'
        else:
            mmap_prefix = None
        self.buffer = PrioritizedReplayBuffer(
            capacity,
            {
                "obs": {
                    "shape": env.observation_space.shape, "dtype": env.observation_space.dtype
                },
                "next_obs": {
                    "shape": env.observation_space.shape, "dtype": env.observation_space.dtype
                },
                "act": {
                    "shape": 1, "dtype": env.action_space.dtype
                },
                "rew": {
                    "dtype": np.float32
                },
                "done": {
                    "dtype": np.uint8
                },
            },
            Nstep={
                "size": self.n,
                "gamma": args.discount,
                "rew": "rew",
                "next": "next_obs",
            },
            mmap_prefix=mmap_prefix,
            alpha=args.priority_exponent,
            # next_of="obs",
            # stack_compress="obs",
        )

    def append(self, state, next_state, action, reward, done):
        self.buffer.add(**{
            "obs": state,
            "next_obs": next_state,
            "act": action,
            "rew": reward,
            "done": done,
        })

    def sample(self, size):
        s = self.buffer.sample(size, self.priority_weight)
        s['indexes'] = s['indexes'].astype(np.int32)
        return torchify((s['indexes'], torch.int32), (s['obs'], torch.float32),
                        (np.squeeze(s['act'], 1), torch.long),
                        (np.squeeze(s['rew'], 1), torch.float32), (s['next_obs'], torch.float32),
                        (s['done'], torch.bool), (s['weights'], torch.float32),
                        device=self.device)

    def update_priorities(self, indexes, new_priorities):
        indexes = indexes.cpu().numpy()
        self.buffer.update_priorities(indexes, new_priorities)
