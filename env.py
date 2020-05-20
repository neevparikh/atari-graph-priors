# -*- coding: utf-8 -*-
from collections import deque
import torch
import gym
from atariari.benchmark.wrapper import AtariARIWrapper
from gym_wrappers import FrameStack, MaxAndSkipEnv, AtariPreprocess, ResetARI, \
    TorchWrapper, EpisodicLifeEnv


def make_atari(env, num_frames, training=False):
    """ Wrap env in atari processed env """
    if training:
        env = EpisodicLifeEnv(env)
    return FrameStack(MaxAndSkipEnv(AtariPreprocess(env), 4),
                      num_frames)

def make_ram(env, num_frames, training=False):
    """ Wrap env in reset to match observation """
    if training:
        env = EpisodicLifeEnv(env)
    return FrameStack(env, num_frames)


def make_ari(env, num_frames, training=False):
    """ Wrap env in reset to match observation """
    if training:
        env = EpisodicLifeEnv(env)
    return FrameStack(ResetARI(AtariARIWrapper(env)), num_frames)


def get_env(args):
    # Initialize environment
    if args.architecture == 'ram':
        assert '-ram' in args.env, 'Need to use ram environment with ram architecture.'
    env = gym.make(args.env)
    test_env = gym.make(args.env)

    # Get uuid for run
    if args.uuid == 'env':
        uuid_tag = args.env
    elif args.uuid == 'random':
        import uuid
        uuid_tag = uuid.uuid4()
    else:
        uuid_tag = args.uuid

    if args.architecture == 'ram':
        env = make_ram(env, args.history_length, training=True)
        test_env = make_ram(test_env, args.history_length, training=False)
    elif args.architecture == 'ari':
        env = make_ari(env, args.history_length, training=True)
        test_env = make_ari(test_env, args.history_length, training=False)
    else:
        env = make_atari(env, args.history_length, training=True)
        test_env = make_atari(test_env, args.history_length, training=False)

    # env = TorchWrapper(env, args.device)
    # test_env = TorchWrapper(env, args.device)

    # Set tag for this run
    run_tag = args.env
    run_tag += '_' + args.uuid if args.uuid != 'env' else ''
    run_tag += '_ari' if args.architecture == 'ari' else ''
    run_tag += '_seed_' + str(args.seed)

    return env, test_env, run_tag
