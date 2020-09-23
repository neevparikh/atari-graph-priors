# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import gym

class Env():
  def __init__(self, args):
    self.device = args.device
    self.env = gym.make(args.env)
    self.env.seed(args.seed)
    # print(self.env.__dict__)
    self.env._max_episode_steps = args.max_episode_length
    # self.ale = atari_py.ALEInterface()
    # self.ale.setInt('random_seed', args.seed)
    # self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    # self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    # self.ale.setInt('frame_skip', 0)
    # self.ale.setBool('color_averaging', False)
    # self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    # actions = self.ale.getMinimalActionSet()
    # self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    # print(self.env.get_action_meanings())
    self.actions = self.env.get_action_meanings()#dict([i, e] for i, e in zip(range(len(self.env.action_space.n)), actions))

    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode
    # self.ale.observation_space = gym.spaces.Box(
    #         low=0,
    #         high=255,
    #         shape=(84*84,),
    #         dtype=np.uint8,
    #     )
    self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84*84+128,),
            dtype=np.uint8,
        )
    self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize((84,84)),
            T.Lambda(lambda img: np.array(img)),
        ])

  def _get_state(self,ram_obs):

    state = self.transforms(self.env.render('rgb_array'))

    # state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    state = np.reshape(state,-1)
    state = np.concatenate((np.reshape(ram_obs,-1),state),-1)

    # print(state.shape)

    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      # self.state_buffer.append(torch.zeros(84, 84, device=self.device))
      self.state_buffer.append(torch.zeros(84*84+128, device=self.device))


  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      ram_obs = self.env.reset()
      # self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        ram_obs, _, done, _ = self.env.step(0)  # Assumes raw action 0 is always no-op
        if done: #self.ale.game_over():
          ram_obs = self.env.reset()
          #self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state(ram_obs)
    self.state_buffer.append(observation)
    # self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    # frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    frame_buffer = torch.zeros(2, 84*84+128, device=self.device)
    reward, done = 0, False
    for t in range(4):
      # print(action)
      ram_obs, r, done, info = self.env.step(action) #self.actions.get(action)) #self.ale.act(self.actions.get(action))
      reward+=r
      # print(info)
      # print(frame_buffer.shape)
      if t == 2:
        frame_buffer[0] = self._get_state(ram_obs)
      elif t == 3:
        frame_buffer[1] = self._get_state(ram_obs)
      # done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    # if self.training:
      # lives = self.ale.lives()
      # if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        # self.life_termination = not done  # Only set flag when not truly done
        # done = True
      # self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    self.env.render()
    # cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    # cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
