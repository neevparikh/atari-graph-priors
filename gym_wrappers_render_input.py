from collections import deque

import torch
import numpy as np
import torchvision.transforms as T
import gym
import cv2

# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class AtariPreprocessPixelInput():
    def __init__(self, shape=(84, 84)): #Do we still want to do this?
        self.shape = shape
        self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize(self.shape),
            T.Lambda(lambda img: np.array(img)),
        ])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )
    def to_grayscale(self,rendered_pixel):
        return np.dot(rendered_pixel[...,:3], [0.299, 0.587, 0.114])
    def get_state(self, rendered_pixel):
        rendered_pixel = self.to_grayscale(rendered_pixel)
        return self.transforms(rendered_pixel)

class CombineRamPixel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # get_pixel_name = env.unwrapped.spec.id
        # self.pixel_env = gym.make(get_pixel_name.replace('-ram',''))
        # print("Found atari game:",self.pixel_env.unwrapped.spec.id)

        self.pixel_env = AtariPreprocessPixelInput()

        self.pixel_shape = self.pixel_env.observation_space.shape
        self.ram_shape = self.observation_space.shape
        new_total_shape = (self.ram_shape[0]+self.pixel_shape[0]*self.pixel_shape[1],)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_total_shape,
            dtype=np.float32
        )


    def combine_states(self,ram_state,pixel_state):
         return np.reshape(np.concatenate((ram_state,pixel_state)),(-1))

    
    def observation(self, obs):
        pixel_state = self.pixel_env.get_state(self.render(mode='rgb_array'))
        return  self.combine_states(obs,pixel_state)

