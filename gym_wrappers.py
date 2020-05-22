import gym
from collections import deque
import numpy as np
import torchvision.transforms as T
# import cv2
import torch


class TorchWrapper(gym.Wrapper):

    def __init__(self, env, device):
        gym.Wrapper.__init__(self, env)
        self.device = device

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return torch.as_tensor(state.astype(np.float32)).to(self.device)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return torch.as_tensor(next_state.astype(np.float32)).to(self.device), reward, done, info


class ResetARI(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # change the observation space to accurately represent
        # the shape of the labeled RAM observations
        self.observation_space = gym.spaces.Box(
            0,
            255,  # max value
            shape=(len(self.env.labels()),),
            dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # reset the env and get the current labeled RAM
        return np.array(list(self.env.labels().values()))

    def step(self, action):
        # we don't need the obs here, just the labels in info
        _, reward, done, info = self.env.step(action)
        # grab the labeled RAM out of info and put as next_state
        next_state = np.array(list(info['labels'].values()))
        return next_state, reward, done, info


# WARNING
# Only works for env with 'player_x' and 'player_y' annotations
class ResetARIOneHot(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # change the observation space to accurately represent
        # the shape of the labeled RAM observations
        self.observation_space = gym.spaces.Box(
            0,
            255,  # max value
            shape=(len(self.info_to_state(self.env.labels())),),
            dtype=np.uint8)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        # reset the env and get the current labeled RAM
        return self.info_to_state(self.env.labels())

    def step(self, action):
        # we don't need the obs here, just the labels in info
        _, reward, done, info = self.env.step(action)
        next_state = self.info_to_state(info['labels'])
        return next_state, reward, done, info

    def info_to_state(self, labels):
        # one hot x and y position
        x_one_hot = np.eye(256)[labels['player_x']]
        y_one_hot = np.eye(256)[labels['player_y']]

        # grab the rest of the labeled RAM out of info and put as next_state
        next_state = []
        for l in labels.keys():
            if l == 'player_x' or l == 'player_y':
                continue
            next_state.append(labels[l])

        # join one hot positions to labeled RAM
        return np.concatenate((x_one_hot,y_one_hot,np.array(next_state)))


# Adapted from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class AtariPreprocess(gym.Wrapper):

    def __init__(self, env, shape=(84, 84)):
        """ Preprocessing as described in the Nature DQN paper (Mnih 2015) """
        gym.Wrapper.__init__(self, env)
        self.shape = shape
        self.transforms = T.Compose([
            T.ToPILImage(mode='YCbCr'),
            T.Lambda(lambda img: img.split()[0]),
            T.Resize(self.shape),
            T.Lambda(lambda img: np.array(img, copy=False)),
        ])
        # self.transforms = lambda img: cv2.resize(cv2.cvtColor(
        #     img, cv2.COLOR_RGB2GRAY),
        #                                          shape,
        #                                          interpolation=cv2.INTER_AREA)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.shape,
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        return self.transforms(self.env.reset(**kwargs))

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self.transforms(next_state), reward, done, info


class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=((k,) + shp),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return torch.as_tensor(np.stack(list(self.frames),
                                        axis=0)).to(dtype=torch.float32).div_(255)


class LazyFrames(object):

    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once.  It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers.  This object should only be
        converted to numpy array before being passed to the model."""
        self._frames = frames

    def _force(self):
        return np.stack(self._frames, axis=0)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]
