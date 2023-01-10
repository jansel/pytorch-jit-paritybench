import sys
_module = sys.modules[__name__]
del sys
monitor = _module
replay_memory = _module
result_show = _module
wrappers = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import random


import time


from collections import deque


import matplotlib.pyplot as plt


from copy import deepcopy


N_ENVS = 16


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class LazyFrames(object):

    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class ResultsWriter(object):

    def __init__(self, filename=None, header='', extra_keys=()):
        self.extra_keys = extra_keys
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + '.' + Monitor.EXT
            self.f = open(filename, 'wt')
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't') + tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


def wrap_cover(env_name):

    def wrap_():
        """Apply a common set of wrappers for Atari games."""
        env = gym.make(env_name)
        env = Monitor(env, './')
        assert 'NoFrameskip' in env.spec.id
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ProcessFrame84(env)
        env = ImageToPyTorch(env)
        env = FrameStack(env, 4)
        env = ClippedRewardsWrapper(env)
        return env
    return wrap_


N_OPTIONS = 10


N_QUANT = 200


STATE_LEN = 4


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.fc_quantiles = nn.Linear(512, N_ACTIONS * N_QUANT)
        self.fc_options = nn.Linear(512, N_OPTIONS)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        mb_size = x.size(0)
        y = self.feature_extraction(x / 255.0)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        quantiles = self.fc_quantiles(y)
        quantiles = quantiles.view(-1, N_ACTIONS, N_QUANT)
        options = self.fc_options(y)
        return quantiles, options

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

