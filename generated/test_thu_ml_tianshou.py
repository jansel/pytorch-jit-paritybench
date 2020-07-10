import sys
_module = sys.modules[__name__]
del sys
conf = _module
ant_v2_ddpg = _module
ant_v2_sac = _module
ant_v2_td3 = _module
continuous_net = _module
discrete_net = _module
halfcheetahBullet_v0_sac = _module
mujoco = _module
maze_env_utils = _module
point = _module
point_maze_env = _module
register = _module
point_maze_td3 = _module
pong_a2c = _module
pong_dqn = _module
pong_ppo = _module
setup = _module
test = _module
base = _module
env = _module
test_batch = _module
test_buffer = _module
test_collector = _module
test_env = _module
continuous = _module
net = _module
test_ddpg = _module
test_ppo = _module
test_sac_with_il = _module
test_td3 = _module
discrete = _module
net = _module
test_a2c_with_il = _module
test_dqn = _module
test_drqn = _module
test_pdqn = _module
test_pg = _module
test_ppo = _module
tianshou = _module
data = _module
batch = _module
buffer = _module
collector = _module
atari = _module
utils = _module
vecenv = _module
exploration = _module
random = _module
policy = _module
base = _module
imitation = _module
base = _module
modelfree = _module
a2c = _module
ddpg = _module
dqn = _module
pg = _module
ppo = _module
sac = _module
td3 = _module
utils = _module
trainer = _module
offpolicy = _module
onpolicy = _module
config = _module
moving_average = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


from torch.utils.tensorboard import SummaryWriter


from torch import nn


import torch.nn.functional as F


import re


import time


from typing import Any


from typing import List


from typing import Union


from typing import Iterator


from typing import Optional


import warnings


from typing import Dict


from typing import Callable


from abc import ABC


from abc import abstractmethod


from copy import deepcopy


from typing import Tuple


class Actor(nn.Module):

    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class ActorProb(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class Critic(nn.Module):

    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s, **kwargs):
        logits, h = self.preprocess(s, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits


class Net(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu', softmax=False):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        if softmax:
            self.model += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class DQN(nn.Module):

    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_shape)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.reshape(x.size(0), -1))
        return self.head(x), state


class RecurrentActorProb(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action, device='cpu'):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape), hidden_size=128, num_layers=layer_num, batch_first=True)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = s.view(bsz, length, -1)
        logits, _ = self.nn(s)
        logits = logits[:, (-1)]
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class RecurrentCritic(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape), hidden_size=128, num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(128 + np.prod(action_shape), 1)

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, (-1)]
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s


class Recurrent(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.nn = nn.LSTM(input_size=128, hidden_size=128, num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = self.fc1(s.view([bsz * length, dim]))
        s = s.view(bsz, length, -1)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            s, (h, c) = self.nn(s, (state['h'].transpose(0, 1).contiguous(), state['c'].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, (-1)])
        return s, {'h': h.transpose(0, 1).detach(), 'c': c.transpose(0, 1).detach()}


class Batch(object):
    """Tianshou provides :class:`~tianshou.data.Batch` as the internal data
    structure to pass any kind of data to other methods, for example, a
    collector gives a :class:`~tianshou.data.Batch` to policy for learning.
    Here is the usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import Batch
        >>> data = Batch(a=4, b=[5, 5], c='2312312')
        >>> data.b
        [5, 5]
        >>> data.b = np.array([3, 4, 5])
        >>> print(data)
        Batch(
            a: 4,
            b: array([3, 4, 5]),
            c: '2312312',
        )

    In short, you can define a :class:`Batch` with any key-value pair. The
    current implementation of Tianshou typically use 7 reserved keys in
    :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()``        function return 4 arguments, and the last one is ``info``);
    * ``policy`` the data computed by policy in step :math:`t`;

    :class:`~tianshou.data.Batch` has other methods, including
    :meth:`~tianshou.data.Batch.__getitem__`,
    :meth:`~tianshou.data.Batch.__len__`,
    :meth:`~tianshou.data.Batch.append`,
    and :meth:`~tianshou.data.Batch.split`:
    ::

        >>> data = Batch(obs=np.array([0, 11, 22]), rew=np.array([6, 6, 6]))
        >>> # here we test __getitem__
        >>> index = [2, 1]
        >>> data[index].obs
        array([22, 11])

        >>> # here we test __len__
        >>> len(data)
        3

        >>> data.append(data)  # similar to list.append
        >>> data.obs
        array([0, 11, 22, 0, 11, 22])

        >>> # split whole data into multiple small batch
        >>> for d in data.split(size=2, shuffle=False):
        ...     print(d.obs, d.rew)
        [ 0 11] [6 6]
        [22  0] [6 6]
        [11 22] [6 6]
    """

    def __init__(self, **kwargs) ->None:
        super().__init__()
        self._meta = {}
        for k, v in kwargs.items():
            if (isinstance(v, list) or isinstance(v, np.ndarray)) and len(v) > 0 and isinstance(v[0], dict) and k != 'info':
                self._meta[k] = list(v[0].keys())
                for k_ in v[0].keys():
                    k__ = '_' + k + '@' + k_
                    self.__dict__[k__] = np.array([v[i][k_] for i in range(len(v))])
            elif isinstance(v, dict) or isinstance(v, Batch):
                self._meta[k] = list(v.keys())
                for k_ in v.keys():
                    k__ = '_' + k + '@' + k_
                    self.__dict__[k__] = v[k_]
            else:
                self.__dict__[k] = kwargs[k]

    def __getitem__(self, index: Union[str, slice]) ->Union['Batch', dict]:
        """Return self[index]."""
        if isinstance(index, str):
            return self.__getattr__(index)
        b = Batch()
        for k in self.__dict__:
            if k != '_meta' and self.__dict__[k] is not None:
                b.__dict__.update(**{k: self.__dict__[k][index]})
        b._meta = self._meta
        return b

    def __getattr__(self, key: str) ->Union['Batch', Any]:
        """Return self.key"""
        if key not in self._meta:
            if key not in self.__dict__:
                raise AttributeError(key)
            return self.__dict__[key]
        d = {}
        for k_ in self._meta[key]:
            k__ = '_' + key + '@' + k_
            d[k_] = self.__dict__[k__]
        return Batch(**d)

    def __repr__(self) ->str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k in sorted(list(self.__dict__) + list(self._meta)):
            if k[0] != '_' and (self.__dict__.get(k, None) is not None or k in self._meta):
                rpl = '\n' + ' ' * (6 + len(k))
                obj = pprint.pformat(self.__getattr__(k)).replace('\n', rpl)
                s += f'    {k}: {obj},\n'
                flag = True
        if flag:
            s += ')'
        else:
            s = self.__class__.__name__ + '()'
        return s

    def keys(self) ->List[str]:
        """Return self.keys()."""
        return sorted([i for i in self.__dict__ if i[0] != '_'] + list(self._meta))

    def get(self, k: str, d: Optional[Any]=None) ->Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        if k in self.__dict__ or k in self._meta:
            return self.__getattr__(k)
        return d

    def to_numpy(self) ->np.ndarray:
        """Change all torch.Tensor to numpy.ndarray. This is an inplace
        operation.
        """
        for k in self.__dict__:
            if isinstance(self.__dict__[k], torch.Tensor):
                self.__dict__[k] = self.__dict__[k].cpu().numpy()

    def append(self, batch: 'Batch') ->None:
        """Append a :class:`~tianshou.data.Batch` object to current batch."""
        assert isinstance(batch, Batch), 'Only append Batch is allowed!'
        for k in batch.__dict__:
            if k == '_meta':
                self._meta.update(batch._meta)
                continue
            if batch.__dict__[k] is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = batch.__dict__[k]
            elif isinstance(batch.__dict__[k], np.ndarray):
                self.__dict__[k] = np.concatenate([self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], torch.Tensor):
                self.__dict__[k] = torch.cat([self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], list):
                self.__dict__[k] += batch.__dict__[k]
            else:
                s = 'No support for append with type' + str(type(batch.__dict__[k])) + 'in class Batch.'
                raise TypeError(s)

    def __len__(self) ->int:
        """Return len(self)."""
        return min([len(self.__dict__[k]) for k in self.__dict__ if k != '_meta' and self.__dict__[k] is not None])

    def split(self, size: Optional[int]=None, shuffle: bool=True) ->Iterator['Batch']:
        """Split whole data into multiple small batch.

        :param int size: if it is ``None``, it does not split the data batch;
            otherwise it will divide the data batch with the given size.
            Default to ``None``.
        :param bool shuffle: randomly shuffle the entire data batch if it is
            ``True``, otherwise remain in the same. Default to ``True``.
        """
        length = len(self)
        if size is None:
            size = length
        temp = 0
        if shuffle:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        while temp < length:
            yield self[index[temp:temp + size]]
            temp += size


class ReplayBuffer(object):
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from
    interaction between the policy and environment. It stores basically 7 types
    of data, as mentioned in :class:`~tianshou.data.Batch`, based on
    ``numpy.ndarray``. Here is the usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import ReplayBuffer
        >>> buf = ReplayBuffer(size=20)
        >>> for i in range(3):
        ...     buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> len(buf)
        3
        >>> buf.obs
        # since we set size = 20, len(buf.obs) == 20.
        array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0.])

        >>> buf2 = ReplayBuffer(size=10)
        >>> for i in range(15):
        ...     buf2.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
        >>> len(buf2)
        10
        >>> buf2.obs
        # since its size = 10, it only stores the last 10 steps' result.
        array([10., 11., 12., 13., 14.,  5.,  6.,  7.,  8.,  9.])

        >>> # move buf2's result into buf (meanwhile keep it chronologically)
        >>> buf.update(buf2)
        array([ 0.,  1.,  2.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.])

        >>> # get a random sample from buffer
        >>> # the batch_data is equal to buf[incide].
        >>> batch_data, indice = buf.sample(batch_size=4)
        >>> batch_data.obs == buf[indice].obs
        array([ True,  True,  True,  True])

    :class:`~tianshou.data.ReplayBuffer` also supports frame_stack sampling
    (typically for RNN usage, see issue#19), ignoring storing the next
    observation (save memory in atari tasks), and multi-modal observation (see
    issue#38, need version >= 0.2.3):
    ::

        >>> buf = ReplayBuffer(size=9, stack_num=4, ignore_obs_next=True)
        >>> for i in range(16):
        ...     done = i % 5 == 0
        ...     buf.add(obs={'id': i}, act=i, rew=i, done=done,
        ...             obs_next={'id': i + 1})
        >>> print(buf)  # you can see obs_next is not saved in buf
        ReplayBuffer(
            act: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
            done: array([0., 1., 0., 0., 0., 0., 1., 0., 0.]),
            info: array([{}, {}, {}, {}, {}, {}, {}, {}, {}], dtype=object),
            obs: Batch(
                     id: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
                 ),
            policy: Batch(),
            rew: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
        )
        >>> index = np.arange(len(buf))
        >>> print(buf.get(index, 'obs').id)
        [[ 7.  7.  8.  9.]
         [ 7.  8.  9. 10.]
         [11. 11. 11. 11.]
         [11. 11. 11. 12.]
         [11. 11. 12. 13.]
         [11. 12. 13. 14.]
         [12. 13. 14. 15.]
         [ 7.  7.  7.  7.]
         [ 7.  7.  7.  8.]]
        >>> # here is another way to get the stacked data
        >>> # (stack only for obs and obs_next)
        >>> abs(buf.get(index, 'obs')['id'] - buf[index].obs.id).sum().sum()
        0.0
        >>> # we can get obs_next through __getitem__, even if it doesn't exist
        >>> print(buf[:].obs_next.id)
        [[ 7.  8.  9. 10.]
         [ 7.  8.  9. 10.]
         [11. 11. 11. 12.]
         [11. 11. 12. 13.]
         [11. 12. 13. 14.]
         [12. 13. 14. 15.]
         [12. 13. 14. 15.]
         [ 7.  7.  7.  8.]
         [ 7.  7.  8.  9.]]
    """

    def __init__(self, size: int, stack_num: Optional[int]=0, ignore_obs_next: bool=False, **kwargs) ->None:
        super().__init__()
        self._maxsize = size
        self._stack = stack_num
        self._save_s_ = not ignore_obs_next
        self._meta = {}
        self.reset()

    def __len__(self) ->int:
        """Return len(self)."""
        return self._size

    def __repr__(self) ->str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k in sorted(list(self.__dict__) + list(self._meta)):
            if k[0] != '_' and (self.__dict__.get(k, None) is not None or k in self._meta):
                rpl = '\n' + ' ' * (6 + len(k))
                obj = pprint.pformat(self.__getattr__(k)).replace('\n', rpl)
                s += f'    {k}: {obj},\n'
                flag = True
        if flag:
            s += ')'
        else:
            s = self.__class__.__name__ + '()'
        return s

    def __getattr__(self, key: str) ->Union[Batch, np.ndarray]:
        """Return self.key"""
        if key not in self._meta:
            if key not in self.__dict__:
                raise AttributeError(key)
            return self.__dict__[key]
        d = {}
        for k_ in self._meta[key]:
            k__ = '_' + key + '@' + k_
            d[k_] = self.__dict__[k__]
        return Batch(**d)

    def _add_to_buffer(self, name: str, inst: Union[dict, Batch, np.ndarray, float, int, bool]) ->None:
        if inst is None:
            if getattr(self, name, None) is None:
                self.__dict__[name] = None
            return
        if name in self._meta:
            for k in inst.keys():
                self._add_to_buffer('_' + name + '@' + k, inst[k])
            return
        if self.__dict__.get(name, None) is None:
            if isinstance(inst, np.ndarray):
                self.__dict__[name] = np.zeros([self._maxsize, *inst.shape])
            elif isinstance(inst, dict) or isinstance(inst, Batch):
                if name == 'info':
                    self.__dict__[name] = np.array([{} for _ in range(self._maxsize)])
                else:
                    if self._meta.get(name, None) is None:
                        self._meta[name] = list(inst.keys())
                    for k in inst.keys():
                        k_ = '_' + name + '@' + k
                        self._add_to_buffer(k_, inst[k])
            else:
                self.__dict__[name] = np.zeros([self._maxsize])
        if isinstance(inst, np.ndarray) and self.__dict__[name].shape[1:] != inst.shape:
            raise ValueError(f'Cannot add data to a buffer with different shape, key: {name}, expect shape: {self.__dict__[name].shape[1:]}, given shape: {inst.shape}.')
        if name not in self._meta:
            if name == 'info':
                inst = deepcopy(inst)
            self.__dict__[name][self._index] = inst

    def update(self, buffer: 'ReplayBuffer') ->None:
        """Move the data from the given buffer to self."""
        i = begin = buffer._index % len(buffer)
        while True:
            self.add(buffer.obs[i], buffer.act[i], buffer.rew[i], buffer.done[i], buffer.obs_next[i] if self._save_s_ else None, buffer.info[i], buffer.policy[i])
            i = (i + 1) % len(buffer)
            if i == begin:
                break

    def add(self, obs: Union[dict, np.ndarray], act: Union[np.ndarray, float], rew: float, done: bool, obs_next: Optional[Union[dict, np.ndarray]]=None, info: dict={}, policy: Optional[Union[dict, Batch]]={}, **kwargs) ->None:
        """Add a batch of data into replay buffer."""
        assert isinstance(info, dict), 'You should return a dict in the last argument of env.step().'
        self._add_to_buffer('obs', obs)
        self._add_to_buffer('act', act)
        self._add_to_buffer('rew', rew)
        self._add_to_buffer('done', done)
        if self._save_s_:
            self._add_to_buffer('obs_next', obs_next)
        self._add_to_buffer('info', info)
        self._add_to_buffer('policy', policy)
        if self._maxsize > 0:
            self._size = min(self._size + 1, self._maxsize)
            self._index = (self._index + 1) % self._maxsize
        else:
            self._size = self._index = self._index + 1

    def reset(self) ->None:
        """Clear all the data in replay buffer."""
        self._index = self._size = 0

    def sample(self, batch_size: int) ->Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size equal to batch_size.         Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        if batch_size > 0:
            indice = np.random.choice(self._size, batch_size)
        else:
            indice = np.concatenate([np.arange(self._index, self._size), np.arange(0, self._index)])
        return self[indice], indice

    def get(self, indice: Union[slice, np.ndarray], key: str, stack_num: Optional[int]=None) ->Union[Batch, np.ndarray]:
        """Return the stacked result, e.g. [s_{t-3}, s_{t-2}, s_{t-1}, s_t],
        where s is self.key, t is indice. The stack_num (here equals to 4) is
        given from buffer initialization procedure.
        """
        if stack_num is None:
            stack_num = self._stack
        if not isinstance(indice, np.ndarray):
            if np.isscalar(indice):
                indice = np.array(indice)
            elif isinstance(indice, slice):
                indice = np.arange(0 if indice.start is None else self._size - indice.start if indice.start < 0 else indice.start, self._size if indice.stop is None else self._size - indice.stop if indice.stop < 0 else indice.stop, 1 if indice.step is None else indice.step)
        last_index = (self._index - 1 + self._size) % self._size
        last_done, self.done[last_index] = self.done[last_index], True
        if key == 'obs_next' and not self._save_s_:
            indice += 1 - self.done[indice].astype(np.int)
            indice[indice == self._size] = 0
            key = 'obs'
        if stack_num == 0:
            self.done[last_index] = last_done
            if key in self._meta:
                return {k: self.__dict__['_' + key + '@' + k][indice] for k in self._meta[key]}
            else:
                return self.__dict__[key][indice]
        if key in self._meta:
            many_keys = self._meta[key]
            stack = {k: [] for k in self._meta[key]}
        else:
            stack = []
            many_keys = None
        for i in range(stack_num):
            if many_keys is not None:
                for k_ in many_keys:
                    k__ = '_' + key + '@' + k_
                    stack[k_] = [self.__dict__[k__][indice]] + stack[k_]
            else:
                stack = [self.__dict__[key][indice]] + stack
            pre_indice = indice - 1
            pre_indice[pre_indice == -1] = self._size - 1
            indice = pre_indice + self.done[pre_indice].astype(np.int)
            indice[indice == self._size] = 0
        self.done[last_index] = last_done
        if many_keys is not None:
            for k in stack:
                stack[k] = np.stack(stack[k], axis=1)
            stack = Batch(**stack)
        else:
            stack = np.stack(stack, axis=1)
        return stack

    def __getitem__(self, index: Union[slice, np.ndarray]) ->Batch:
        """Return a data batch: self[index]. If stack_num is set to be > 0,
        return the stacked obs and obs_next with shape [batch, len, ...].
        """
        return Batch(obs=self.get(index, 'obs'), act=self.act[index], rew=self.rew[index], done=self.done[index], obs_next=self.get(index, 'obs_next'), info=self.info[index], policy=self.get(index, 'policy'))


class BasePolicy(ABC, nn.Module):
    """Tianshou aims to modularizing RL algorithms. It comes into several
    classes of policies in Tianshou. All of the policy classes must inherit
    :class:`~tianshou.policy.BasePolicy`.

    A policy class typically has four parts:

    * :meth:`~tianshou.policy.BasePolicy.__init__`: initialize the policy,         including coping the target network and so on;
    * :meth:`~tianshou.policy.BasePolicy.forward`: compute action with given         observation;
    * :meth:`~tianshou.policy.BasePolicy.process_fn`: pre-process data from         the replay buffer (this function can interact with replay buffer);
    * :meth:`~tianshou.policy.BasePolicy.learn`: update policy with a given         batch of data.

    Most of the policy needs a neural network to predict the action and an
    optimizer to optimize the policy. The rules of self-defined networks are:

    1. Input: observation ``obs`` (may be a ``numpy.ndarray`` or         ``torch.Tensor``), hidden state ``state`` (for RNN usage), and other         information ``info`` provided by the environment.
    2. Output: some ``logits`` and the next hidden state ``state``. The logits        could be a tuple instead of a ``torch.Tensor``. It depends on how the         policy process the network output. For example, in PPO, the return of         the network might be ``(mu, sigma), state`` for Gaussian policy.

    Since :class:`~tianshou.policy.BasePolicy` inherits ``torch.nn.Module``,
    you can use :class:`~tianshou.policy.BasePolicy` almost the same as
    ``torch.nn.Module``, for instance, loading and saving the model:
    ::

        torch.save(policy.state_dict(), 'policy.pth')
        policy.load_state_dict(torch.load('policy.pth'))
    """

    def __init__(self, **kwargs) ->None:
        super().__init__()

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray) ->Batch:
        """Pre-process the data from the provided replay buffer. Check out
        :ref:`policy_concept` for more information.
        """
        return batch

    @abstractmethod
    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, **kwargs) ->Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following        keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over                 given batch data.
            * ``state`` a dict, an numpy.ndarray or a torch.Tensor, the                 internal state of the policy, ``None`` as default.

        Other keys are user-defined. It depends on the algorithm. For example,
        ::

            # some code
            return Batch(logits=..., act=..., state=None, dist=...)

        After version >= 0.2.3, the keyword "policy" is reserverd and the
        corresponding data will be stored into the replay buffer in numpy. For
        instance,
        ::

            # some code
            return Batch(..., policy=Batch(log_prob=dist.log_prob(act)))
            # and in the sampled data batch, you can directly call
            # batch.policy.log_prob to get your data, although it is stored in
            # np.ndarray.
        """
        pass

    @abstractmethod
    def learn(self, batch: Batch, **kwargs) ->Dict[str, Union[float, List[float]]]:
        """Update policy with a given batch of data.

        :return: A dict which includes loss and its corresponding label.
        """
        pass

    @staticmethod
    def compute_episodic_return(batch: Batch, v_s_: Optional[Union[np.ndarray, torch.Tensor]]=None, gamma: float=0.99, gae_lambda: float=0.95) ->Batch:
        """Compute returns over given full-length episodes, including the
        implementation of Generalized Advantage Estimation (arXiv:1506.02438).

        :param batch: a data batch which contains several full-episode data
            chronologically.
        :type batch: :class:`~tianshou.data.Batch`
        :param v_s_: the value function of all next states :math:`V(s')`.
        :type v_s_: numpy.ndarray
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage
            Estimation, should be in [0, 1], defaults to 0.95.
        """
        if v_s_ is None:
            v_s_ = np.zeros_like(batch.rew)
        else:
            if not isinstance(v_s_, np.ndarray):
                v_s_ = np.array(v_s_, np.float)
            v_s_ = v_s_.reshape(batch.rew.shape)
        batch.returns = np.roll(v_s_, 1, axis=0)
        m = (1.0 - batch.done) * gamma
        delta = batch.rew + v_s_ * m - batch.returns
        m *= gae_lambda
        gae = 0.0
        for i in range(len(batch.rew) - 1, -1, -1):
            gae = delta[i] + m[i] * gae
            batch.returns[i] += gae
        return batch


class ImitationPolicy(BasePolicy):
    """Implementation of vanilla imitation learning (for continuous action space).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> a)
    :param torch.optim.Optimizer optim: for optimizing the model.
    :param str mode: indicate the imitation type ("continuous" or "discrete"
        action space), defaults to "continuous".

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer, mode: str='continuous', **kwargs) ->None:
        super().__init__()
        self.model = model
        self.optim = optim
        assert mode in ['continuous', 'discrete'], f'Mode {mode} is not in ["continuous", "discrete"]'
        self.mode = mode

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, **kwargs) ->Batch:
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        if self.mode == 'discrete':
            a = logits.max(dim=1)[1]
        else:
            a = logits
        return Batch(logits=logits, act=a, state=h)

    def learn(self, batch: Batch, **kwargs) ->Dict[str, float]:
        self.optim.zero_grad()
        if self.mode == 'continuous':
            a = self(batch).act
            a_ = torch.tensor(batch.act, dtype=torch.float, device=a.device)
            loss = F.mse_loss(a, a_)
        elif self.mode == 'discrete':
            a = self(batch).logits
            a_ = torch.tensor(batch.act, dtype=torch.long, device=a.device)
            loss = F.nll_loss(a, a_)
        loss.backward()
        self.optim.step()
        return {'loss': loss.item()}


class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor: torch.nn.Module, actor_optim: torch.optim.Optimizer, critic: torch.nn.Module, critic_optim: torch.optim.Optimizer, tau: float=0.005, gamma: float=0.99, exploration_noise: float=0.1, action_range: Optional[Tuple[float, float]]=None, reward_normalization: bool=False, ignore_done: bool=False, **kwargs) ->None:
        super().__init__(**kwargs)
        if actor is not None:
            self.actor, self.actor_old = actor, deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim = actor_optim
        if critic is not None:
            self.critic, self.critic_old = critic, deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim = critic_optim
        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        self._tau = tau
        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self._gamma = gamma
        assert 0 <= exploration_noise, 'noise should not be negative'
        self._eps = exploration_noise
        assert action_range is not None
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2
        self._action_scale = (action_range[1] - action_range[0]) / 2
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()

    def set_eps(self, eps: float) ->None:
        """Set the eps for exploration."""
        self._eps = eps

    def train(self) ->None:
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self) ->None:
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self) ->None:
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray) ->Batch:
        if self._rew_norm:
            bfr = buffer.rew[:min(len(buffer), 1000)]
            mean, std = bfr.mean(), bfr.std()
            if std > self.__eps:
                batch.rew = (batch.rew - mean) / std
        if self._rm_done:
            batch.done = batch.done * 0.0
        return batch

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, model: str='actor', input: str='obs', eps: Optional[float]=None, **kwargs) ->Batch:
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for exploration use.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        logits, h = model(obs, state=state, info=batch.info)
        logits += self._action_bias
        if eps is None:
            eps = self._eps
        if eps > 0:
            logits += torch.randn(size=logits.shape, device=logits.device) * eps
        logits = logits.clamp(self._range[0], self._range[1])
        return Batch(act=logits, state=h)

    def learn(self, batch: Batch, **kwargs) ->Dict[str, float]:
        with torch.no_grad():
            target_q = self.critic_old(batch.obs_next, self(batch, model='actor_old', input='obs_next', eps=0).act)
            dev = target_q.device
            rew = torch.tensor(batch.rew, dtype=torch.float, device=dev)[:, (None)]
            done = torch.tensor(batch.done, dtype=torch.float, device=dev)[:, (None)]
            target_q = rew + (1.0 - done) * self._gamma * target_q
        current_q = self.critic(batch.obs, batch.act)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(batch.obs, self(batch, eps=0).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {'loss/actor': actor_loss.item(), 'loss/critic': critic_loss.item()}


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer implementation.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param str mode: defaults to ``weight``.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for more
        detailed explanation.
    """

    def __init__(self, size: int, alpha: float, beta: float, mode: str='weight', **kwargs) ->None:
        if mode != 'weight':
            raise NotImplementedError
        super().__init__(size, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._weight_sum = 0.0
        self.weight = np.zeros(size, dtype=np.float64)
        self._amortization_freq = 50
        self._amortization_counter = 0

    def add(self, obs: Union[dict, np.ndarray], act: Union[np.ndarray, float], rew: float, done: bool, obs_next: Optional[Union[dict, np.ndarray]]=None, info: dict={}, policy: Optional[Union[dict, Batch]]={}, weight: float=1.0, **kwargs) ->None:
        """Add a batch of data into replay buffer."""
        self._weight_sum += np.abs(weight) ** self._alpha - self.weight[self._index]
        self._add_to_buffer('weight', np.abs(weight) ** self._alpha)
        super().add(obs, act, rew, done, obs_next, info, policy)
        self._check_weight_sum()

    def sample(self, batch_size: int, importance_sample: bool=True) ->Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with priority probability.         Return all the data in the buffer if batch_size is ``0``.

        :return: Sample data and its corresponding index inside the buffer.
        """
        if batch_size > 0 and batch_size <= self._size:
            indice = np.random.choice(self._size, batch_size, p=(self.weight / self.weight.sum())[:self._size], replace=False)
        elif batch_size == 0:
            indice = np.concatenate([np.arange(self._index, self._size), np.arange(0, self._index)])
        else:
            raise ValueError('batch_size should be less than len(self)')
        batch = self[indice]
        if importance_sample:
            impt_weight = Batch(impt_weight=1 / np.power(self._size * (batch.weight / self._weight_sum), self._beta))
            batch.append(impt_weight)
        self._check_weight_sum()
        return batch, indice

    def reset(self) ->None:
        self._amortization_counter = 0
        super().reset()

    def update_weight(self, indice: Union[slice, np.ndarray], new_weight: np.ndarray) ->None:
        """Update priority weight by indice in this buffer.

        :param np.ndarray indice: indice you want to update weight
        :param np.ndarray new_weight: new priority weight you wangt to update
        """
        self._weight_sum += np.power(np.abs(new_weight), self._alpha).sum() - self.weight[indice].sum()
        self.weight[indice] = np.power(np.abs(new_weight), self._alpha)

    def __getitem__(self, index: Union[slice, np.ndarray]) ->Batch:
        return Batch(obs=self.get(index, 'obs'), act=self.act[index], rew=self.rew[index], done=self.done[index], obs_next=self.get(index, 'obs_next'), info=self.info[index], weight=self.weight[index], policy=self.get(index, 'policy'))

    def _check_weight_sum(self) ->None:
        self._amortization_counter += 1
        if self._amortization_counter % self._amortization_freq == 0:
            self._weight_sum = np.sum(self.weight)
            self._amortization_counter = 0


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (``0``
        if you do not use the target network).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer, discount_factor: float=0.99, estimation_step: int=1, target_update_freq: Optional[int]=0, **kwargs) ->None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0
        assert 0 <= discount_factor <= 1, 'discount_factor should in [0, 1]'
        self._gamma = discount_factor
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._cnt = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()

    def set_eps(self, eps: float) ->None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self) ->None:
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.model.train()

    def eval(self) ->None:
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.model.eval()

    def sync_weight(self) ->None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray) ->Batch:
        """Compute the n-step return for Q-learning targets:

        .. math::
            G_t = \\sum_{i = t}^{t + n - 1} \\gamma^{i - t}(1 - d_i)r_i +
            \\gamma^n (1 - d_{t + n}) \\max_a Q_{old}(s_{t + n}, \\arg\\max_a
            (Q_{new}(s_{t + n}, a)))

        , where :math:`\\gamma` is the discount factor,
        :math:`\\gamma \\in [0, 1]`, :math:`d_t` is the done flag of step
        :math:`t`. If there is no target network, the :math:`Q_{old}` is equal
        to :math:`Q_{new}`.
        """
        returns = np.zeros_like(indice)
        gammas = np.zeros_like(indice) + self._n_step
        for n in range(self._n_step - 1, -1, -1):
            now = (indice + n) % len(buffer)
            gammas[buffer.done[now] > 0] = n
            returns[buffer.done[now] > 0] = 0
            returns = buffer.rew[now] + self._gamma * returns
        terminal = (indice + self._n_step - 1) % len(buffer)
        terminal_data = buffer[terminal]
        if self._target:
            a = self(terminal_data, input='obs_next', eps=0).act
            target_q = self(terminal_data, model='model_old', input='obs_next').logits
            if isinstance(target_q, torch.Tensor):
                target_q = target_q.detach().cpu().numpy()
            target_q = target_q[np.arange(len(a)), a]
        else:
            target_q = self(terminal_data, input='obs_next').logits
            if isinstance(target_q, torch.Tensor):
                target_q = target_q.detach().cpu().numpy()
            target_q = target_q.max(axis=1)
        target_q[gammas != self._n_step] = 0
        returns += self._gamma ** gammas * target_q
        batch.returns = returns
        if isinstance(buffer, PrioritizedReplayBuffer):
            q = self(batch).logits
            q = q[np.arange(len(q)), batch.act]
            r = batch.returns
            if isinstance(r, np.ndarray):
                r = torch.tensor(r, device=q.device, dtype=q.dtype)
            td = r - q
            buffer.update_weight(indice, td.detach().cpu().numpy())
            impt_weight = torch.tensor(batch.impt_weight, device=q.device, dtype=torch.float)
            loss = (td.pow(2) * impt_weight).mean()
            if not hasattr(batch, 'loss'):
                batch.loss = loss
            else:
                batch.loss += loss
        return batch

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, model: str='model', input: str='obs', eps: Optional[float]=None, **kwargs) ->Batch:
        """Compute action over the given batch data.

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        q, h = model(obs, state=state, info=batch.info)
        act = q.max(dim=1)[1].detach().cpu().numpy()
        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0):
            for i in range(len(q)):
                if np.random.rand() < eps:
                    act[i] = np.random.randint(q.shape[1])
        return Batch(logits=q, act=act, state=h)

    def learn(self, batch: Batch, **kwargs) ->Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        if hasattr(batch, 'loss'):
            loss = batch.loss
        else:
            q = self(batch).logits
            q = q[np.arange(len(q)), batch.act]
            r = batch.returns
            if isinstance(r, np.ndarray):
                r = torch.tensor(r, device=q.device, dtype=q.dtype)
            loss = F.mse_loss(q, r)
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {'loss': loss.item()}


class PGPolicy(BasePolicy):
    """Implementation of Vanilla Policy Gradient.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1].

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer, dist_fn: torch.distributions.Distribution=torch.distributions.Categorical, discount_factor: float=0.99, reward_normalization: bool=False, **kwargs) ->None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0 <= discount_factor <= 1, 'discount factor should in [0, 1]'
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray) ->Batch:
        """Compute the discounted returns for each frame:

        .. math::
            G_t = \\sum_{i=t}^T \\gamma^{i-t}r_i

        , where :math:`T` is the terminal time step, :math:`\\gamma` is the
        discount factor, :math:`\\gamma \\in [0, 1]`.
        """
        return self.compute_episodic_return(batch, gamma=self._gamma, gae_lambda=1.0)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, **kwargs) ->Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs) ->Dict[str, List[float]]:
        losses = []
        r = batch.returns
        if self._rew_norm and r.std() > self.__eps:
            batch.returns = (r - r.mean()) / r.std()
        for _ in range(repeat):
            for b in batch.split(batch_size):
                self.optim.zero_grad()
                dist = self(b).dist
                a = torch.tensor(b.act, device=dist.logits.device)
                r = torch.tensor(b.returns, device=dist.logits.device)
                loss = -(dist.log_prob(a) * r).sum()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
        return {'loss': losses}


class PPOPolicy(PGPolicy):
    """Implementation of Proximal Policy Optimization. arXiv:1707.06347

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic
        network.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1], defaults to 0.99.
    :param float max_grad_norm: clipping gradients in back propagation,
        defaults to ``None``.
    :param float eps_clip: :math:`\\epsilon` in :math:`L_{CLIP}` in the original
        paper, defaults to 0.2.
    :param float vf_coef: weight for value loss, defaults to 0.5.
    :param float ent_coef: weight for entropy loss, defaults to 0.01.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation, defaults to 0.95.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound,
        defaults to 5.0 (set ``None`` if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1,
        defaults to ``True``.
    :param bool reward_normalization: normalize the returns to Normal(0, 1),
        defaults to ``True``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor: torch.nn.Module, critic: torch.nn.Module, optim: torch.optim.Optimizer, dist_fn: torch.distributions.Distribution, discount_factor: float=0.99, max_grad_norm: Optional[float]=None, eps_clip: float=0.2, vf_coef: float=0.5, ent_coef: float=0.01, action_range: Optional[Tuple[float, float]]=None, gae_lambda: float=0.95, dual_clip: float=None, value_clip: bool=True, reward_normalization: bool=True, **kwargs) ->None:
        super().__init__(None, None, dist_fn, discount_factor, **kwargs)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = action_range
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self._batch = 64
        assert 0 <= gae_lambda <= 1, 'GAE lambda should be in [0, 1].'
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1, 'Dual-clip PPO parameter should greater than 1.'
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        self.__eps = np.finfo(np.float32).eps.item()

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray) ->Batch:
        if self._rew_norm:
            mean, std = batch.rew.mean(), batch.rew.std()
            if std > self.__eps:
                batch.rew = (batch.rew - mean) / std
        if self._lambda in [0, 1]:
            return self.compute_episodic_return(batch, None, gamma=self._gamma, gae_lambda=self._lambda)
        v_ = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False):
                v_.append(self.critic(b.obs_next))
        v_ = torch.cat(v_, dim=0).cpu().numpy()
        return self.compute_episodic_return(batch, v_, gamma=self._gamma, gae_lambda=self._lambda)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, **kwargs) ->Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, h = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs) ->Dict[str, List[float]]:
        self._batch = batch_size
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        v = []
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(batch_size, shuffle=False):
                v.append(self.critic(b.obs))
                old_log_prob.append(self(b).dist.log_prob(torch.tensor(b.act, device=v[0].device)))
        batch.v = torch.cat(v, dim=0)
        dev = batch.v.device
        batch.act = torch.tensor(batch.act, dtype=torch.float, device=dev)
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.returns = torch.tensor(batch.returns, dtype=torch.float, device=dev).reshape(batch.v.shape)
        if self._rew_norm:
            mean, std = batch.returns.mean(), batch.returns.std()
            if std > self.__eps:
                batch.returns = (batch.returns - mean) / std
        batch.adv = batch.returns - batch.v
        if self._rew_norm:
            mean, std = batch.adv.mean(), batch.adv.std()
            if std > self.__eps:
                batch.adv = (batch.adv - mean) / std
        for _ in range(repeat):
            for b in batch.split(batch_size):
                dist = self(b).dist
                value = self.critic(b.obs)
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip_loss = -torch.max(torch.min(surr1, surr2), self._dual_clip * b.adv).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.item())
                if self._value_clip:
                    v_clip = b.v + (value - b.v).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = 0.5 * torch.max(vf1, vf2).mean()
                else:
                    vf_loss = 0.5 * (b.returns - value).pow(2).mean()
                vf_losses.append(vf_loss.item())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                loss = clip_loss + self._w_vf * vf_loss - self._w_ent * e_loss
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self._max_grad_norm)
                self.optim.step()
        return {'loss': losses, 'loss/clip': clip_losses, 'loss/vf': vf_losses, 'loss/ent': ent_losses}


class DiagGaussian(torch.distributions.Normal):
    """Diagonal Gaussian Distribution

    """

    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param float alpha: entropy regularization coefficient, default to 0.2.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor: torch.nn.Module, actor_optim: torch.optim.Optimizer, critic1: torch.nn.Module, critic1_optim: torch.optim.Optimizer, critic2: torch.nn.Module, critic2_optim: torch.optim.Optimizer, tau: float=0.005, gamma: float=0.99, alpha: float=0.2, action_range: Optional[Tuple[float, float]]=None, reward_normalization: bool=False, ignore_done: bool=False, **kwargs) ->None:
        super().__init__(None, None, None, None, tau, gamma, 0, action_range, reward_normalization, ignore_done, **kwargs)
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()

    def train(self) ->None:
        self.training = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) ->None:
        self.training = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def sync_weight(self) ->None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]]=None, input: str='obs', **kwargs) ->Batch:
        obs = getattr(batch, input)
        logits, h = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = DiagGaussian(*logits)
        x = dist.rsample()
        y = torch.tanh(x)
        act = y * self._action_scale + self._action_bias
        log_prob = dist.log_prob(x) - torch.log(self._action_scale * (1 - y.pow(2)) + self.__eps)
        act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, dist=dist, log_prob=log_prob)

    def learn(self, batch: Batch, **kwargs) ->Dict[str, float]:
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            dev = a_.device
            batch.act = torch.tensor(batch.act, dtype=torch.float, device=dev)
            target_q = torch.min(self.critic1_old(batch.obs_next, a_), self.critic2_old(batch.obs_next, a_)) - self._alpha * obs_next_result.log_prob
            rew = torch.tensor(batch.rew, dtype=torch.float, device=dev)[:, (None)]
            done = torch.tensor(batch.done, dtype=torch.float, device=dev)[:, (None)]
            target_q = rew + (1.0 - done) * self._gamma * target_q
        current_q1 = self.critic1(batch.obs, batch.act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        current_q2 = self.critic2(batch.obs, batch.act)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        obs_result = self(batch)
        a = obs_result.act
        current_q1a = self.critic1(batch.obs, a)
        current_q2a = self.critic2(batch.obs, a)
        actor_loss = (self._alpha * obs_result.log_prob - torch.min(current_q1a, current_q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {'loss/actor': actor_loss.item(), 'loss/critic1': critic1_loss.item(), 'loss/critic2': critic2_loss.item()}


class TD3Policy(DDPGPolicy):
    """Implementation of Twin Delayed Deep Deterministic Policy Gradient,
    arXiv:1802.09477

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param float policy_noise: the noise used in updating policy network,
        default to 0.2.
    :param int update_actor_freq: the update frequency of actor network,
        default to 2.
    :param float noise_clip: the clipping range used in updating policy
        network, default to 0.5.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor: torch.nn.Module, actor_optim: torch.optim.Optimizer, critic1: torch.nn.Module, critic1_optim: torch.optim.Optimizer, critic2: torch.nn.Module, critic2_optim: torch.optim.Optimizer, tau: float=0.005, gamma: float=0.99, exploration_noise: float=0.1, policy_noise: float=0.2, update_actor_freq: int=2, noise_clip: float=0.5, action_range: Optional[Tuple[float, float]]=None, reward_normalization: bool=False, ignore_done: bool=False, **kwargs) ->None:
        super().__init__(actor, actor_optim, None, None, tau, gamma, exploration_noise, action_range, reward_normalization, ignore_done, **kwargs)
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def train(self) ->None:
        self.training = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) ->None:
        self.training = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def sync_weight(self) ->None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def learn(self, batch: Batch, **kwargs) ->Dict[str, float]:
        with torch.no_grad():
            a_ = self(batch, model='actor_old', input='obs_next').act
            dev = a_.device
            noise = torch.randn(size=a_.shape, device=dev) * self._policy_noise
            if self._noise_clip >= 0:
                noise = noise.clamp(-self._noise_clip, self._noise_clip)
            a_ += noise
            a_ = a_.clamp(self._range[0], self._range[1])
            target_q = torch.min(self.critic1_old(batch.obs_next, a_), self.critic2_old(batch.obs_next, a_))
            rew = torch.tensor(batch.rew, dtype=torch.float, device=dev)[:, (None)]
            done = torch.tensor(batch.done, dtype=torch.float, device=dev)[:, (None)]
            target_q = rew + (1.0 - done) * self._gamma * target_q
        current_q1 = self.critic1(batch.obs, batch.act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        current_q2 = self.critic2(batch.obs, batch.act)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        if self._cnt % self._freq == 0:
            actor_loss = -self.critic1(batch.obs, self(batch, eps=0).act).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1
        return {'loss/actor': self._last, 'loss/critic1': critic1_loss.item(), 'loss/critic2': critic2_loss.item()}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActorProb,
     lambda: ([], {'layer_num': 1, 'state_shape': 4, 'action_shape': 4, 'max_action': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Net,
     lambda: ([], {'layer_num': 1, 'state_shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Recurrent,
     lambda: ([], {'layer_num': 1, 'state_shape': 4, 'action_shape': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RecurrentActorProb,
     lambda: ([], {'layer_num': 1, 'state_shape': 4, 'action_shape': 4, 'max_action': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RecurrentCritic,
     lambda: ([], {'layer_num': 1, 'state_shape': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_thu_ml_tianshou(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

