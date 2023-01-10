import sys
_module = sys.modules[__name__]
del sys
MiniImagenet = _module
main = _module
meta = _module
naive = _module
omniglot = _module
omniglotNShot = _module

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


from torch.utils.data import Dataset


from torchvision.transforms import transforms


import numpy as np


import collections


import random


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torch import nn


from torch import optim


from torch import autograd


from torch.nn import functional as F


import torch.utils.data as data


import torchvision.transforms as transforms


class Learner(nn.Module):
    """
	This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
	process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
	for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
	by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
	then backprop on theta network, which should be done on metalaerner class.
	For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
	meta-test set.
	"""

    def __init__(self, net_cls, *args):
        """
		It will receive a class: net_cls and its parameters: args for net_cls.
		:param net_cls: class, not instance
		:param args: the parameters for net_cls
		"""
        super(Learner, self).__init__()
        assert net_cls.__class__ == type
        self.net = net_cls(*args)
        self.net_pi = net_cls(*args)
        self.optimizer = optim.SGD(self.net_pi.parameters(), 0.1)

    def parameters(self):
        """
		Override this function to return only net parameters for MetaLearner's optimize
		it will ignore theta_pi network parameters.
		:return:
		"""
        return self.net.parameters()

    def update_pi(self):
        """
		copy parameters from self.net -> self.net_pi
		:return:
		"""
        for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, support_x, support_y, query_x, query_y, num_updates):
        """
		learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
		:param support_x: [setsz, c_, h, w]
		:param support_y: [setsz]
		:param query_x:   [querysz, c_, h, w]
		:param query_y:   [querysz]
		:param num_updates: 5
		:return:
		"""
        self.update_pi()
        for i in range(num_updates):
            loss, pred = self.net_pi(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        loss, pred = self.net_pi(query_x, query_y)
        _, indices = torch.max(pred, dim=1)
        correct = torch.eq(indices, query_y).sum().data[0]
        acc = correct / query_y.size(0)
        grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)
        return loss, grads_pi, acc

    def net_forward(self, support_x, support_y):
        """
		This function is purely for updating net network. In metalearner, we need the get the loss op from net network
		to write our merged gradients into net network, hence will call this function to get a dummy loss op.
		:param support_x: [setsz, c, h, w]
		:param support_y: [sessz, c, h, w]
		:return: dummy loss and dummy pred
		"""
        loss, pred = self.net(support_x, support_y)
        return loss, pred


class MetaLearner(nn.Module):
    """
	As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
	on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
	network to update theta parameters, which is the initialization point we want to find.
	"""

    def __init__(self, net_cls, net_cls_args, n_way, k_shot, meta_batchsz, beta, num_updates):
        """

		:param net_cls: class, not instance. the class of specific Network for learner
		:param net_cls_args: tuple, args for net_cls, like (n_way, imgsz)
		:param n_way:
		:param k_shot:
		:param meta_batchsz: number of tasks/episode
		:param beta: learning rate for meta-learner
		:param num_updates: number of updates for learner
		"""
        super(MetaLearner, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_batchsz = meta_batchsz
        self.beta = beta
        self.num_updates = num_updates
        self.learner = Learner(net_cls, *net_cls_args)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)

    def write_grads(self, dummy_loss, sum_grads_pi):
        """
		write loss into learner.net, gradients come from sum_grads_pi.
		Since the gradients info is not calculated by general backward, we need this function to write the right gradients
		into theta network and update theta parameters as wished.
		:param dummy_loss: dummy loss, nothing but to write our gradients by hook
		:param sum_grads_pi: the summed gradients
		:return:
		"""
        hooks = []
        for i, v in enumerate(self.learner.parameters()):

            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]
            hooks.append(v.register_hook(closure()))
        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()
        for h in hooks:
            h.remove()

    def forward(self, support_x, support_y, query_x, query_y):
        """
		Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
		we gather the loss and sum all the loss and then update theta network.
		setsz = n_way * k_shotf
		querysz = n_way * k_shot
		:param support_x: [meta_batchsz, setsz, c_, h, w]
		:param support_y: [meta_batchsz, setsz]
		:param query_x:   [meta_batchsz, querysz, c_, h, w]
		:param query_y:   [meta_batchsz, querysz]
		:return:
		"""
        sum_grads_pi = None
        meta_batchsz = support_y.size(0)
        accs = []
        for i in range(meta_batchsz):
            _, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:
                sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)
        return accs

    def pred(self, support_x, support_y, query_x, query_y):
        """
		predict for query_x
		:param support_x:
		:param support_y:
		:param query_x:
		:param query_y:
		:return:
		"""
        meta_batchsz = support_y.size(0)
        accs = []
        for i in range(meta_batchsz):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
        return np.array(accs).mean()


class Naive(nn.Module):
    """
	Define your network here.
	"""

    def __init__(self, n_way, imgsz):
        super(Naive, self).__init__()
        if imgsz > 28:
            self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.AvgPool2d(kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3), nn.AvgPool2d(kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        else:
            self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3), nn.AvgPool2d(kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3), nn.AvgPool2d(kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        dummy_img = Variable(torch.randn(2, 3, imgsz, imgsz))
        repsz = self.net(dummy_img).size()
        _, c, h, w = repsz
        self.fc_dim = c * h * w
        self.fc = nn.Sequential(nn.Linear(self.fc_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, n_way))
        self.criteon = nn.CrossEntropyLoss()
        None
        None

    def forward(self, x, target):
        x = self.net(x)
        x = x.view(-1, self.fc_dim)
        pred = self.fc(x)
        loss = self.criteon(pred, target)
        return loss, pred

