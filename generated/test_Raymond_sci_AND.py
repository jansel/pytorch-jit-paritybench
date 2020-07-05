import sys
_module = sys.modules[__name__]
del sys
datasets = _module
cifar = _module
svhn = _module
lib = _module
ans_discovery = _module
criterion = _module
non_parametric_classifier = _module
normalize = _module
protocols = _module
utils = _module
main = _module
models = _module
resnet_cifar = _module
packages = _module
config = _module
transforms = _module
loggers = _module
std_logger = _module
tf_logger = _module
lr_policy = _module
fixed = _module
multistep = _module
step = _module
networks = _module
optimizers = _module
adam = _module
rmsprop = _module
sgd = _module
register = _module
session = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Function


from torch import nn


import math


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


import time


_global_config['ANs_size'] = 4


_global_config['device'] = 0


_global_config['ANs_select_rate'] = 4


class ANsDiscovery(nn.Module):
    """Discovery ANs
    
    Discovery ANs according to current round, select_rate and most importantly,
    all sample's corresponding entropy
    """

    def __init__(self, nsamples):
        """Object used to discovery ANs
        
        Discovery ANs according to the total amount of samples, ANs selection
        rate, ANs size
        
        Arguments:
            nsamples {int} -- total number of sampels
            select_rate {float} -- ANs selection rate
            ans_size {int} -- ANs size
        
        Keyword Arguments:
            device {str} -- [description] (default: {'cpu'})
        """
        super(ANsDiscovery, self).__init__()
        self.select_rate = cfg.ANs_select_rate
        self.ANs_size = cfg.ANs_size
        self.register_buffer('samples_num', torch.tensor(nsamples))
        self.register_buffer('anchor_indexes', torch.LongTensor([]))
        self.register_buffer('instance_indexes', torch.arange(nsamples).long())
        self.register_buffer('position', -1 * torch.arange(nsamples).long() - 1)
        self.register_buffer('neighbours', torch.LongTensor([]))
        self.register_buffer('entropy', torch.FloatTensor(nsamples))
        self.register_buffer('consistency', torch.tensor(0.0))

    def get_ANs_num(self, round):
        """Get number of ANs
        
        Get number of ANs at target round according to the select rate
        
        Arguments:
            round {int} -- target round
        
        Returns:
            int -- number of ANs
        """
        return int(self.samples_num.float() * self.select_rate * round)

    def update(self, round, npc, cheat_labels=None):
        """Update ANs
        
        Discovery new ANs and update `anchor_indexes`, `instance_indexes` and
        `neighbours`
        
        Arguments:
            round {int} -- target round
            npc {Module} -- non-parametric classifier
            cheat_labels {list} -- used to compute consistency of chosen ANs only
        
        Returns:
            number -- [updated consistency]
        """
        with torch.no_grad():
            batch_size = 100
            ANs_num = self.get_ANs_num(round)
            logger.debug('Going to choose %d samples as anchors' % ANs_num)
            features = npc.memory
            logger.debug("Start to compute each sample's entropy")
            for start in xrange(0, self.samples_num, batch_size):
                logger.progress(start, self.samples_num, 'processing %d/%d samples...')
                end = start + batch_size
                end = min(end, self.samples_num)
                preds = F.softmax(npc(features[start:end], None), 1)
                self.entropy[start:end] = -(preds * preds.log()).sum(1)
            logger.debug('Compute entropy done, max(%.2f), min(%.2f), mean(%.2f)' % (self.entropy.max(), self.entropy.min(), self.entropy.mean()))
            self.anchor_indexes = self.entropy.topk(ANs_num, largest=False)[1]
            self.instance_indexes = torch.ones_like(self.position).scatter_(0, self.anchor_indexes, 0).nonzero().view(-1)
            anchor_entropy = self.entropy.index_select(0, self.anchor_indexes)
            instance_entropy = self.entropy.index_select(0, self.instance_indexes)
            if self.anchor_indexes.size(0) > 0:
                logger.debug('Entropies of anchor samples: max(%.2f), min(%.2f), mean(%.2f)' % (anchor_entropy.max(), anchor_entropy.min(), anchor_entropy.mean()))
            if self.instance_indexes.size(0) > 0:
                logger.debug('Entropies of instance sample: max(%.2f), min(%.2f), mean(%.2f)' % (instance_entropy.max(), instance_entropy.min(), instance_entropy.mean()))
            logger.debug('Start to get the position of both anchor and instance samples')
            instance_cnt = 0
            for i in xrange(self.samples_num):
                logger.progress(i, self.samples_num, 'processing %d/%d samples...')
                if (i == self.anchor_indexes).any():
                    self.position[i] = (self.anchor_indexes == i).max(0)[1]
                    continue
                instance_cnt -= 1
                self.position[i] = instance_cnt
            logger.debug('Start to find %d neighbours for each anchor sample' % self.ANs_size)
            anchor_features = features.index_select(0, self.anchor_indexes)
            self.neighbours = torch.LongTensor(ANs_num, self.ANs_size)
            for start in xrange(0, ANs_num, batch_size):
                logger.progress(start, ANs_num, 'processing %d/%d samples...')
                end = start + batch_size
                end = min(end, ANs_num)
                sims = torch.mm(anchor_features[start:end], features.t())
                sims.scatter_(1, self.anchor_indexes[start:end].view(-1, 1), -1.0)
                _, self.neighbours[start:end] = sims.topk(self.ANs_size, largest=True, dim=1)
            logger.debug('ANs discovery done')
            if cheat_labels is None:
                return 0.0
            logger.debug('Start to compute ANs consistency')
            anchor_label = cheat_labels.index_select(0, self.anchor_indexes)
            neighbour_label = cheat_labels.index_select(0, self.neighbours.view(-1)).view_as(self.neighbours)
            self.consistency = (anchor_label.view(-1, 1) == neighbour_label).float().mean()
            return self.consistency


class Criterion(nn.Module):

    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, x, y, ANs):
        batch_size, _ = x.shape
        anchor_indexes, instance_indexes = self.__split(y, ANs)
        preds = F.softmax(x, 1)
        l_ans = 0.0
        if anchor_indexes.size(0) > 0:
            y_ans = y.index_select(0, anchor_indexes)
            y_ans_neighbour = ANs.position.index_select(0, y_ans)
            neighbours = ANs.neighbours.index_select(0, y_ans_neighbour)
            x_ans = preds.index_select(0, anchor_indexes)
            x_ans_neighbour = x_ans.gather(1, neighbours).sum(1)
            x_ans = x_ans.gather(1, y_ans.view(-1, 1)).view(-1) + x_ans_neighbour
            l_ans = -1 * torch.log(x_ans).sum(0)
        l_inst = 0.0
        if instance_indexes.size(0) > 0:
            y_inst = y.index_select(0, instance_indexes)
            x_inst = preds.index_select(0, instance_indexes)
            x_inst = x_inst.gather(1, y_inst.view(-1, 1))
            l_inst = -1 * torch.log(x_inst).sum(0)
        return (l_inst + l_ans) / batch_size

    def __split(self, y, ANs):
        pos = ANs.position.index_select(0, y.view(-1))
        return (pos >= 0).nonzero().view(-1), (pos < 0).nonzero().view(-1)


class NonParametricClassifierOP(Function):

    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)
        out = torch.mm(x.data, memory.t())
        out.div_(T)
        self.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        gradOutput.data.div_(T)
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        return gradInput, None, None, None, None


_global_config['npc_temperature'] = 4


_global_config['npc_momentum'] = 4


class NonParametricClassifier(nn.Module):
    """Non-parametric Classifier
    
    Non-parametric Classifier from
    "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"
    
    Extends:
        nn.Module
    """

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        """Non-parametric Classifier initial functin
        
        Initial function for non-parametric classifier
        
        Arguments:
            inputSize {int} -- in-channels dims
            outputSize {int} -- out-channels dims
        
        Keyword Arguments:
            T {int} -- distribution temperate (default: {0.05})
            momentum {int} -- memory update momentum (default: {0.5})
        """
        super(NonParametricClassifier, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize
        self.register_buffer('params', torch.tensor([cfg.npc_temperature, cfg.npc_momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out


class Normalize(nn.Module):
    """Normalize module
    
    Module used to normalize matrix
    
    Extends:
        nn.Module
    """

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, low_dim)
        self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.l2norm(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NonParametricClassifier,
     lambda: ([], {'inputSize': 4, 'outputSize': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Raymond_sci_AND(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

