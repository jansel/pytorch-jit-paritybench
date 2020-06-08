import sys
_module = sys.modules[__name__]
del sys
MiniImagenet = _module
csmlv0 = _module
mainv0 = _module
naive5_train = _module
learner = _module
meta = _module
miniimagenet_train = _module
omniglot = _module
omniglotNShot = _module
omniglot_train = _module
test = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


from torch import optim


from torch import autograd


from torch import multiprocessing


from torch.autograd import Variable


from torch.nn import functional as F


from torch.utils.data import TensorDataset


import numpy as np


from torch.utils.data import DataLoader


from copy import deepcopy


class Concept(nn.Module):

    def __init__(self):
        super(Concept, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1), nn.ReLU(inplace=True), nn.
            MaxPool2d(kernel_size=2), nn.Conv2d(64, 64, kernel_size=3,
            padding=0), nn.BatchNorm2d(64, momentum=1), nn.ReLU(inplace=
            True), nn.MaxPool2d(kernel_size=2), nn.Conv2d(64, 64,
            kernel_size=3, padding=1), nn.BatchNorm2d(64, momentum=1), nn.
            ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1), nn.ReLU(inplace=True))

    def load(self, src):
        """
        Load parameters from central pool
        :param src:
        :return:
        """
        self.load_state_dict(src.state_dict())

    def forward(self, x):
        x = self.net(x)
        return x


class Relation(nn.Module):

    def __init__(self):
        super(Relation, self).__init__()
        self.g = nn.Sequential(nn.Linear(2 * (64 + 2), 256), nn.ReLU(
            inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True), nn.
            Linear(256, 256), nn.ReLU(inplace=True))
        self.f = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256),
            nn.ReLU(inplace=True))

    def forward(self, x):
        pass


class OutLayer(nn.Module):

    def __init__(self):
        super(OutLayer, self).__init__()
        self.net = nn.Sequential(nn.Linear(64 * 3 * 3, 5))

    def forward(self, x):
        x = F.avg_pool2d(x, 5, 5)
        x = x.view(x.size(0), -1)
        return self.net(x)


class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            elif name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name is 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]),
                    requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]),
                    requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d',
                'max_pool2d', 'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''
        for name, param in self.config:
            if name is 'conv2d':
                tmp = (
                    'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'
                     % (param[1], param[0], param[2], param[3], param[4],
                    param[5]))
                info += tmp + '\n'
            elif name is 'convt2d':
                tmp = (
                    'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'
                     % (param[0], param[1], param[2], param[3], param[4],
                    param[5]))
                info += tmp + '\n'
            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % param[0]
                info += tmp + '\n'
            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0
                    ], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0
                    ], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape',
                'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError
        return info

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0
        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=
                    param[5])
                idx += 2
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[
                    bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w,
                    bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1.0 / 2)
        clip_coef = max_norm / (total_norm + 1e-06)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)
        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses_q = [(0) for _ in range(self.update_step + 1)]
        corrects = [(0) for _ in range(self.update_step + 1)]
        for i in range(task_num):
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                zip(grad, self.net.parameters())))
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(),
                    bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p
                    [0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct
        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4
        querysz = x_qry.size(0)
        corrects = [(0) for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net)
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip
            (grad, net.parameters())))
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
        for k in range(1, self.update_step_test):
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                zip(grad, fast_weights)))
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k + 1] = corrects[k + 1] + correct
        del net
        accs = np.array(corrects) / querysz
        return accs


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.a = nn.ParameterList([nn.Parameter(torch.zeros(3, 4))])
        b = [torch.ones(2, 3), torch.ones(2, 3)]
        for i in range(2):
            self.register_buffer('b%d' % i, b[i])

    def forward(self, input):
        return self.a[0]


class MAML(nn.Module):

    def __init__(self):
        super(MAML, self).__init__()
        self.net = Net()

    def forward(self, input):
        return self.net(input)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_dragen1860_MAML_Pytorch(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(Concept(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(Relation(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_002(self):
        self._check(Learner(*[], **{'config': _mock_config(), 'imgc': 4, 'imgsz': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_003(self):
        self._check(Net(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_004(self):
        self._check(MAML(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
