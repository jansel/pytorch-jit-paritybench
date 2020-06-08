import sys
_module = sys.modules[__name__]
del sys
bootstrap = _module
__version__ = _module
compare = _module
datasets = _module
dataset = _module
factory = _module
transforms = _module
utils = _module
engines = _module
engine = _module
logger = _module
lib = _module
options = _module
overwrite_print = _module
models = _module
criterions = _module
bce = _module
cross_entropy = _module
nll = _module
metrics = _module
accuracy = _module
model = _module
networks = _module
data_parallel = _module
new = _module
optimizers = _module
grad_clipper = _module
lr_scheduler = _module
run = _module
setup = _module
views = _module
generate = _module
plotly = _module
tensorboard = _module
conf = _module
test_new = _module
test_options = _module

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


import torch.nn as nn


import torch


from torch.nn.parallel._functions import Gather


from torch.nn.utils.clip_grad import clip_grad_norm


class BCEWithLogitsLoss(nn.Module):

    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out.squeeze(1), batch['class_id'].float
            ().squeeze(1))
        return out


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out, batch['class_id'].view(-1))
        return out


class NLLLoss(nn.Module):

    def __init__(self):
        super(NLLLoss, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out, batch['class_id'].squeeze(1))
        return out


def accuracy(output, target, topk=None, ignore_index=None):
    """Computes the precision@k for the specified values of k"""
    topk = topk or [1, 5]
    if ignore_index is not None:
        target_mask = target != ignore_index
        target = target[target_mask]
        output_mask = target_mask.unsqueeze(1)
        output_mask = output_mask.expand_as(output)
        output = output[output_mask]
        output = output.view(-1, output_mask.size(1))
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size)[0])
    return res


class Accuracy(nn.Module):

    def __init__(self, topk=None):
        super(Accuracy, self).__init__()
        self.topk = topk or [1, 5]

    def __call__(self, cri_out, net_out, batch):
        out = {}
        acc_out = accuracy(net_out.data.cpu(), batch['class_id'].data.cpu(),
            topk=self.topk)
        for i, k in enumerate(self.topk):
            out['accuracy_top{}'.format(k)] = acc_out[i]
        return out


class DataParallel(nn.DataParallel):

    def __getattr__(self, key):
        try:
            return super(DataParallel, self).__getattr__(key)
        except AttributeError:
            return self.module.__getattribute__(key)

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __getattr__(self, key):
        try:
            return super(DistributedDataParallel, self).__getattr__(key)
        except AttributeError:
            return self.module.__getattribute__(key)

    def state_dict(self, *args, **kwgs):
        return self.module.state_dict(*args, **kwgs)

    def load_state_dict(self, *args, **kwgs):
        self.module.load_state_dict(*args, **kwgs)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Cadene_bootstrap_pytorch(_paritybench_base):
    pass
