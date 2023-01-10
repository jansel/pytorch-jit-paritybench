import sys
_module = sys.modules[__name__]
del sys
beamdecode = _module
data = _module
decoder = _module
_init_path = _module
embedding = _module
record = _module
train = _module
feature = _module
models = _module
base = _module
conv = _module
trainable = _module
train = _module

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


import torch.nn.functional as F


import numpy as np


import scipy


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.nn.utils import remove_weight_norm


import torch.nn as nn


from torch.nn.utils import weight_norm


import torch.optim as optim


class MASRModel(nn.Module):

    def __init__(self, **config):
        super().__init__()
        self.config = config

    @classmethod
    def load(cls, path):
        package = torch.load(path)
        state_dict = package['state_dict']
        config = package['config']
        m = cls(**config)
        m.load_state_dict(state_dict)
        return m

    def to_train(self):
        self.__class__.__bases__ = TrainableModel,
        return self

    def predict(self, *args):
        raise NotImplementedError()

    def _default_decode(self, yp, yp_lens):
        idxs = yp.argmax(1)
        texts = []
        for idx, out_len in zip(idxs, yp_lens):
            idx = idx[:out_len]
            text = ''
            last = None
            for i in idx:
                if i.item() not in (last, self.blank):
                    text += self.vocabulary[i.item()]
                last = i
            texts.append(text)
        return texts

    def decode(self, *outputs):
        return self._default_decode(*outputs)


class ConvBlock(nn.Module):

    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = weight_norm(self.conv)
        self.act = nn.GLU(1)
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GatedConv(MASRModel):
    """ This is a model between Wav2letter and Gated Convnets.
        The core block of this model is Gated Convolutional Network"""

    def __init__(self, vocabulary, blank=0, name='masr'):
        """ vocabulary : str : string of all labels such that vocaulary[0] == ctc_blank  """
        super().__init__(vocabulary=vocabulary, name=name, blank=blank)
        self.blank = blank
        self.vocabulary = vocabulary
        self.name = name
        output_units = len(vocabulary)
        modules = []
        modules.append(ConvBlock(nn.Conv1d(161, 500, 48, 2, 97), 0.2))
        for i in range(7):
            modules.append(ConvBlock(nn.Conv1d(250, 500, 7, 1), 0.3))
        modules.append(ConvBlock(nn.Conv1d(250, 2000, 32, 1), 0.5))
        modules.append(ConvBlock(nn.Conv1d(1000, 2000, 1, 1), 0.5))
        modules.append(weight_norm(nn.Conv1d(1000, output_units, 1, 1)))
        self.cnn = nn.Sequential(*modules)

    def forward(self, x, lens):
        x = self.cnn(x)
        for module in self.modules():
            if type(module) == nn.modules.Conv1d:
                lens = (lens - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
        return x, lens

    def predict(self, path):
        self.eval()
        wav = feature.load_audio(path)
        spec = feature.spectrogram(wav)
        spec.unsqueeze_(0)
        x_lens = spec.size(-1)
        out = self.cnn(spec)
        out_len = torch.tensor([out.size(-1)])
        text = self.decode(out, out_len)
        self.train()
        return text[0]


class TrainableModel(MASRModel):

    def __init__(self, **config):
        super().__init__(**config)

    def save(self, path):
        state_dict = self.state_dict()
        config = self.config
        package = {'state_dict': state_dict, 'config': config}
        torch.save(package, path)

    def loss(self, *pred_targets):
        preds, targets = pred_targets
        return self._default_loss(*preds, *targets)

    def cer(self, texts, *targets):
        return self._default_cer(texts, *targets)

    def _default_loss(self, yp, yp_lens, y, y_lens):
        criterion = CTCLoss(size_average=True)
        yp = yp.permute(2, 0, 1)
        loss = criterion(yp, y, yp_lens, y_lens)
        return loss

    def _default_cer(self, texts, y, y_lens):
        index = 0
        cer = 0
        for text, y_len in zip(texts, y_lens):
            target = y[index:index + y_len]
            target = ''.join(self.vocabulary[i] for i in target)
            None
            cer += distance(text, target) / len(target)
            index += y_len
        cer /= len(y_lens)
        return cer

    def test(self, test_index, batch_size=64):
        self.eval()
        test_dataset = data.MASRDataset(test_index, self.vocabulary)
        test_loader = data.MASRDataLoader(test_dataset, batch_size, shuffle=False, num_workers=16)
        test_steps = len(test_loader)
        cer = 0
        for inputs, targets in tqdm(test_loader, total=test_steps):
            x, x_lens = inputs
            x = x
            outputs = self.forward(x, x_lens)
            texts = self.decode(*outputs)
            cer += self.cer(texts, *targets)
        cer /= test_steps
        self.train()
        return cer

    def fit(self, train_index, dev_index, epochs=100, train_batch_size=64, lr=0.6, momentum=0.8, grad_clip=0.2, dev_batch_size=64, sorta_grad=True, tensorboard=True, quiet=False):
        self
        self.train()
        if tensorboard:
            writer = SummaryWriter()
        optimizer = optim.SGD(self.parameters(), lr, momentum, nesterov=True)
        train_dataset = data.MASRDataset(train_index, self.vocabulary)
        train_loader_shuffle = data.MASRDataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=16)
        if sorta_grad:
            train_loader_sort = data.MASRDataLoader(train_dataset, train_batch_size, shuffle=False, num_workers=16)
        train_steps = len(train_loader_shuffle)
        gstep = 0
        for epoch in range(epochs):
            avg_loss = 0
            if epoch == 0 and sorta_grad:
                train_loader = train_loader_sort
            else:
                train_loader = train_loader_shuffle
            for step, (inputs, targets) in enumerate(train_loader):
                x, x_lens = inputs
                x = x
                gstep += 1
                outputs = self.forward(x, x_lens)
                loss = self.loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                avg_loss += loss.item()
                if not quiet:
                    None
                if tensorboard:
                    writer.add_scalar('loss/step', loss.item(), gstep)
            cer = self.test(dev_index, dev_batch_size)
            avg_loss /= train_steps
            if not quiet:
                None
            if tensorboard:
                writer.add_scalar('cer/epoch', cer, epoch + 1)
                writer.add_scalar('loss/epoch', loss, epoch + 1)
            self.save('pretrained/{}_epoch_{}.pth'.format(self.name, epoch + 1))

