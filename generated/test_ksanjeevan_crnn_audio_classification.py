import sys
_module = sys.modules[__name__]
del sys
data = _module
data_manager = _module
data_sets = _module
transforms = _module
eval = _module
evaluate = _module
infer = _module
net = _module
audio = _module
base_model = _module
loss = _module
metric = _module
model = _module
run = _module
train = _module
base_trainer = _module
trainer = _module
utils = _module
logger = _module
util = _module
visualization = _module

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


import numpy as np


import torch


import torch.utils.data as data


from torch.utils.data.dataloader import default_collate


import torch.nn as nn


from torch.distributions import Uniform


import logging


import torch.nn.functional as F


class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):
        super(SpecNormalization, self).__init__()
        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x

    def z_transform(self, x):
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x):
        return self._norm(x)


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, config=''):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.classes = None

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__(
            ) + '\nTrainable parameters: {}'.format(params)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ksanjeevan_crnn_audio_classification(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SpecNormalization(*[], **{'norm_type': 4}), [torch.rand([4, 4, 4, 4])], {})

