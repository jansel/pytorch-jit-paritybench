import sys
_module = sys.modules[__name__]
del sys
core = _module
nlu = _module
engine = _module
neuralnet = _module
config = _module
dataset = _module
model = _module
optimize_graph = _module
train = _module
utils = _module
make_dataset = _module
speechrecognition = _module
decoder = _module
demo = _module
engine = _module
dataset = _module
model = _module
optimize_graph = _module
scorer = _module
train = _module
utils = _module
commonvoice_create_jsons = _module
speechsynthesis = _module
engine = _module
dataset = _module
model = _module
optimize_graph = _module
train = _module
collect_wakeword_audio = _module
create_wakeword_jsons = _module
replicate_audios = _module
split_audio_into_chunks = _module
split_commonvoice = _module

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


from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader


import pandas as pd


import numpy as np


from sklearn import preprocessing


from sklearn import model_selection


import itertools


import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_fscore_support


from sklearn.metrics import confusion_matrix


import time


import torchaudio


from torch.nn import functional as F


from collections import OrderedDict


import torch.optim as optim


import torch.utils.data as data


from sklearn.metrics import classification_report


class NLUModel(nn.Module):

    def __init__(self, num_entity, num_intent, num_scenario):
        super(NLUModel, self).__init__()
        self.num_entity = num_entity
        self.num_intent = num_intent
        self.num_scenario = num_scenario
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL)
        self.drop_1 = nn.Dropout(0.3)
        self.drop_2 = nn.Dropout(0.3)
        self.drop_3 = nn.Dropout(0.3)
        self.out_entity = nn.Linear(768, self.num_entity)
        self.out_intent = nn.Linear(768, self.num_intent)
        self.out_scenario = nn.Linear(768, self.num_scenario)

    def forward(self, ids, mask, token_type_ids):
        out = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        hs, cls_hs = out['last_hidden_state'], out['pooler_output']
        entity_hs = self.drop_1(hs)
        intent_hs = self.drop_2(cls_hs)
        scenario_hs = self.drop_3(cls_hs)
        entity_hs = self.out_entity(entity_hs)
        intent_hs = self.out_intent(intent_hs)
        scenario_hs = self.out_scenario(scenario_hs)
        return entity_hs, intent_hs, scenario_hs


class SpecAugment(nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()
        self.rate = rate
        self.specaug = nn.Sequential(torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask), torchaudio.transforms.TimeMasking(time_mask_param=time_mask))
        self.specaug2 = nn.Sequential(torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask), torchaudio.transforms.TimeMasking(time_mask_param=time_mask), torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask), torchaudio.transforms.TimeMasking(time_mask_param=time_mask))
        policies = {(1): self.policy1, (2): self.policy2, (3): self.policy3}
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)


class LogMelSpec(nn.Module):

    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, win_length=win_length, hop_length=hop_length)

    def forward(self, x):
        x = self.transform(x)
        x = np.log(x + 1e-14)
        return x


class ActDropNormCNN1D(nn.Module):

    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    hyper_parameters = {'num_classes': 29, 'n_feats': 81, 'dropout': 0.1, 'hidden_size': 1024, 'num_layers': 1}

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(nn.Conv1d(n_feats, n_feats, 10, 2, padding=10 // 2), ActDropNormCNN1D(n_feats, dropout))
        self.dense = nn.Sequential(nn.Linear(n_feats, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0, bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return torch.zeros(n * 1, batch_size, hs), torch.zeros(n * 1, batch_size, hs)

    def forward(self, x, hidden):
        x = x.squeeze(1)
        x = self.cnn(x)
        x = self.dense(x)
        x = x.transpose(0, 1)
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out)))
        return self.final_fc(x), (hn, cn)


class MFCC(nn.Module):

    def __init__(self, sample_rate, fft_size=400, window_stride=(400, 200), num_filt=40, num_coeffs=40):
        super(MFCC, self).__init__()
        self.sample_rate = sample_rate
        self.window_stride = window_stride
        self.fft_size = fft_size
        self.num_filt = num_filt
        self.num_coeffs = num_coeffs
        self.mfcc = lambda x: mfcc_spec(x, self.sample_rate, self.window_stride, self.fft_size, self.num_filt, self.num_coeffs)

    def forward(self, x):
        return torch.Tensor(self.mfcc(x.squeeze(0).numpy())).transpose(0, 1).unsqueeze(0)


class RandomCut(nn.Module):
    """Augmentation technique that randomly cuts start or end of audio"""

    def __init__(self, max_cut=10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x):
        """Randomly cuts from start or end of batch"""
        side = torch.randint(0, 1, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            return x[:-cut, :, :]
        elif side == 1:
            return x[cut:, :, :]


class LSTMWakeWord(nn.Module):

    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device='cpu'):
        super(LSTMWakeWord, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1
        self.device = device
        self.layernorm = nn.LayerNorm(feature_size)
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_size * self.directions, num_classes)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return torch.zeros(n * d, batch_size, hs), torch.zeros(n * d, batch_size, hs)

    def forward(self, x):
        x = self.layernorm(x)
        hidden = self._init_hidden(x.size()[1])
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ActDropNormCNN1D,
     lambda: ([], {'n_feats': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogMelSpec,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     False),
    (RandomCut,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpecAugment,
     lambda: ([], {'rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_LearnedVector_A_Hackers_AI_Voice_Assistant(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

