import sys
_module = sys.modules[__name__]
del sys
config = _module
A3D_download = _module
A3D_split = _module
anomaly_measures = _module
rnn_ed = _module
trackers = _module
utils = _module
data_prep_utils = _module
eval_utils = _module
flow_utils = _module
fol_dataloader = _module
train_val_utils = _module
visualize_utils = _module
run_AD = _module
run_fol_for_AD = _module
clear_detection_results = _module
merge_hevi_pkls = _module
odo_to_ego_motion = _module
video2frames = _module
test_fol = _module
train_ego_pred = _module
train_fol = _module

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


import numpy as np


import copy


import torch


from torch import nn


from torch import optim


from torch.nn import functional as F


from collections import deque


from torch.utils import data


import matplotlib.pyplot as plt


import time


class EncoderGRU(nn.Module):

    def __init__(self, args):
        super(EncoderGRU, self).__init__()
        self.args = args
        self.enc = nn.GRUCell(input_size=self.args.input_embed_size, hidden_size=self.args.enc_hidden_size)

    def forward(self, embedded_input, h_init):
        """
        The encoding process
        Params:
            x: input feature, (batch_size, time, feature dims)
            h_init: initial hidden state, (batch_size, enc_hidden_size)
        Returns:
            h: updated hidden state of the next time step, (batch.size, enc_hiddden_size)
        """
        h = self.enc(embedded_input, h_init)
        return h


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecoderGRU(nn.Module):

    def __init__(self, args):
        super(DecoderGRU, self).__init__()
        self.args = args
        self.hidden_to_pred_input = nn.Sequential(nn.Linear(self.args.dec_hidden_size, self.args.predictor_input_size), nn.ReLU())
        self.dec = nn.GRUCell(input_size=self.args.predictor_input_size, hidden_size=self.args.dec_hidden_size)
        if self.args.non_linear_output:
            self.hidden_to_pred = nn.Sequential(nn.Linear(self.args.dec_hidden_size, self.args.pred_dim), nn.Tanh())
        else:
            self.hidden_to_pred = nn.Linear(self.args.dec_hidden_size, self.args.pred_dim)

    def forward(self, h, embedded_ego_pred=None):
        """
        A RNN preditive model for future observation prediction
        Params:
            h: hidden state tensor from the encoder, (batch_size, enc_hidden_size)
            embedded_ego_pred: (batch_size, pred_timesteps, input_embed_size)
        """
        output = torch.zeros(h.shape[0], self.args.pred_timesteps, self.args.pred_dim)
        all_pred_h = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.dec_hidden_size])
        all_pred_inputs = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.predictor_input_size])
        pred_inputs = torch.zeros(h.shape[0], self.args.predictor_input_size)
        for i in range(self.args.pred_timesteps):
            if self.args.with_ego:
                pred_inputs = (embedded_ego_pred[:, i, :] + pred_inputs) / 2
            all_pred_inputs[:, i, :] = pred_inputs
            h = self.dec(pred_inputs, h)
            pred_inputs = self.hidden_to_pred_input(h)
            all_pred_h[:, i, :] = h
            output[:, i, :] = self.hidden_to_pred(h)
        return output, all_pred_h, all_pred_inputs


class FolRNNED(nn.Module):
    """Future object localization module"""

    def __init__(self, args):
        super(FolRNNED, self).__init__()
        self.args = copy.deepcopy(args)
        self.box_enc_args = copy.deepcopy(args)
        self.flow_enc_args = copy.deepcopy(args)
        if self.args.enc_concat_type == 'cat':
            self.args.dec_hidden_size = self.args.box_enc_size + self.args.flow_enc_size
        elif self.args.box_enc_size != self.args.flow_enc_size:
            raise ValueError('Box encoder size %d != flow encoder size %d' % (self.args.box_enc_size, self.args.flow_enc_size))
        else:
            self.args.dec_hidden_size = self.args.box_enc_size
        self.box_enc_args.enc_hidden_size = self.args.box_enc_size
        self.flow_enc_args.enc_hidden_size = self.args.flow_enc_size
        self.box_encoder = EncoderGRU(self.box_enc_args)
        self.flow_encoder = EncoderGRU(self.flow_enc_args)
        self.args.non_linear_output = True
        self.predictor = DecoderGRU(self.args)
        self.box_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size), nn.ReLU())
        self.flow_embed = nn.Sequential(nn.Linear(50, self.args.input_embed_size), nn.ReLU())
        self.ego_pred_embed = nn.Sequential(nn.Linear(3, self.args.input_embed_size), nn.ReLU())

    def forward(self, box, flow, ego_pred):
        """
        The RNN encoder decoder model rewritten from fvl2019icra-keras
        Params:
            box: (batch_size, segment_len, 4)
            flow: (batch_size, segment_len, 5, 5, 2)
            ego_pred: (batch_size, segment_len, pred_timesteps, 3) or None
            
            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            fol_predictions: predicted with shape (batch_size, segment_len, pred_timesteps, pred_dim)
        """
        self.args.batch_size = box.shape[0]
        if len(flow.shape) > 3:
            flow = flow.view(self.args.batch_size, self.args.segment_len, -1)
        embedded_box_input = self.box_embed(box)
        embedded_flow_input = self.flow_embed(flow)
        embedded_ego_input = self.ego_pred_embed(ego_pred)
        box_h = torch.zeros(self.args.batch_size, self.args.box_enc_size)
        flow_h = torch.zeros(self.args.batch_size, self.args.flow_enc_size)
        fol_predictions = torch.zeros(self.args.batch_size, self.args.segment_len, self.args.pred_timesteps, self.args.pred_dim)
        for i in range(self.args.segment_len):
            box_h = self.box_encoder(embedded_box_input[:, i, :], box_h)
            flow_h = self.flow_encoder(embedded_flow_input[:, i, :], flow_h)
            if self.args.enc_concat_type == 'cat':
                hidden_state = torch.cat((box_h, flow_h), dims=1)
            elif self.args.enc_concat_type in ['sum', 'avg', 'average']:
                hidden_state = (box_h + flow_h) / 2
            else:
                raise NameError(self.args.enc_concat_type, ' is unknown!!')
            if self.args.with_ego:
                output, _, _ = self.predictor(hidden_state, embedded_ego_input[:, i, :, :])
            else:
                output, _, _ = self.predictor(hidden_state, None)
            fol_predictions[:, i, :, :] = output
        return fol_predictions

    def predict(self, box, flow, box_h, flow_h, ego_pred):
        """
        predictor function, run forward inference to predict the future bboxes
        Params:
            box: (1, 4)
            flow: (1, 1, 5, 5, 2)
            ego_pred: (1, pred_timesteps, 3)
        return:
            box_changes:()
            box_h, 
            flow_h
        """
        if len(flow.shape) > 3:
            flow = flow.view(1, -1)
        embedded_box_input = self.box_embed(box)
        embedded_flow_input = self.flow_embed(flow)
        embedded_ego_input = None
        if self.args.with_ego:
            embedded_ego_input = self.ego_pred_embed(ego_pred)
        box_h = self.box_encoder(embedded_box_input, box_h)
        flow_h = self.flow_encoder(embedded_flow_input, flow_h)
        if self.args.enc_concat_type == 'cat':
            hidden_state = torch.cat((box_h, flow_h), dims=1)
        elif self.args.enc_concat_type in ['sum', 'avg', 'average']:
            hidden_state = (box_h + flow_h) / 2
        else:
            raise NameError(self.args.enc_concat_type, ' is unknown!!')
        box_changes, _, _ = self.predictor(hidden_state, embedded_ego_input)
        return box_changes, box_h, flow_h


class EgoRNNED(nn.Module):

    def __init__(self, args):
        super(EgoRNNED, self).__init__()
        self.args = copy.deepcopy(args)
        self.args.input_embed_size = self.args.ego_embed_size
        self.args.enc_hidden_size = self.args.ego_enc_size
        self.args.dec_hidden_size = self.args.ego_dec_size
        self.args.pred_dim = self.args.ego_dim
        self.args.predictor_input_size = self.args.ego_pred_input_size
        self.args.with_ego = False
        self.ego_encoder = EncoderGRU(self.args)
        self.ego_embed = nn.Sequential(nn.Linear(3, self.args.ego_embed_size), nn.ReLU())
        self.args.non_linear_output = False
        self.predictor = DecoderGRU(self.args)

    def forward(self, ego_x, image=None):
        """
        The RNN encoder decoder model for ego motion prediction
        Params:
            ego_x: (batch_size, segment_len, ego_dim)
            image: (batch_size, segment_len, feature_dim) e.g. feature_dim = 1024
            
            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            predictions: predicted ego motion with shape (batch_size, segment_len, pred_timesteps, ego_dim)
        """
        self.args.batch_size = ego_x.shape[0]
        embedded_ego_input = self.ego_embed(ego_x)
        ego_h = torch.zeros(self.args.batch_size, self.args.enc_hidden_size)
        predictions = torch.zeros(self.args.batch_size, self.args.segment_len, self.args.pred_timesteps, self.args.pred_dim)
        for i in range(self.args.segment_len):
            ego_h = self.ego_encoder(embedded_ego_input[:, i, :], ego_h)
            output, _, _ = self.predictor(ego_h)
            predictions[:, i, :, :] = output
        return predictions

    def predict(self, ego_x, ego_h, image=None):
        """
        Params:
            ego_x: (1, 3)
            ego_h: (1, 64)
            #image: (1, 1, 1024) e.g. feature_dim = 1024
        returns:
            ego_changes: (pred_timesteps, 3)
            ego_h: (1, ego_enc_size)
        """
        embedded_ego_input = self.ego_embed(ego_x)
        ego_h = self.ego_encoder(embedded_ego_input, ego_h)
        ego_changes, _, _ = self.predictor(ego_h)
        return ego_changes, ego_h


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderGRU,
     lambda: ([], {'args': _mock_config(input_embed_size=4, enc_hidden_size=4)}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_MoonBlvd_tad_IROS2019(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

