import sys
_module = sys.modules[__name__]
del sys
dataset = _module
inference = _module
focal_loss = _module
metrics = _module
weight_init = _module
decoder = _module
pse = _module
stclassifier = _module
tae = _module
dataset_preparation = _module
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


from torch import Tensor


from torch.utils import data


import pandas as pd


import numpy as np


import torch.utils.data as data


import torch.nn.functional as F


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.init as init


import copy


from sklearn.model_selection import KFold


from sklearn.metrics import confusion_matrix


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class linlayer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))
        out = self.lin(out)
        out = out.permute((0, 2, 1))
        out = self.bn(out)
        out = F.relu(out)
        return out


def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out


def masked_std(x, mask):
    m = masked_mean(x, mask)
    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))
    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2
    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 1e-31)
    out = out.permute(1, 0)
    return out


def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()


def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()


pooling_methods = {'mean': masked_mean, 'std': masked_std, 'max': maximum, 'min': minimum}


class PixelSetEncoder(nn.Module):

    def __init__(self, input_dim, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[64, 128], with_extra=True, extra_size=4):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
            pooling (str): Pixel-embedding pooling strategy, can be chosen in ('mean','std','max,'min')
                or any underscore-separated combination thereof.
            mlp2 (list): Dimensions of the successive feature spaces of MLP2
            with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
            extra_size (int, optional): Number of channels of the additional features, if any.
        """
        super(PixelSetEncoder, self).__init__()
        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling
        self.with_extra = with_extra
        self.extra_size = extra_size
        self.name = 'PSE-{}-{}-{}'.format('|'.join(list(map(str, self.mlp1_dim))), pooling, '|'.join(list(map(str, self.mlp2_dim))))
        self.output_dim = input_dim * len(pooling.split('_')) if len(self.mlp2_dim) == 0 else self.mlp2_dim[-1]
        inter_dim = self.mlp1_dim[-1] * len(pooling.split('_'))
        if self.with_extra:
            self.name += 'Extra'
            inter_dim += self.extra_size
        assert input_dim == mlp1[0]
        assert inter_dim == mlp2[0]
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(linlayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(nn.Linear(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.mlp2_dim[i + 1]))
            if i < len(self.mlp2_dim) - 2:
                layers.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, input):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
        Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
        Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
        Extra-features : Batch_size x (Sequence length) x Number of features

        If the input tensors have a temporal dimension, it will be combined with the batch dimension so that the
        complete sequences are processed at once. Then the temporal dimension is separated back to produce a tensor of
        shape Batch_size x Sequence length x Embedding dimension
        """
        a, b = input
        if len(a) == 2:
            out, mask = a
            extra = b
            if len(extra) == 2:
                extra, bm = extra
        else:
            out, mask = a, b
        if len(out.shape) == 4:
            reshape_needed = True
            batch, temp = out.shape[:2]
            out = out.view(batch * temp, *out.shape[2:])
            mask = mask.view(batch * temp, -1)
            if self.with_extra:
                extra = extra.view(batch * temp, -1)
        else:
            reshape_needed = False
        out = self.mlp1(out)
        out = torch.cat([pooling_methods[n](out, mask) for n in self.pooling.split('_')], dim=1)
        if self.with_extra:
            out = torch.cat([out, extra], dim=1)
        out = self.mlp2(out)
        if reshape_needed:
            out = out.view(batch, temp, -1)
        return out


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))
        self.fc2 = nn.Sequential(nn.BatchNorm1d(n_head * d_k), nn.Linear(n_head * d_k, n_head * d_k))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()
        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)
        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        v = v.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)
        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """ Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table)
    else:
        return torch.FloatTensor(sinusoid_table)


class TemporalAttentionEncoder(nn.Module):

    def __init__(self, in_channels=128, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24, positions=None):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model

        """
        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.name = 'TAE_dk{}_{}Heads_{}_T{}_do{}'.format(d_k, n_head, '|'.join(list(map(str, self.n_neurons))), T, dropout)
        if positions is None:
            positions = len_max_seq + 1
        else:
            self.name += '_bespokePos'
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(positions, self.in_channels, T=T), freeze=True)
        self.inlayernorm = nn.LayerNorm(self.in_channels)
        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1), nn.LayerNorm(d_model, len_max_seq))
            self.name += '_dmodel{}'.format(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None
        self.outlayernorm = nn.LayerNorm(self.d_model)
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        assert self.n_neurons[0] == n_head * self.d_model
        assert self.n_neurons[-1] == self.d_model
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]), nn.BatchNorm1d(self.n_neurons[i + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sz_b, seq_len, d = x.shape
        x = self.inlayernorm(x)
        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len)
        enc_output = x + self.position_enc(src_pos)
        if self.inconv is not None:
            enc_output = self.inconv(enc_output.permute(0, 2, 1)).permute(0, 2, 1)
        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)
        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))
        return enc_output


def get_decoder(n_neurons):
    layers = []
    for i in range(len(n_neurons) - 1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
        if i < len(n_neurons) - 2:
            layers.extend([nn.BatchNorm1d(n_neurons[i + 1]), nn.ReLU()])
    m = nn.Sequential(*layers)
    return m


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True, extra_size=4, n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24, positions=None, mlp4=[128, 64, 32, 20]):
        super(PseTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra, extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model, n_neurons=mlp3, dropout=dropout, T=T, len_max_seq=len_max_seq, positions=positions)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)
        out = self.temporal_encoder(out)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)
        None
        None


class PseTae_pretrained(nn.Module):

    def __init__(self, weight_folder, hyperparameters, device='cuda', fold='all'):
        """
        Pretrained PseTea classifier.
        The class can either load the weights of a single fold or aggregate the predictions of the different sets of
        weights obtained during k-fold cross-validation and produces a single prediction.
        Args:
            weight_folder (str): Path to the folder containing the different sets of weights obtained during each fold
            (res_dir of the training script)
            hyperparameters (dict): Hyperparameters of the PseTae classifier
            device (str): Device on which the model should be loaded ('cpu' or 'cuda')
            fold( str or int): load all folds ('all') or number of the fold to load
        """
        super(PseTae_pretrained, self).__init__()
        self.weight_folder = weight_folder
        self.hyperparameters = hyperparameters
        self.fold_folders = [os.path.join(weight_folder, f) for f in os.listdir(weight_folder) if os.path.isdir(os.path.join(weight_folder, f))]
        if fold == 'all':
            self.n_folds = len(self.fold_folders)
        else:
            self.n_folds = 1
            self.fold_folders = [self.fold_folders[int(fold) - 1]]
        self.model_instances = []
        None
        for f in self.fold_folders:
            m = PseTae(**hyperparameters)
            if device == 'cpu':
                map_loc = 'cpu'
            else:
                map_loc = 'cuda:{}'.format(torch.cuda.current_device())
                m = m
            d = torch.load(os.path.join(f, 'model.pth.tar'), map_location=map_loc)
            m.load_state_dict(d['state_dict'])
            self.model_instances.append(m)
        None

    def forward(self, input):
        """ Returns class logits
        Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            outputs = [F.log_softmax(m(input), dim=1) for m in self.model_instances]
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
        return outputs

    def predict_class(self, input):
        """Returns class prediction
                Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            pred = self.forward(input).argmax(dim=1)
        return pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'n_head': 4, 'd_k': 4, 'd_in': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (linlayer,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_VSainteuf_pytorch_psetae(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

