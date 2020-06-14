import sys
_module = sys.modules[__name__]
del sys
config = _module
demo = _module
lib = _module
datasets = _module
concatdataset = _module
dataset = _module
evaluation_metrics = _module
metrics = _module
evaluators = _module
loss = _module
sequenceCrossEntropyLoss = _module
models = _module
attention_recognition_head = _module
model_builder = _module
resnet_aster = _module
stn_head = _module
tps_spatial_transformer = _module
create_sub_lmdb = _module
create_svtp_lmdb = _module
trainers = _module
utils = _module
labelmaps = _module
logging = _module
meters = _module
osutils = _module
serialization = _module
visualization_utils = _module
main = _module

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


import string


import math


import torch


import torch.nn.functional as F


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn import init


from collections import OrderedDict


import torch.nn as nn


import itertools


from torch.nn import Parameter


def _assert_no_grad(variable):
    assert not variable.requires_grad, "nn criterions don't compute the gradient w.r.t. targets - please mark these variables as not requiring gradients"


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class SequenceCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=True, ignore_index=-100,
        sequence_normalize=False, sample_normalize=True):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        assert (sequence_normalize and sample_normalize) == False

    def forward(self, input, target, length):
        _assert_no_grad(target)
        batch_size, def_max_length = target.size(0), target.size(1)
        mask = torch.zeros(batch_size, def_max_length)
        for i in range(batch_size):
            mask[(i), :length[i]].fill_(1)
        mask = mask.type_as(input)
        max_length = max(length)
        assert max_length == input.size(1)
        target = target[:, :max_length]
        mask = mask[:, :max_length]
        input = to_contiguous(input).view(-1, input.size(2))
        input = F.log_softmax(input, dim=1)
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = -input.gather(1, target.long()) * mask
        output = torch.sum(output)
        if self.sequence_normalize:
            output = output / torch.sum(mask)
        if self.sample_normalize:
            output = output / batch_size
        return output


class AttentionRecognitionHead(nn.Module):
    """
  input: [b x 16 x 64 x in_planes]
  output: probability sequence: [b x T x num_classes]
  """

    def __init__(self, num_classes, in_planes, sDim, attDim, max_len_labels):
        super(AttentionRecognitionHead, self).__init__()
        self.num_classes = num_classes
        self.in_planes = in_planes
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=
            num_classes, attDim=attDim)

    def forward(self, x):
        x, targets, lengths = x
        batch_size = x.size(0)
        state = torch.zeros(1, batch_size, self.sDim)
        outputs = []
        for i in range(max(lengths)):
            if i == 0:
                y_prev = torch.zeros(batch_size).fill_(self.num_classes)
            else:
                y_prev = targets[:, (i - 1)]
            output, state = self.decoder(x, state, y_prev)
            outputs.append(output)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    def sample(self, x):
        x, _, _ = x
        batch_size = x.size(0)
        state = torch.zeros(1, batch_size, self.sDim)
        predicted_ids, predicted_scores = [], []
        for i in range(self.max_len_labels):
            if i == 0:
                y_prev = torch.zeros(batch_size).fill_(self.num_classes)
            else:
                y_prev = predicted
            output, state = self.decoder(x, state, y_prev)
            output = F.softmax(output, dim=1)
            score, predicted = output.max(1)
            predicted_ids.append(predicted.unsqueeze(1))
            predicted_scores.append(score.unsqueeze(1))
        predicted_ids = torch.cat(predicted_ids, 1)
        predicted_scores = torch.cat(predicted_scores, 1)
        return predicted_ids, predicted_scores

    def beam_search(self, x, beam_width, eos):

        def _inflate(tensor, times, dim):
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            return tensor.repeat(*repeat_dims)
        batch_size, l, d = x.size()
        inflated_encoder_feats = x.unsqueeze(1).permute((1, 0, 2, 3)).repeat((
            beam_width, 1, 1, 1)).permute((1, 0, 2, 3)).contiguous().view(-
            1, l, d)
        state = torch.zeros(1, batch_size * beam_width, self.sDim)
        pos_index = (torch.Tensor(range(batch_size)) * beam_width).long().view(
            -1, 1)
        sequence_scores = torch.Tensor(batch_size * beam_width, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.Tensor([(i * beam_width) for i in
            range(0, batch_size)]).long(), 0.0)
        y_prev = torch.zeros(batch_size * beam_width).fill_(self.num_classes)
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        for i in range(self.max_len_labels):
            output, state = self.decoder(inflated_encoder_feats, state, y_prev)
            log_softmax_output = F.log_softmax(output, dim=1)
            sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
            sequence_scores += log_softmax_output
            scores, candidates = sequence_scores.view(batch_size, -1).topk(
                beam_width, dim=1)
            y_prev = (candidates % self.num_classes).view(batch_size *
                beam_width)
            sequence_scores = scores.view(batch_size * beam_width, 1)
            predecessors = (candidates / self.num_classes + pos_index.
                expand_as(candidates)).view(batch_size * beam_width, 1)
            state = state.index_select(1, predecessors.squeeze())
            stored_scores.append(sequence_scores.clone())
            eos_indices = y_prev.view(-1, 1).eq(eos)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.masked_fill_(eos_indices, -float('inf'))
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(y_prev)
        p = list()
        l = [([self.max_len_labels] * beam_width) for _ in range(batch_size)]
        sorted_score, sorted_idx = stored_scores[-1].view(batch_size,
            beam_width).topk(beam_width)
        s = sorted_score.clone()
        batch_eos_found = [0] * batch_size
        t = self.max_len_labels - 1
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(
            batch_size * beam_width)
        while t >= 0:
            current_symbol = stored_emitted_symbols[t].index_select(0,
                t_predecessors)
            t_predecessors = stored_predecessors[t].index_select(0,
                t_predecessors).squeeze()
            eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_width)
                    res_k_idx = beam_width - batch_eos_found[b_idx
                        ] % beam_width - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_width + res_k_idx
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = stored_scores[t][idx[0], [0]]
                    l[b_idx][res_k_idx] = t + 1
            p.append(current_symbol)
            t -= 1
        s, re_sorted_idx = s.topk(beam_width)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[(
                b_idx), :]]
        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)
            ).view(batch_size * beam_width)
        p = [step.index_select(0, re_sorted_idx).view(batch_size,
            beam_width, -1) for step in reversed(p)]
        p = torch.cat(p, -1)[:, (0), :]
        return p, torch.ones_like(p)


class AttentionUnit(nn.Module):

    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim
        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

    def init_weights(self):
        init.normal_(self.sEmbed.weight, std=0.01)
        init.constant_(self.sEmbed.bias, 0)
        init.normal_(self.xEmbed.weight, std=0.01)
        init.constant_(self.xEmbed.bias, 0)
        init.normal_(self.wEmbed.weight, std=0.01)
        init.constant_(self.wEmbed.bias, 0)

    def forward(self, x, sPrev):
        batch_size, T, _ = x.size()
        x = x.view(-1, self.xDim)
        xProj = self.xEmbed(x)
        xProj = xProj.view(batch_size, T, -1)
        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)
        sProj = torch.unsqueeze(sProj, 1)
        sProj = sProj.expand(batch_size, T, self.attDim)
        sumTanh = torch.tanh(sProj + xProj)
        sumTanh = sumTanh.view(-1, self.attDim)
        vProj = self.wEmbed(sumTanh)
        vProj = vProj.view(batch_size, T)
        alpha = F.softmax(vProj, dim=1)
        return alpha


class DecoderUnit(nn.Module):

    def __init__(self, sDim, xDim, yDim, attDim):
        super(DecoderUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.attDim = attDim
        self.emdDim = attDim
        self.attention_unit = AttentionUnit(sDim, xDim, attDim)
        self.tgt_embedding = nn.Embedding(yDim + 1, self.emdDim)
        self.gru = nn.GRU(input_size=xDim + self.emdDim, hidden_size=sDim,
            batch_first=True)
        self.fc = nn.Linear(sDim, yDim)

    def init_weights(self):
        init.normal_(self.tgt_embedding.weight, std=0.01)
        init.normal_(self.fc.weight, std=0.01)
        init.constant_(self.fc.bias, 0)

    def forward(self, x, sPrev, yPrev):
        batch_size, T, _ = x.size()
        alpha = self.attention_unit(x, sPrev)
        context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
        yProj = self.tgt_embedding(yPrev.long())
        output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1
            ), sPrev)
        output = output.squeeze(1)
        output = self.fc(output)
        return output, state


def _normalize_text(text):
    text = ''.join(filter(lambda x: x in string.digits + string.
        ascii_letters, text))
    return text.lower()


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError('Cannot convert {} to numpy array'.format(type(
            tensor)))
    return tensor


def get_str_list(output, target, dataset=None):
    assert output.dim() == 2 and target.dim() == 2
    end_label = dataset.char2id[dataset.EOS]
    unknown_label = dataset.char2id[dataset.UNKNOWN]
    num_samples, max_len_labels = output.size()
    num_classes = len(dataset.char2id.keys())
    assert num_samples == target.size(0) and max_len_labels == target.size(1)
    output = to_numpy(output)
    target = to_numpy(target)
    pred_list, targ_list = [], []
    for i in range(num_samples):
        pred_list_i = []
        for j in range(max_len_labels):
            if output[i, j] != end_label:
                if output[i, j] != unknown_label:
                    pred_list_i.append(dataset.id2char[output[i, j]])
            else:
                break
        pred_list.append(pred_list_i)
    for i in range(num_samples):
        targ_list_i = []
        for j in range(max_len_labels):
            if target[i, j] != end_label:
                if target[i, j] != unknown_label:
                    targ_list_i.append(dataset.id2char[target[i, j]])
            else:
                break
        targ_list.append(targ_list_i)
    if True:
        pred_list = [_normalize_text(pred) for pred in pred_list]
        targ_list = [_normalize_text(targ) for targ in targ_list]
    else:
        pred_list = [''.join(pred) for pred in pred_list]
        targ_list = [''.join(targ) for targ in targ_list]
    return pred_list, targ_list


def Accuracy(output, target, dataset=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    return accuracy


def _lexicon_search(lexicon, word):
    edit_distances = []
    for lex_word in lexicon:
        edit_distances.append(editdistance.eval(_normalize_text(lex_word),
            _normalize_text(word)))
    edit_distances = np.asarray(edit_distances, dtype=np.int)
    argmin = np.argmin(edit_distances)
    return lexicon[argmin]


def Accuracy_with_lexicon(output, target, dataset=None, file_names=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    accuracys = []
    acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    accuracys.append(accuracy)
    if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons50[file_name],
            pred) for file_name, pred in zip(file_names, pred_list)]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list,
            targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)
    if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons1k[file_name],
            pred) for file_name, pred in zip(file_names, pred_list)]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list,
            targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)
    if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
        accuracys.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexiconsfull[file_name
            ], pred) for file_name, pred in zip(file_names, pred_list)]
        acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list,
            targ_list)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        accuracys.append(accuracy)
    return accuracys


def EditDistance(output, target, dataset=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(
        pred_list, targ_list)]
    eds = sum(ed_list)
    return eds


def EditDistance_with_lexicon(output, target, dataset=None, file_names=None):
    pred_list, targ_list = get_str_list(output, target, dataset)
    eds = []
    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(
        pred_list, targ_list)]
    ed = sum(ed_list)
    eds.append(ed)
    if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
        eds.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons50[file_name],
            pred) for file_name, pred in zip(file_names, pred_list)]
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(
            refined_pred_list, targ_list)]
        ed = sum(ed_list)
        eds.append(ed)
    if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
        eds.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexicons1k[file_name],
            pred) for file_name, pred in zip(file_names, pred_list)]
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(
            refined_pred_list, targ_list)]
        ed = sum(ed_list)
        eds.append(ed)
    if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
        eds.append(0)
    else:
        refined_pred_list = [_lexicon_search(dataset.lexiconsfull[file_name
            ], pred) for file_name, pred in zip(file_names, pred_list)]
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(
            refined_pred_list, targ_list)]
        ed = sum(ed_list)
        eds.append(ed)
    return eds


__factory = {'accuracy': Accuracy, 'editdistance': EditDistance,
    'accuracy_with_lexicon': Accuracy_with_lexicon,
    'editdistance_with_lexicon': EditDistance_with_lexicon}


def create(name, *args, **kwargs):
    """Create a model instance.
  
  Parameters
  ----------
  name: str
    Model name. One of __factory
  pretrained: bool, optional
    If True, will use ImageNet pretrained model. Default: True
  num_classes: int, optional
    If positive, will change the original classifier the fit the new classifier with num_classes. Default: True
  with_words: bool, optional
    If True, the input of this model is the combination of image and word. Default: False
  """
    if name not in __factory:
        raise KeyError('Unknown model:', name)
    return __factory[name](*args, **kwargs)


parser = argparse.ArgumentParser(description='Softmax loss classification')


def get_args(sys_args):
    global_args = parser.parse_args(sys_args)
    return global_args


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class AsterBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AsterBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_ASTER(nn.Module):
    """For aster or crnn"""

    def __init__(self, with_lstm=False, n_group=1):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = with_lstm
        self.n_group = n_group
        in_channels = 3
        self.layer0 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=
            (3, 3), stride=1, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])
        self.layer2 = self._make_layer(64, 4, [2, 2])
        self.layer3 = self._make_layer(128, 6, [2, 1])
        self.layer4 = self._make_layer(256, 6, [2, 1])
        self.layer5 = self._make_layer(512, 3, [2, 1])
        if with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2,
                batch_first=True)
            self.out_planes = 2 * 256
        else:
            self.out_planes = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes,
                stride), nn.BatchNorm2d(planes))
        layers = []
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        cnn_feat = x5.squeeze(2)
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            rnn_feat, _ = self.rnn(cnn_feat)
            return rnn_feat
        else:
            return cnn_feat


def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
        padding=1)
    block = nn.Sequential(conv_layer, nn.BatchNorm2d(out_planes), nn.ReLU(
        inplace=True))
    return block


class STNHead(nn.Module):

    def __init__(self, in_planes, num_ctrlpoints, activation='none'):
        super(STNHead, self).__init__()
        self.in_planes = in_planes
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(conv3x3_block(in_planes, 32), nn.
            MaxPool2d(kernel_size=2, stride=2), conv3x3_block(32, 64), nn.
            MaxPool2d(kernel_size=2, stride=2), conv3x3_block(64, 128), nn.
            MaxPool2d(kernel_size=2, stride=2), conv3x3_block(128, 256), nn
            .MaxPool2d(kernel_size=2, stride=2), conv3x3_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2), conv3x3_block(256, 256))
        self.stn_fc1 = nn.Sequential(nn.Linear(2 * 256, 512), nn.
            BatchNorm1d(512), nn.ReLU(inplace=True))
        self.stn_fc2 = nn.Linear(512, num_ctrlpoints * 2)
        self.init_weights(self.stn_convnet)
        self.init_weights(self.stn_fc1)
        self.init_stn(self.stn_fc2)

    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def init_stn(self, stn_fc2):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1.0 - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0
            ).astype(np.float32)
        if self.activation is 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1.0 / ctrl_points - 1.0)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        x = x.view(-1, self.num_ctrlpoints, 2)
        return img_feat, x


def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts_arr = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
        axis=0)
    output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, (0)] + pairwise_diff_square[:,
        :, (1)]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


class TPSSpatialTransformer(nn.Module):

    def __init__(self, output_image_size=None, num_control_points=None,
        margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins
        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points,
            margins)
        N = num_control_points
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, (-3)].fill_(1)
        forward_kernel[(-3), :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        inverse_kernel = torch.inverse(forward_kernel)
        HW = self.target_height * self.target_width
        target_coordinate = list(itertools.product(range(self.target_height
            ), range(self.target_width)))
        target_coordinate = torch.Tensor(target_coordinate)
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y / (self.target_height - 1)
        X = X / (self.target_width - 1)
        target_coordinate = torch.cat([X, Y], dim=1)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate
            , target_control_points)
        target_coordinate_repr = torch.cat([target_coordinate_partial_repr,
            torch.ones(HW, 1), target_coordinate], dim=1)
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, self.padding_matrix.expand(
            batch_size, 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr,
            mapping_matrix)
        grid = source_coordinate.view(-1, self.target_height, self.
            target_width, 2)
        grid = torch.clamp(grid, 0, 1)
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ayumiymk_aster_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(AsterBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(AttentionUnit(*[], **{'sDim': 4, 'xDim': 4, 'attDim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ResNet_ASTER(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

