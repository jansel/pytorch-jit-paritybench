import sys
_module = sys.modules[__name__]
del sys
DAN = _module
cfgs_hw = _module
cfgs_scene = _module
dataset_hw = _module
dataset_scene = _module
main = _module
resnet = _module
utils = _module

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


import torch.nn as nn


from torch.nn import init


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import math


import torch.utils.model_zoo as model_zoo


import numpy as np


class Feature_Extractor(nn.Module):

    def __init__(self, strides, compress_layer, input_shape):
        super(Feature_Extractor, self).__init__()
        self.model = resnet.resnet45(strides, compress_layer)
        self.input_shape = input_shape

    def forward(self, input):
        features = self.model(input)
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[
            1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


class CAM(nn.Module):

    def __init__(self, scales, maxT, depth, num_channels):
        super(CAM, self).__init__()
        fpn = []
        for i in range(1, len(scales)):
            assert not scales[i - 1][1] / scales[i][1
                ] % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not scales[i - 1][2] / scales[i][2
                ] % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]
            r_h, r_w = int(scales[i - 1][1] / scales[i][1]), int(scales[i -
                1][2] / scales[i][2])
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][
                0], (ksize_h, ksize_w), (r_h, r_w), (int((ksize_h - 1) / 2),
                int((ksize_w - 1) / 2))), nn.BatchNorm2d(scales[i][0]), nn.
                ReLU(True)))
        self.fpn = nn.Sequential(*fpn)
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth / 2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1
                ]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([(_ ** 2) for _ in stride])
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels, tuple(
            conv_ksizes[0]), tuple(strides[0]), (int((conv_ksizes[0][0] - 1
            ) / 2), int((conv_ksizes[0][1] - 1) / 2))), nn.BatchNorm2d(
            num_channels), nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                tuple(conv_ksizes[i]), tuple(strides[i]), (int((conv_ksizes
                [i][0] - 1) / 2), int((conv_ksizes[i][1] - 1) / 2))), nn.
                BatchNorm2d(num_channels), nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels,
                num_channels, tuple(deconv_ksizes[int(depth / 2) - i]),
                tuple(strides[int(depth / 2) - i]), (int(deconv_ksizes[int(
                depth / 2) - i][0] / 4.0), int(deconv_ksizes[int(depth / 2) -
                i][1] / 4.0))), nn.BatchNorm2d(num_channels), nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
            tuple(deconv_ksizes[0]), tuple(strides[0]), (int(deconv_ksizes[
            0][0] / 4.0), int(deconv_ksizes[0][1] / 4.0))), nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + input[i + 1]
        conv_feats = []
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)
            conv_feats.append(x)
        for i in range(0, len(self.deconvs) - 1):
            x = self.deconvs[i](x)
            x = x + conv_feats[len(conv_feats) - 2 - i]
        x = self.deconvs[-1](x)
        return x


class CAM_transposed(nn.Module):

    def __init__(self, scales, maxT, depth, num_channels):
        super(CAM_transposed, self).__init__()
        fpn = []
        for i in range(1, len(scales)):
            assert not scales[i - 1][1] / scales[i][1
                ] % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not scales[i - 1][2] / scales[i][2
                ] % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]
            r_h, r_w = scales[i - 1][1] / scales[i][1], scales[i - 1][2
                ] / scales[i][2]
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][
                0], (ksize_h, ksize_w), (r_h, r_w), ((ksize_h - 1) / 2, (
                ksize_w - 1) / 2)), nn.BatchNorm2d(scales[i][0]), nn.ReLU(
                True)))
        fpn.append(nn.Sequential(nn.Conv2d(scales[i][0], 1, (1, ksize_w), (
            1, r_w), (0, (ksize_w - 1) / 2)), nn.Sigmoid()))
        self.fpn = nn.Sequential(*fpn)
        in_shape = scales[-1]
        deconvs = []
        ksize_h = 1 if in_shape[1] == 1 else 4
        for i in range(1, depth / 2):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels,
                num_channels, (ksize_h, 4), (r_h, 2), (int(ksize_h / 4.0), 
                1)), nn.BatchNorm2d(num_channels), nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
            (ksize_h, 4), (r_h, 2), (int(ksize_h / 4.0), 1)), nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn) - 1):
            x = self.fpn[i](x) + input[i + 1]
        x = self.fpn[-1](x)
        x = x.permute(0, 3, 1, 2).contiguous()
        for i in range(0, len(self.deconvs)):
            x = self.deconvs[i](x)
        return x


class DTD(nn.Module):

    def __init__(self, nclass, nchannel, dropout=0.3):
        super(DTD, self).__init__()
        self.nclass = nclass
        self.nchannel = nchannel
        self.pre_lstm = nn.LSTM(nchannel, int(nchannel / 2), bidirectional=True
            )
        self.rnn = nn.GRUCell(nchannel * 2, nchannel)
        self.generator = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(
            nchannel, nclass))
        self.char_embeddings = Parameter(torch.randn(nclass, nchannel))

    def forward(self, feature, A, text, text_length, test=False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)
        C, _ = self.pre_lstm(C)
        C = F.dropout(C, p=0.3, training=self.training)
        if not test:
            lenText = int(text_length.sum())
            nsteps = int(text_length.max())
            gru_res = torch.zeros(C.size()).type_as(C.data)
            out_res = torch.zeros(lenText, self.nclass).type_as(feature.data)
            out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)
            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB)
                .long().type_as(text.data))
            for i in range(0, nsteps):
                hidden = self.rnn(torch.cat((C[(i), :, :], prev_emb), dim=1
                    ), hidden)
                gru_res[(i), :, :] = hidden
                prev_emb = self.char_embeddings.index_select(0, text[:, (i)])
            gru_res = self.generator(gru_res)
            start = 0
            for i in range(0, nB):
                cur_length = int(text_length[i])
                out_res[start:start + cur_length] = gru_res[0:cur_length, (
                    i), :]
                out_attns[start:start + cur_length] = A[(i), 0:cur_length, :, :
                    ]
                start += cur_length
            return out_res, out_attns
        else:
            lenText = nT
            nsteps = nT
            out_res = torch.zeros(lenText, nB, self.nclass).type_as(feature
                .data)
            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB)
                .long().type_as(text.data))
            out_length = torch.zeros(nB)
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                hidden = self.rnn(torch.cat((C[(now_step), :, :], prev_emb),
                    dim=1), hidden)
                tmp_result = self.generator(hidden)
                out_res[now_step] = tmp_result
                tmp_result = tmp_result.topk(1)[1].squeeze()
                for j in range(nB):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                prev_emb = self.char_embeddings.index_select(0, tmp_result)
                now_step += 1
            for j in range(0, nB):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(
                feature.data)
            for i in range(0, nB):
                cur_length = int(out_length[i])
                output[start:start + cur_length] = out_res[0:cur_length, (i), :
                    ]
                start += cur_length
            return output, out_length


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, strides, compress_layer=True):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=strides[0],
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3]
            )
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4]
            )
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5]
            )
        self.compress_layer = compress_layer
        if compress_layer:
            self.layer6 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3,
                1), padding=(0, 0), stride=(1, 1)), nn.BatchNorm2d(256), nn
                .ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, multiscale=False):
        out_features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_shape = x.size()[2:]
        x = self.layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer5(x)
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)
        return out_features


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Wang_Tianwei_Decoupled_attention_network(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

