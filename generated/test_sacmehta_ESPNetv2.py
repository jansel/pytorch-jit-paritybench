import sys
_module = sys.modules[__name__]
del sys
LRSchedule = _module
Model = _module
cnn_utils = _module
evaluate = _module
main = _module
utils = _module
DataSet = _module
IOUEval = _module
Transforms = _module
Model = _module
SegmentationModel = _module
cnn_utils = _module
gen_cityscapes = _module
loadData = _module
main = _module
train_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch.nn import init


import torch.nn.functional as F


import math


import torch


import torch.nn as nn


import time


from torch.autograd import Variable


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import numpy as np


import torchvision.models as preModels


import random


from torch import nn


import torch.optim.lr_scheduler


class EESP(nn.Module):
    """
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        """
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'
            ], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(
            n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13
            ): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride,
                groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == input.size():
            expanded = expanded + input
        return self.module_act(expanded)


class DownSampler(nn.Module):
    """
    Down-sampling fucntion that has three parallel branches: (1) avg pooling,
    (2) EESP block with stride of 2 and (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    """

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        """
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        """
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim,
            down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(CBR(config_inp_reinf,
                config_inp_reinf, 3, 1), CB(config_inp_reinf, nout, 1, 1))
        self.act = nn.PReLU(nout)

    def forward(self, input, input2=None):
        """
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        """
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)
        if input2 is not None:
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1,
                    stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)
        return self.act(output)


class EESPNet(nn.Module):
    """
    This class defines the ESPNetv2 architecture for the ImageNet classification
    """

    def __init__(self, classes=1000, s=1):
        """
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        """
        super().__init__()
        reps = [0, 3, 7, 3]
        channels = 3
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s <= 2.0:
            config.append(1280)
        else:
            ValueError('Configuration not supported')
        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim
            ), 'Length of branching factor array and receptive field array should be the same.'
        self.level1 = CBR(channels, config[0], 3, 2)
        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=
            r_lim[0], reinf=self.input_reinforcement)
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=
            r_lim[1], reinf=self.input_reinforcement)
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2],
                r_lim=r_lim[2]))
        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=
            r_lim[2], reinf=self.input_reinforcement)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3],
                r_lim=r_lim[3]))
        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=
            r_lim[3])
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4],
                r_lim=r_lim[4]))
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))
        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        """
        Function to initialze the parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        """
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        """
        out_l1 = self.level1(input)
        if not self.input_reinforcement:
            del input
            input = None
        out_l2 = self.level2_0(out_l1, input)
        out_l3_0 = self.level3_0(out_l2, input)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, input)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        out_l5_0 = self.level5_0(out_l4)
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)
        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)
        return self.classifier(output_1x1)


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilatedB(nn.Module):
    """
    This class defines the dilated convolution with batch normalization.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        return self.bn(self.conv(input))


class EESP(nn.Module):
    """
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param g: number of groups to be used in the feature map reduction step.
        """
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'
            ], 'One of these is suppported (avg or esp)'
        assert n == n1, 'n(={}) and n1(={}) should be equal for Depth-wise Convolution '.format(
            n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)
        map_receptive_ksize = {(3): 1, (5): 2, (7): 3, (9): 4, (11): 5, (13
            ): 6, (15): 7, (17): 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride,
                groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded
        if expanded.size() == input.size():
            expanded = expanded + input
        return self.module_act(expanded)


class DownSampler(nn.Module):
    """
    Down-sampling fucntion that has two parallel branches: (1) avg pooling
    and (2) EESP block with stride of 2. The output feature maps of these branches
    are then concatenated and thresholded using an activation function (PReLU in our
    case) to produce the final output.
    """

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        """
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param g: number of groups to be used in the feature map reduction step.
        """
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim,
            down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(CBR(config_inp_reinf,
                config_inp_reinf, 3, 1), CB(config_inp_reinf, nout, 1, 1))
        self.act = nn.PReLU(nout)

    def forward(self, input, input2=None):
        """
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        """
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)
        if input2 is not None:
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1,
                    stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)
        return self.act(output)


class EESPNet(nn.Module):
    """
    This class defines the ESPNetv2 architecture for the ImageNet classification
    """

    def __init__(self, classes=20, s=1):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param s: factor that scales the number of output feature maps
        """
        super().__init__()
        reps = [0, 3, 7, 3]
        channels = 3
        r_lim = [13, 11, 9, 7, 5]
        K = [4] * len(r_lim)
        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s in [1.5, 2]:
            config.append(1280)
        else:
            ValueError('Configuration not supported')
        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim
            ), 'Length of branching factor array and receptive field array should be the same.'
        self.level1 = CBR(channels, config[0], 3, 2)
        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=
            r_lim[0], reinf=self.input_reinforcement)
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=
            r_lim[1], reinf=self.input_reinforcement)
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, k=K[2],
                r_lim=r_lim[2]))
        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=
            r_lim[2], reinf=self.input_reinforcement)
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, k=K[3],
                r_lim=r_lim[3]))
        self.level5_0 = DownSampler(config[3], config[4], k=K[3], r_lim=
            r_lim[3])
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, k=K[4],
                r_lim=r_lim[4]))
        self.level5.append(CBR(config[4], config[4], 3, 1, groups=config[4]))
        self.level5.append(CBR(config[4], config[5], 1, 1, groups=K[4]))
        self.classifier = nn.Linear(config[5], classes)
        self.init_params()

    def init_params(self):
        """
        Function to initialze the parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2, seg=True):
        """
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        """
        out_l1 = self.level1(input)
        if not self.input_reinforcement:
            del input
            input = None
        out_l2 = self.level2_0(out_l1, input)
        out_l3_0 = self.level3_0(out_l2, input)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, input)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        if not seg:
            out_l5_0 = self.level5_0(out_l4)
            for i, layer in enumerate(self.level5):
                if i == 0:
                    out_l5 = layer(out_l5_0)
                else:
                    out_l5 = layer(out_l5)
            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
            output_g = F.dropout(output_g, p=p, training=self.training)
            output_1x1 = output_g.view(output_g.size(0), -1)
            return self.classifier(output_1x1)
        return out_l1, out_l2, out_l3, out_l4


class EESPNet_Seg(nn.Module):

    def __init__(self, classes=20, s=1, pretrained=None, gpus=1):
        super().__init__()
        classificationNet = EESPNet(classes=1000, s=s)
        if gpus >= 1:
            classificationNet = nn.DataParallel(classificationNet)
        if pretrained:
            if not os.path.isfile(pretrained):
                None
            None
            classificationNet.load_state_dict(torch.load(pretrained))
        self.net = classificationNet.module
        del classificationNet
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
        if s <= 0.5:
            p = 0.1
        else:
            p = 0.2
        self.proj_L4_C = CBR(self.net.level4[-1].module_act.num_parameters,
            self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2 * self.net.level3[-1].module_act.num_parameters
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize // 2, stride=1, k
            =4, r_lim=7), PSPModule(pspSize // 2, pspSize // 2))
        self.project_l3 = nn.Sequential(nn.Dropout2d(p=p), C(pspSize // 2,
            classes, 1, 1))
        self.act_l3 = BR(classes)
        self.project_l2 = CBR(self.net.level2_0.act.num_parameters +
            classes, classes, 1, 1)
        self.project_l1 = nn.Sequential(nn.Dropout2d(p=p), C(self.net.
            level1.act.num_parameters + classes, classes, 1, 1))

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        return x

    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode=
            'bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode=
            'bilinear', align_corners=True)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear',
            align_corners=True)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))
        if self.training:
            return F.interpolate(merge_l1, scale_factor=2, mode='bilinear',
                align_corners=True), self.hierarchicalUpsample(
                proj_merge_l3_bef_act)
        else:
            return F.interpolate(merge_l1, scale_factor=2, mode='bilinear',
                align_corners=True)


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([C(features, features, 3, 1, groups=
            features) for size in sizes])
        self.project = CBR(features * (len(sizes) + 1), out_features, 1, 1)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(input=stage(feats), size=(h, w), mode
                ='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(torch.cat(out, dim=1))


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
        This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
       This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        """

        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilatedB(nn.Module):
    """
    This class defines the dilated convolution with batch normalization.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=
            padding, bias=False, dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        return self.bn(self.conv(input))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sacmehta_ESPNetv2(_paritybench_base):
    pass
    def test_000(self):
        self._check(BR(*[], **{'nOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(C(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(CB(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(CBR(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(CDilated(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(CDilatedB(*[], **{'nIn': 4, 'nOut': 4, 'kSize': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(EESP(*[], **{'nIn': 64, 'nOut': 64}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_007(self):
        self._check(EESPNet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_008(self):
        self._check(EESPNet_Seg(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_009(self):
        self._check(PSPModule(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

