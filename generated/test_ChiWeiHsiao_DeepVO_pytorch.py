import sys
_module = sys.modules[__name__]
del sys
Dataloader_loss = _module
data_helper = _module
helper = _module
main = _module
model = _module
params = _module
preprocess = _module
test = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import numpy as np


import time


import torch


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.nn.modules import loss


from torch import functional as F


from torch.utils.data.sampler import Sampler


from torchvision import transforms


import torch.nn as nn


from torch.nn.init import kaiming_normal_


from torch.nn.init import orthogonal_


class DeepvoLoss(loss._Loss):

    def __init__(self, size_average=True, reduce=True):
        super(DeepvoLoss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input[0:3], target[0:3], size_average=self.size_average, reduce=self.reduce) + 100 * F.mse_loss(input[3:6], target[3:6], size_average=self.size_average, reduce=self.reduce)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout))


class Parameters:

    def __init__(self):
        self.n_processors = 8
        self.data_dir = '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/chsiao/KITTI/'
        self.image_dir = self.data_dir + '/images/'
        self.pose_dir = self.data_dir + '/pose_GT/'
        self.train_video = ['00', '01', '02', '05', '08', '09']
        self.valid_video = ['04', '06', '07', '10']
        self.partition = None
        self.resize_mode = 'rescale'
        self.img_w = 608
        self.img_h = 184
        self.img_means = 0.19007764876619865, 0.15170388157131237, 0.10659445665650864
        self.img_stds = 0.2610784009469139, 0.25729316928935814, 0.25163823815039915
        self.minus_point_5 = True
        self.seq_len = 5, 7
        self.sample_times = 3
        self.train_data_info_path = 'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
        self.valid_data_info_path = 'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
        self.rnn_hidden_size = 1000
        self.conv_dropout = 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0
        self.clip = None
        self.batch_norm = True
        self.epochs = 250
        self.batch_size = 8
        self.pin_mem = True
        self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
        self.pretrained_flownet = None
        self.resume = True
        self.resume_t_or_v = '.train'
        self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.optim.items()]))
        self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.optim.items()]))
        if not os.path.isdir(os.path.dirname(self.record_path)):
            os.makedirs(os.path.dirname(self.record_path))
        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))
        if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
            os.makedirs(os.path.dirname(self.train_data_info_path))


class DeepVO(nn.Module):

    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO, self).__init__()
        self.batchNorm = batchNorm
        self.clip = par.clip
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6 = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        self.rnn = nn.LSTM(input_size=int(np.prod(__tmp.size())), hidden_size=par.rnn_hidden_size, num_layers=2, dropout=par.rnn_dropout_between, batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                kaiming_normal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.0)
                kaiming_normal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = 100 * angle_loss + translation_loss
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip != None:
            torch.nn.utils.clip_grad_norm(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss

