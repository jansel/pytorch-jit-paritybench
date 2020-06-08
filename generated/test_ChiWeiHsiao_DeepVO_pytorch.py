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


import math


import numpy as np


import torch


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.nn.modules import loss


from torch import functional as F


import torch.nn as nn


from torch.nn.init import kaiming_normal_


from torch.nn.init import orthogonal_


class DeepvoLoss(loss._Loss):

    def __init__(self, size_average=True, reduce=True):
        super(DeepvoLoss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input[0:3], target[0:3], size_average=self.
            size_average, reduce=self.reduce) + 100 * F.mse_loss(input[3:6],
            target[3:6], size_average=self.size_average, reduce=self.reduce)


class Parameters:

    def __init__(self):
        self.n_processors = 8
        self.data_dir = (
            '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/chsiao/KITTI/'
            )
        self.image_dir = self.data_dir + '/images/'
        self.pose_dir = self.data_dir + '/pose_GT/'
        self.train_video = ['00', '01', '02', '05', '08', '09']
        self.valid_video = ['04', '06', '07', '10']
        self.partition = None
        self.resize_mode = 'rescale'
        self.img_w = 608
        self.img_h = 184
        self.img_means = (0.19007764876619865, 0.15170388157131237, 
            0.10659445665650864)
        self.img_stds = (0.2610784009469139, 0.25729316928935814, 
            0.25163823815039915)
        self.minus_point_5 = True
        self.seq_len = 5, 7
        self.sample_times = 3
        self.train_data_info_path = (
            'datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format
            (''.join(self.train_video), ''.join(self.valid_video), self.
            partition, self.seq_len[0], self.seq_len[1], self.sample_times))
        self.valid_data_info_path = (
            'datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format
            (''.join(self.train_video), ''.join(self.valid_video), self.
            partition, self.seq_len[0], self.seq_len[1], self.sample_times))
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
        self.load_model_path = (
            'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.
            join(self.train_video), ''.join(self.valid_video), self.img_h,
            self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size,
            self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.
            optim.items()]), self.resume_t_or_v))
        self.load_optimizer_path = (
            'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format
            (''.join(self.train_video), ''.join(self.valid_video), self.
            img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.
            batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k,
            v in self.optim.items()]), self.resume_t_or_v))
        self.record_path = ('records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'
            .format(''.join(self.train_video), ''.join(self.valid_video),
            self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.
            batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k,
            v in self.optim.items()])))
        self.save_model_path = (
            'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.
            join(self.train_video), ''.join(self.valid_video), self.img_h,
            self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size,
            self.rnn_hidden_size, '_'.join([(k + str(v)) for k, v in self.
            optim.items()])))
        self.save_optimzer_path = (
            'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(
            ''.join(self.train_video), ''.join(self.valid_video), self.
            img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.
            batch_size, self.rnn_hidden_size, '_'.join([(k + str(v)) for k,
            v in self.optim.items()])))
        if not os.path.isdir(os.path.dirname(self.record_path)):
            os.makedirs(os.path.dirname(self.record_path))
        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))
        if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
            os.makedirs(os.path.dirname(self.train_data_info_path))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ChiWeiHsiao_DeepVO_pytorch(_paritybench_base):
    pass
