import sys
_module = sys.modules[__name__]
del sys
util = _module
demo_data_preprocess = _module
demo_decayKernels = _module
demo_featureHawkes = _module
demo_featureHawkes2 = _module
demo_featureHawkes3 = _module
demo_linearHawkes = _module
demo_mixHawkes = _module
demo_nonlinearHawkes = _module
DecayKernel = _module
DecayKernelFamily = _module
EndogenousImpact = _module
EndogenousImpactFamily = _module
ExogenousIntensity = _module
ExogenousIntensityFamily = _module
HawkesProcess = _module
MixHawkesProcess = _module
OtherLayers = _module
PointProcess = _module
DataIO = _module
DataOperation = _module

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


import torch


import torch.optim as optim


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


from typing import Optional


import matplotlib.pyplot as plt


import copy


import torch.nn as nn


from typing import Dict


import time


import torch.nn.functional as F


from typing import List


from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class BasicEndogenousImpact(nn.Module):
    """
    The parent class of endogenous impact functions sum_i phi_{kk_i}(t-t_i) for k = 1,...,C,
    which actually a simple endogenous impact with phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t)
    """

    def __init__(self, num_type: int, kernel):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t),
        for m = 1, ..., M, A_m = [a_{kk'm}] in R^{C*C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        """
        super(BasicEndogenousImpact, self).__init__()
        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.endogenous_impact_type = "sum_m a_(kk'm) * kernel_m(t)"
        self.num_type = num_type
        self.dim_embedding = num_type
        for m in range(self.num_base):
            emb = nn.Embedding(self.num_type, self.dim_embedding)
            emb.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)

    def print_info(self):
        """
        Print basic information of the exogenous intensity function.
        """
        logger.info("Endogenous impact function: phi_(kk')(t) = {}.".format(self.endogenous_impact_type))
        logger.info('The number of event types = {}.'.format(self.num_type))
        self.decay_kernel.print_info()

    def intensity(self, sample_dict: Dict):
        """
        Calculate intensity of event
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c} sum_{i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        events = sample_dict['ci']
        history = sample_dict['cjs']
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        gt = self.decay_kernel.values(dts.numpy())
        gt = torch.from_numpy(gt)
        gt = gt.type(torch.FloatTensor)
        phi_c = 0
        for m in range(self.num_base):
            A_cm = self.basis[m](events)
            A_cm = A_cm.squeeze(1)
            A_cm = A_cm.gather(1, history)
            A_cm = A_cm.unsqueeze(1)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        sum_i int_{0}^{dt_i} phi_cc_i(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c} sum_{i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        history = sample_dict['cjs']
        all_types = sample_dict['Cs']
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        t_start = history_time[:, -1].repeat(1, history_time.size(1)) - history_time
        t_stop = dts
        Gt = self.decay_kernel.integrations(t_stop.numpy(), t_start.numpy())
        Gt = torch.from_numpy(Gt)
        Gt = Gt.type(torch.FloatTensor)
        history2 = history.unsqueeze(1).repeat(1, all_types.size(0), 1)
        pHi = 0
        for m in range(self.num_base):
            A_all = self.basis[m](all_types)
            A_all = A_all.squeeze(1).unsqueeze(0)
            A_all = A_all.repeat(Gt.size(0), 1, 1)
            A_all = A_all.gather(2, history2)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        all_types = sample_dict['Cs']
        A_all = 0
        for m in range(self.num_base):
            A_tmp = self.basis[m](all_types)
            A_tmp = torch.transpose(A_tmp, 1, 2)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all

    def forward(self, sample_dict: Dict):
        """
        Calculate
        1) phi_{c_i,c_j}(t_i - t_j) for c_i in "events";
        2) int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c} sum_{i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        phi_c = self.intensity(sample_dict)
        pHi = self.expect_counts(sample_dict)
        return phi_c, pHi

    def plot_and_save(self, infect: torch.Tensor, output_name: str=None):
        """
        Plot endogenous impact function for all event types
        Args:
        :param infect: a (num_type, num_type+1, M) FloatTensor containing all endogenous impact
        :param output_name: the name of the output png file
        """
        impact = infect.sum(2).data.cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(impact)
        plt.colorbar()
        if output_name is None:
            plt.savefig('endogenous_impact.png')
        else:
            plt.savefig(output_name)
        plt.close('all')
        logger.info('Done!')


class Identity(nn.Module):
    """
    An identity layer f(x) = x
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NaiveEndogenousImpact(BasicEndogenousImpact):
    """
    The class of naive endogenous impact functions sum_i phi_{kk_i}(t-t_i) for k = 1,...,C,
    which actually a simple endogenous impact with phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t)
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t),
        for m = 1, ..., M, A_m = [a_{kk'm}] in R^{C*C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')}
        """
        super(NaiveEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        if activation is None:
            self.endogenous_impact_type = "sum_m a_(kk'm) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(a_(kk'm)) * kernel_m(t))".format(activation)
            self.activation = activation
        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = num_type
        for m in range(self.num_base):
            emb = nn.Embedding(self.num_type, self.dim_embedding)
            emb.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of events
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        events = sample_dict['ci']
        history = sample_dict['cjs']
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        gt = self.decay_kernel.values(dts)
        phi_c = 0
        for m in range(self.num_base):
            A_cm = self.basis[m](events)
            A_cm = A_cm.squeeze(1)
            A_cm = A_cm.gather(1, history)
            A_cm = self.act(A_cm)
            A_cm = A_cm.unsqueeze(1)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        history = sample_dict['cjs']
        all_types = sample_dict['Cs']
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time
        t_stop = dts
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        pHi = 0
        history2 = history.unsqueeze(1).repeat(1, all_types.size(0), 1)
        for m in range(self.num_base):
            A_all = self.basis[m](all_types)
            A_all = A_all.squeeze(1).unsqueeze(0)
            A_all = A_all.repeat(Gt.size(0), 1, 1)
            A_all = A_all.gather(2, history2)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        all_types = sample_dict['Cs']
        A_all = 0
        for m in range(self.num_base):
            A_tmp = self.basis[m](all_types)
            A_tmp = self.act(torch.transpose(A_tmp, 1, 2))
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class FactorizedEndogenousImpact(BasicEndogenousImpact):
    """
    The class of factorized endogenous impact functions
    phi_{cc'}(t) = sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)
    Here, U_m=[u_{cm}] and V_m=[v_{cm}], m=1,...,M, are embedding matrices
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_{m} a_{kk'm} kernel_m(t),
        for m = 1, ..., M, A_m = [a_{kk'm}] in R^{C*C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)}
        """
        super(FactorizedEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        dim_embedding = parameter_set['dim_embedding']
        if activation is None:
            self.endogenous_impact_type = "sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(u_(cm)^T * v_(c'm)) * kernel_m(t))".format(activation)
            self.activation = activation
        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type_u = num_type
        self.num_type_v = num_type
        self.dim_embedding = dim_embedding
        for m in range(self.num_base):
            emb_u = nn.Embedding(self.num_type_u, self.dim_embedding)
            emb_v = nn.Embedding(self.num_type_v, self.dim_embedding)
            emb_u.weight = nn.Parameter(torch.FloatTensor(self.num_type_u, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
            emb_v.weight = nn.Parameter(torch.FloatTensor(self.num_type_v, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
            if m == 0:
                self.basis_u = nn.ModuleList([emb_u])
                self.basis_v = nn.ModuleList([emb_v])
            else:
                self.basis_u.append(emb_u)
                self.basis_v.append(emb_v)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of event
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        events = sample_dict['ci']
        history = sample_dict['cjs']
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        gt = self.decay_kernel.values(dts)
        phi_c = 0
        for m in range(self.num_base):
            u_cm = self.basis_u[m](events)
            v_cm = self.basis_v[m](history)
            v_cm = torch.transpose(v_cm, 1, 2)
            A_cm = torch.bmm(u_cm, v_cm)
            A_cm = self.act(A_cm)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        history = sample_dict['cjs']
        all_types = sample_dict['Cs']
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time
        t_stop = dts
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        pHi = 0
        for m in range(self.num_base):
            v_cm = self.basis_v[m](history)
            v_cm = torch.transpose(v_cm, 1, 2)
            u_all = self.basis_u[m](all_types)
            u_all = torch.transpose(u_all, 0, 1)
            u_all = u_all.repeat(Gt.size(0), 1, 1)
            A_all = torch.matmul(u_all, v_cm)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        A_all = 0
        all_types = sample_dict['Cs'][:, 0]
        for m in range(self.num_base):
            u_all = self.basis_u[m](all_types)
            v_all = self.basis_v[m](all_types)
            A_tmp = torch.matmul(u_all, torch.t(v_all)).unsqueeze(2)
            A_tmp = self.act(A_tmp)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class LinearEndogenousImpact(BasicEndogenousImpact):
    """
    The class of linear endogenous impact functions
    phi_{cc'}(t) = sum_m (w_{cm}^T * f_{c'}) * kernel_m(t)
    Here W_m = [w_{cm}], for m=1,...,M, are embedding matrices
    f_{c'} is the feature vector associated with the c'-th history event
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{kk'}(t) = sum_m (w_{cm}^T * f_{c'}) * kernel_m(t),
        for m = 1, ..., M, W_m = [w_{cm}] in R^{(C+1)*D}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)}
        """
        super(LinearEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        dim_feature = parameter_set['dim_feature']
        if activation is None:
            self.endogenous_impact_type = "sum_m (u_{cm}^T * v_{c'm}) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(u_(cm)^T * v_(c'm)) * kernel_m(t))".format(activation)
            self.activation = activation
        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = dim_feature
        for m in range(self.num_base):
            emb = nn.Embedding(self.num_type, self.dim_embedding)
            emb.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)
        self.emb_event = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb_event.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of events
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        events = sample_dict['ci']
        history = sample_dict['cjs']
        history_feat = sample_dict['fcjs']
        if history_feat is None:
            history_feat = self.emb_event(history)
            history_feat = torch.transpose(history_feat, 1, 2)
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        gt = self.decay_kernel.values(dts)
        phi_c = 0
        for m in range(self.num_base):
            u_cm = self.basis[m](events)
            A_cm = torch.bmm(u_cm, history_feat)
            A_cm = self.act(A_cm)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        history = sample_dict['cjs']
        all_types = sample_dict['Cs']
        history_feat = sample_dict['fcjs']
        if history_feat is None:
            history_feat = self.emb_event(history)
            history_feat = torch.transpose(history_feat, 1, 2)
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time
        t_stop = dts
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        pHi = 0
        for m in range(self.num_base):
            u_all = self.basis[m](all_types)
            u_all = u_all.squeeze(1)
            A_all = torch.matmul(u_all, history_feat)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'FCs': all_features (num_type, dim_feature)
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        A_all = 0
        all_types = sample_dict['Cs'][:, 0]
        all_features = sample_dict['FCs']
        if all_features is None:
            all_features = self.emb_event(all_types)
        for m in range(self.num_base):
            u_all = self.basis[m](all_types)
            A_tmp = torch.matmul(u_all, torch.t(all_features)).unsqueeze(2)
            A_tmp = self.act(A_tmp)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class BilinearEndogenousImpact(BasicEndogenousImpact):
    """
    The class of bilinear endogenous impact functions
    phi_{cc'}(t) = sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)
    Here W_m for m=1,...,M, are embedding matrices
    f_{c'} is the feature vector associated with the c'-th history event
    """

    def __init__(self, num_type: int, kernel, parameter_set: Dict):
        """
        Initialize endogenous impact: phi_{cc'}(t) = sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)
        for m = 1, ..., M, W_m = [w_{cm}] in R^{(C+1)*D}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param kernel: an instance of a decay kernel class in "DecayKernelFamily"
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)}
        """
        super(BilinearEndogenousImpact, self).__init__(num_type, kernel)
        activation = parameter_set['activation']
        dim_feature = parameter_set['dim_feature']
        if activation is None:
            self.endogenous_impact_type = "sum_m (f_{c}^T * W_m * f_{c'}) * kernel_m(t)"
            self.activation = 'identity'
        else:
            self.endogenous_impact_type = "sum_m {}(f_(c)^T * W_m * f_(c')) * kernel_m(t))".format(activation)
            self.activation = activation
        self.decay_kernel = kernel
        self.num_base = self.decay_kernel.parameters.shape[1]
        self.num_type = num_type
        self.dim_embedding = dim_feature
        for m in range(self.num_base):
            emb = nn.Linear(self.dim_embedding, self.dim_embedding, bias=False)
            emb.weight = nn.Parameter(torch.FloatTensor(self.dim_embedding, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
            if m == 0:
                self.basis = nn.ModuleList([emb])
            else:
                self.basis.append(emb)
        self.emb_event = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb_event.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def intensity(self, sample_dict: Dict):
        """
        Calculate the intensity of event
        phi_{c_i,c_j}(t_i - t_j) for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'fci': current_feature (batch_size, Dc) FloatTensor of current feature
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        events = sample_dict['ci']
        history = sample_dict['cjs']
        current_feat = sample_dict['fci']
        history_feat = sample_dict['fcjs']
        if history_feat is None:
            current_feat = self.emb_event(events)
            current_feat = current_feat.squeeze(1)
            history_feat = self.emb_event(history)
            history_feat = torch.transpose(history_feat, 1, 2)
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        gt = self.decay_kernel.values(dts)
        phi_c = 0
        for m in range(self.num_base):
            u_cm = self.basis[m](current_feat)
            u_cm = u_cm.unsqueeze(1)
            A_cm = torch.bmm(u_cm, history_feat)
            A_cm = self.act(A_cm)
            phi_c += torch.bmm(A_cm, gt[:, :, m].unsqueeze(2))
        phi_c = phi_c[:, :, 0]
        return phi_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            'FCs': all_feats (num_type, dim_feature) FloatTensor of all event types
            }
        :return:
            phi_c: (batch_size, 1) FloatTensor represents phi_{c_i, c_j}(t_i - t_j);
            pHi: (batch_size, num_type) FloatTensor represents sum_{c, i in history} int_{start}^{stop} phi_cc_i(s)ds
        """
        event_time = sample_dict['ti']
        history_time = sample_dict['tjs']
        history = sample_dict['cjs']
        all_types = sample_dict['Cs']
        all_feats = sample_dict['FCs']
        history_feat = sample_dict['fcjs']
        if history_feat is None:
            all_feats = self.emb_event(all_types)
            all_feats = all_feats.squeeze(1)
            history_feat = self.emb_event(history)
            history_feat = torch.transpose(history_feat, 1, 2)
        dts = event_time.repeat(1, history_time.size(1)) - history_time
        last_time = history_time[:, -1].unsqueeze(1)
        t_start = last_time.repeat(1, history_time.size(1)) - history_time
        t_stop = dts
        Gt = self.decay_kernel.integrations(t_stop, t_start)
        pHi = 0
        for m in range(self.num_base):
            u_all = self.basis[m](all_feats)
            A_all = torch.matmul(u_all, history_feat)
            A_all = self.act(A_all)
            pHi += torch.bmm(A_all, Gt[:, :, m].unsqueeze(2))
        pHi = pHi[:, :, 0]
        return pHi

    def granger_causality(self, sample_dict: Dict):
        """
        Calculate the granger causality among event types
        a_{cc'm}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'FCs': all_features (num_type, dim_feature)
            }
        :return:
            A_all: (num_type, num_type, num_base) FloatTensor represents a_{cc'm} in phi_{cc'}(t)
        """
        A_all = 0
        all_types = sample_dict['Cs'][:, 0]
        all_features = sample_dict['FCs']
        if all_features is None:
            all_features = self.emb_event(all_types)
        for m in range(self.num_base):
            u_all = self.basis[m](all_features)
            A_tmp = torch.matmul(u_all, torch.t(all_features)).unsqueeze(2)
            A_tmp = self.act(A_tmp)
            if m == 0:
                A_all = A_tmp
            else:
                A_all = torch.cat([A_all, A_tmp], dim=2)
        return A_all


class BasicExogenousIntensity(nn.Module):
    """
    The parent class of exogenous intensity function mu(t), which actually a constant exogenous intensity
    """

    def __init__(self, num_type: int):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        """
        super(BasicExogenousIntensity, self).__init__()
        self.exogenous_intensity_type = 'constant'
        self.activation = 'identity'
        self.num_type = num_type
        self.dim_embedding = 1
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))

    def print_info(self):
        """
        Print basic information of the exogenous intensity function.
        """
        logger.info('Exogenous intensity function: mu(t) = {}.'.format(self.exogenous_intensity_type))
        logger.info('The number of event types = {}.'.format(self.num_type))

    def forward(self, sample_dict: Dict):
        """
        Calculate
        1) mu_{c_i} for c_i in "events";
        2) int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        mu_c = self.intensity(sample_dict)
        mU = self.expect_counts(sample_dict)
        return mu_c, mU

    def intensity(self, sample_dict: Dict):
        """
        Calculate intensity mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
        """
        events = sample_dict['ci']
        mu_c = self.emb(events)
        mu_c = mu_c.squeeze(1)
        return mu_c

    def expect_counts(self, sample_dict: Dict):
        """
        Calculate expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']
        mu_all = self.emb(all_types)
        mu_all = mu_all.squeeze(1)
        mU = torch.matmul(dts, torch.t(mu_all))
        return mU

    def plot_and_save(self, mu_all: torch.Tensor, output_name: str=None):
        """
        Plot the stem plot of exogenous intensity functions for all event types
        Args:
        :param mu_all: a (num_type, 1) FloatTensor containing all exogenous intensity functions
        :param output_name: the name of the output png file
        """
        mu_all = mu_all.squeeze(1)
        mu_all = mu_all.data.cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.stem(range(mu_all.shape[0]), mu_all, '-')
        plt.ylabel('Exogenous intensity')
        plt.xlabel('Index of event type')
        if output_name is None:
            plt.savefig('exogenous_intensity.png')
        else:
            plt.savefig(output_name)
        plt.close('all')
        logger.info('Done!')


class NaiveExogenousIntensity(BasicExogenousIntensity):
    """
    The class of constant exogenous intensity function mu(t) = mu
    """

    def __init__(self, num_type: int, parameter_set: Dict=None):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')}
        """
        super(NaiveExogenousIntensity, self).__init__(num_type)
        activation = parameter_set['activation']
        if activation is None:
            self.exogenous_intensity_type = 'constant'
            self.activation = 'identity'
        else:
            self.exogenous_intensity_type = '{}(constant)'.format(activation)
            self.activation = activation
        self.num_type = num_type
        self.dim_embedding = 1
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def intensity(self, sample_dict):
        """
        Calculate intensity
        mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
        """
        events = sample_dict['ci']
        mu_c = self.act(self.emb(events))
        mu_c = mu_c.squeeze(1)
        return mu_c

    def expect_counts(self, sample_dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            }
        :return:
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']
        mu_all = self.act(self.emb(all_types))
        mu_all = mu_all.squeeze(1)
        mU = torch.matmul(dts, torch.t(mu_all))
        return mU


class LinearExogenousIntensity(BasicExogenousIntensity):
    """
    The class of linear exogenous intensity function mu_c(t) = w_c^T * f.
    Here f is nonnegative feature vector of a sequence.
    """

    def __init__(self, num_type: int, parameter_set: Dict):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'activation': value = names of activation layers ('identity', 'relu', 'softplus')
                             'dim_feature': value = the dimension of feature vector (embedding)
                             'num_sequence': the number of sequence}
        """
        super(LinearExogenousIntensity, self).__init__(num_type)
        activation = parameter_set['activation']
        dim_feature = parameter_set['dim_feature']
        num_seq = parameter_set['num_sequence']
        if activation is None:
            self.exogenous_intensity_type = 'w_c^T*f'
            self.activation = 'identity'
        else:
            self.exogenous_intensity_type = '{}(w_c^T*f)'.format(activation)
            self.activation = activation
        self.num_type = num_type
        self.dim_embedding = dim_feature
        self.num_seq = num_seq
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(torch.FloatTensor(self.num_type, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        self.emb_seq = nn.Embedding(self.num_seq, self.dim_embedding)
        self.emb_seq.weight = nn.Parameter(torch.FloatTensor(self.num_seq, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def intensity(self, sample_dict):
        """
        Calculate intensity
        mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn' features: (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        events = sample_dict['ci']
        features = sample_dict['fsn']
        if features is None:
            features = self.emb_seq(sample_dict['sn'])
            features = features.squeeze(1)
        mu_c = self.emb(events)
        mu_c = mu_c.squeeze(1)
        mu_c = mu_c * features
        mu_c = mu_c.sum(1)
        mu_c = mu_c.view(-1, 1)
        mu_c = self.act(mu_c)
        return mu_c

    def expect_counts(self, sample_dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn' features: (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']
        features = sample_dict['fsn']
        if features is None:
            features = self.emb_seq(sample_dict['sn'])
            features = features.squeeze(1)
        mu_all = self.emb(all_types)
        mu_all = mu_all.squeeze(1)
        mu_all = torch.matmul(features, torch.t(mu_all))
        mu_all = self.act(mu_all)
        mU = mu_all * dts.repeat(1, mu_all.size(1))
        return mU


class NeuralExogenousIntensity(BasicExogenousIntensity):
    """
    The class of neural exogenous intensity function mu_c(t) = F(c, f), where F is a 3-layer neural network,
    c is event type, and f is the feature vector.
    Here, we don't need to ensure f to be nonnegative.
    """

    def __init__(self, num_type: int, parameter_set: Dict):
        """
        Initialize exogenous intensity function: mu(t) = mu, mu in R^{C+1}, C is the number of event type
        :param num_type: for a point process with C types of events, num_type = C+1, in which the first type "0"
                         corresponds to an "empty" type never appearing in the sequence.
        :param parameter_set: a dictionary containing parameters
            parameter_set = {'dim_embedding': the dimension of embeddings.
                             'dim_feature': the dimension of feature vector.
                             'dim_hidden': the dimension of hidden vector.
                             'num_sequence': the number of sequence}
        """
        super(NeuralExogenousIntensity, self).__init__(num_type)
        dim_embedding = parameter_set['dim_embedding']
        dim_feature = parameter_set['dim_feature']
        dim_hidden = parameter_set['dim_hidden']
        num_seq = parameter_set['num_sequence']
        self.exogenous_intensity_type = 'F(c, f)'
        self.num_type = num_type
        self.dim_embedding = dim_embedding
        self.dim_feature = dim_feature
        self.dim_hidden = dim_hidden
        self.num_seq = num_seq
        self.emb = nn.Embedding(self.num_type, self.dim_embedding)
        self.emb.weight = nn.Parameter(torch.FloatTensor(self.num_type - 1, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        self.emb_seq = nn.Embedding(self.num_seq, self.dim_embedding)
        self.emb_seq.weight = nn.Parameter(torch.FloatTensor(self.num_seq, self.dim_embedding).uniform_(0.01 / self.dim_embedding, 1 / self.dim_embedding))
        self.softplus = nn.Softplus()
        self.linear1 = nn.Linear(self.dim_embedding, self.dim_hidden)
        self.linear2 = nn.Linear(self.dim_feature, self.dim_hidden)
        self.linear3 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.relu = nn.ReLU()

    def intensity(self, sample_dict):
        """
        Calculate the intensity of event
        mu_{c_i} for c_i in "events";

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn': features (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mu_c: (batch_size, 1) FloatTensor represents mu_{c_i};
        """
        events = sample_dict['ci']
        features = sample_dict['fsn']
        if features is None:
            features = self.emb_seq(sample_dict['sn'])
            features = features.squeeze(1)
        event_feat = self.emb(events)
        event_feat = event_feat.squeeze(1)
        event_feat = self.relu(self.linear1(event_feat))
        seq_feat = self.relu(self.linear2(features))
        seq_feat = self.linear3(seq_feat)
        feat = seq_feat * event_feat
        mu_c = self.softplus(feat.sum(1).view(-1, 1))
        return mu_c

    def expect_counts(self, sample_dict):
        """
        Calculate the expected number of events in dts
        int_{0}^{dt_i} mu_c(s)ds for dt_i in "dts" and c in {1, ..., num_type}

        :param sample_dict is a dictionary contains a batch of samples
        sample_dict = {
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'Cs': all_types (num_type, 1) LongTensor indicates all event types
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn': features (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            }
        :return:
            mU: (batch_size, num_type) FloatTensor represents int_{0}^{dt} mu_c(s)ds
        """
        dts = sample_dict['ti'] - sample_dict['tjs'][:, -1].view(-1, 1)
        all_types = sample_dict['Cs']
        features = sample_dict['fsn']
        if features is None:
            features = self.emb_seq(sample_dict['sn'])
            features = features.squeeze(1)
        seq_feat = self.relu(self.linear2(features))
        seq_feat = self.linear3(seq_feat)
        event_feat_all = self.emb(all_types)
        event_feat_all = event_feat_all.squeeze(1)
        event_feat_all = self.relu(self.linear1(event_feat_all))
        mu_all = torch.matmul(seq_feat, torch.t(event_feat_all))
        mu_all = self.softplus(mu_all)
        mU = mu_all * dts.repeat(1, mu_all.size(1))
        return mU


class HawkesProcessIntensity(nn.Module):
    """
    The class of inhomogeneous Poisson process
    """

    def __init__(self, exogenous_intensity, endogenous_intensity, activation: str=None):
        super(HawkesProcessIntensity, self).__init__()
        self.exogenous_intensity = exogenous_intensity
        self.endogenous_intensity = endogenous_intensity
        if activation is None:
            self.intensity_type = 'exogenous intensity + endogenous impacts'
            self.activation = 'identity'
        else:
            self.intensity_type = '{}(exogenous intensity + endogenous impacts)'.format(activation)
            self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'softplus':
            self.act = nn.Softplus(beta=self.num_type ** 0.5)
        elif self.activation == 'identity':
            self.act = Identity()
        else:
            logger.warning('The actvation layer is {}, which can not be identified... '.format(self.activation))
            logger.warning('Identity activation is applied instead.')
            self.act = Identity()

    def print_info(self):
        logger.info('A generalized Hawkes process intensity:')
        logger.info('Intensity function lambda(t) = {}'.format(self.intensity_type))
        self.exogenous_intensity.print_info()
        self.endogenous_intensity.print_info()

    def forward(self, sample_dict):
        mu, Mu = self.exogenous_intensity(sample_dict)
        alpha, Alpha = self.endogenous_intensity(sample_dict)
        lambda_t = self.act(mu + alpha)
        Lambda_T = self.act(Mu + Alpha)
        return lambda_t, Lambda_T

    def intensity(self, sample_dict):
        mu = self.exogenous_intensity.intensity(sample_dict)
        alpha = self.endogenous_intensity.intensity(sample_dict)
        lambda_t = self.act(mu + alpha)
        return lambda_t

    def expect_counts(self, sample_dict):
        Mu = self.exogenous_intensity.expect_counts(sample_dict)
        Alpha = self.endogenous_intensity.expect_counts(sample_dict)
        Lambda_T = self.act(Mu + Alpha)
        return Lambda_T


class MaxLogLike(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = sum_{i in batch}[ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """

    def __init__(self):
        super(MaxLogLike, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute negative log-likelihood of the given batch
        :param lambda_t: (batchsize, 1) float tensor representing intensity functions
        :param Lambda_t: (batchsize, num_type) float tensor representing the integration of intensity in [t_i-1, t_i]
        :param c: (batchsize, 1) long tensor representing the types of events
        :return: nll (1,)  float tensor representing negative log-likelihood
        """
        return -(lambda_t + self.eps).log().sum() + Lambda_t.sum()


class MaxLogLikePerSample(nn.Module):
    """
    The negative log-likelihood loss of events of point processes
    nll = [ -log lambda_ci(ti) + sum_c Lambda_c(ti) ]
    """

    def __init__(self):
        super(MaxLogLikePerSample, self).__init__()
        self.eps = float(np.finfo(np.float32).eps)

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute negative log-likelihood of the given batch
        :param lambda_t: (batchsize, 1) float tensor representing intensity functions
        :param Lambda_t: (batchsize, num_type) float tensor representing the integration of intensity in [t_i-1, t_i]
        :param c: (batchsize, 1) long tensor representing the types of events
        :return: nll (batchsize,)  float tensor representing negative log-likelihood
        """
        return -(lambda_t[:, 0] + self.eps).log() + Lambda_t.sum(1)


class LeastSquare(nn.Module):
    """
    The least-square loss of events of point processes
    ls = || Lambda_c(t) - N(t) ||_F^2
    """

    def __init__(self):
        super(LeastSquare, self).__init__()
        self.ls_loss = nn.MSELoss()

    def forward(self, lambda_t, Lambda_t, c):
        """
        compute least-square loss between integrated intensity and counting matrix
        :param lambda_t: (batch_size, 1)
        :param Lambda_t: (batch_size, num_type)
        :param c: (batch_size, 1)
        :return:
        """
        mat_onehot = torch.zeros(Lambda_t.size(0), Lambda_t.size(1)).scatter_(1, c, 1)
        return self.ls_loss(Lambda_t, mat_onehot)


class CrossEntropy(nn.Module):
    """
    The cross entropy loss that maximize the conditional probability of current event given its intensity
    ls = -sum_{i in batch} log p(c_i | t_i, c_js, t_js)
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, lambda_t, Lambda_t, c):
        return self.entropy_loss(Lambda_t, c[:, 0])


class GromovWassersteinDiscrepancy(nn.Module):
    """
    Calculate Gromov-Wasserstein discrepancy given optimal transport and cost matrix
    """

    def __init__(self, loss_type):
        super(GromovWassersteinDiscrepancy, self).__init__()
        self.loss_type = loss_type

    def forward(self, As, At, Trans_st, p_s, p_t):
        """
        Calculate GW discrepancy
        :param As: learnable cost matrix of source
        :param At: learnable cost matrix of target
        :param Trans_st: the fixed optimal transport
        :param p_s: the fixed distribution of source
        :param p_t: the fixed distribution of target
        :return: dgw
        """
        ns = p_s.size(0)
        nt = p_t.size(0)
        if self.loss_type == 'L2':
            f1_st = torch.matmul(As ** 2, p_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(p_t), torch.t(At ** 2)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * torch.matmul(torch.matmul(As, Trans_st), torch.t(At))
        else:
            f1_st = torch.matmul(As * torch.log(As + 1e-05) - As, p_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(p_t), torch.t(At)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(As, Trans_st), torch.t(torch.log(At + 1e-05)))
        d_gw = (cost * Trans_st).sum()
        return d_gw


class WassersteinDiscrepancy(nn.Module):
    """
    Calculate Wasserstein discrepancy given optimal transport and
    """

    def __init__(self, loss_type):
        super(WassersteinDiscrepancy, self).__init__()
        self.loss_type = loss_type

    def forward(self, mu_s, mu_t, Trans_st, p_s, p_t):
        """
        Calculate GW discrepancy
        :param mu_s: learnable base intensity of source
        :param mu_t: learnable base intensity of target
        :param Trans_st: the fixed optimal transport
        :param p_s: the fixed distribution of source
        :param p_t: the fixed distribution of target
        :return: dgw
        """
        ns = p_s.size(0)
        nt = p_t.size(0)
        if self.loss_type == 'L2':
            f1_st = (mu_s ** 2).repeat(1, nt)
            f2_st = torch.t(mu_t ** 2).repeat(ns, 1)
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * torch.matmul(mu_s, torch.t(mu_t))
        else:
            f1_st = (mu_s ** 2).repeat(1, nt)
            f2_st = torch.t(mu_t ** 2).repeat(ns, 1)
            cost = f1_st * torch.log(f1_st / (f2_st + 1e-05)) - f1_st + f2_st
        d_w = (cost * Trans_st).sum()
        return d_w


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeastSquare,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (MaxLogLike,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxLogLikePerSample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_HongtengXu_PoPPy(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

