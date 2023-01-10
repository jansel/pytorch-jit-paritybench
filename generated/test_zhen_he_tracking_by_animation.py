import sys
_module = sys.modules[__name__]
del sys
functions = _module
submodules = _module
modules = _module
feature_extractor = _module
loss_calculator = _module
net = _module
renderer = _module
submodules = _module
tracker_array = _module
utils = _module
run = _module
gen_duke = _module
gen_duke_bb = _module
gen_duke_bb_bg = _module
gen_duke_processed = _module
gen_duke_roi = _module
gen_mnist = _module
gen_sprite = _module
get_metric_txt = _module
show_curve = _module

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


from torch.autograd import Function


import torch.nn as nn


import scipy.ndimage as ndimage


import matplotlib.pyplot as plt


import time


import math


class FeatureExtractor(nn.Module):

    def __init__(self, o):
        super(FeatureExtractor, self).__init__()
        self.o = o
        params = o.cnn.copy()
        params['conv_features'] = [o.D + 2] + params['conv_features']
        self.cnn = smd.Conv(params['conv_features'], params['conv_kernels'], params['out_sizes'], bn=params['bn'])

    def forward(self, X_seq):
        o = self.o
        X_seq = X_seq.view(-1, X_seq.size(2), X_seq.size(3), X_seq.size(4))
        C3_seq = self.cnn(X_seq)
        C3_seq = C3_seq.permute(0, 2, 3, 1)
        C2_seq = C3_seq.reshape(-1, o.T, o.dim_C2_1, o.dim_C2_2)
        return C2_seq


class LossCalculator(nn.Module):

    def __init__(self, o):
        super(LossCalculator, self).__init__()
        self.o = o
        self.log = smd.Log()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, output, target, area, **kwargs):
        o = self.o
        losses = {}
        if 'Y_b' in kwargs.keys():
            Y_b = kwargs['Y_b']
            loss_recon = self.mse(output, target) + self.mse((output - Y_b).abs(), (target - Y_b).abs())
        else:
            loss_recon = self.mse(output, target)
        loss = loss_recon
        losses['recon'] = loss_recon.item()
        lam_t = 0.1 if o.task == 'duke' else 13
        loss_tight = lam_t * area.sum()
        loss = loss + loss_tight
        losses['tight'] = loss_tight.item()
        y_e = kwargs['y_e']
        lam_e = 1
        loss_entr = lam_e * self.calc_entropy(y_e)
        loss = loss + loss_entr
        losses['entr'] = loss_entr.item()
        if 'Y_a' in kwargs.keys():
            lam_a = 0.1
            app_sum_thresh = 20
            Y_a = kwargs['Y_a']
            Y_a_sum_indiv = Y_a.view(-1, o.T, o.O, o.D * o.h * o.w).sum(3)
            loss_app_indiv = (app_sum_thresh - Y_a_sum_indiv).clamp(min=0)
            loss_app = lam_a * loss_app_indiv.sum()
            loss = loss + loss_app
            losses['loss_app'] = loss_app.item()
        if torch.cuda.current_device() == 0:
            msg = ''
            for k in losses.keys():
                msg = msg + k + ': %.3f, '
                losses[k] /= loss.item()
            None
        return loss

    def calc_entropy(self, x):
        log = self.log
        x_not = 1 - x
        return -(x * log(x) + x_not * log(x_not)).sum()


class Renderer(nn.Module):

    def __init__(self, o):
        super(Renderer, self).__init__()
        self.o = o
        self.i = 0

    def forward(self, y_e, y_l, y_p, Y_s, Y_a, **kwargs):
        o = self.o
        Y_b = kwargs['Y_b'].view(-1, o.D, o.H, o.W) if 'Y_b' in kwargs.keys() else None
        if o.task == 'mnist':
            Y_s = Y_s.data.clone().fill_(1)
        y_e = y_e.view(-1, o.dim_y_e, 1, 1)
        y_p = y_p.view(-1, o.dim_y_p)
        grid, area = self.get_sampling_grid(y_e, y_p)
        area = area.view(-1, o.T, o.O, 1).mean(2)
        Y_s = Y_s.view(-1, 1, o.h, o.w) * y_e
        Y_a = Y_a.view(-1, o.D, o.h, o.w) * Y_s
        X_s = nn.functional.grid_sample(Y_s, grid, align_corners=False)
        X_a = nn.functional.grid_sample(Y_a, grid, align_corners=False)
        X_s = X_s.view(-1, o.O, 1 * o.H * o.W)
        X_a = X_a.view(-1, o.O, o.D * o.H * o.W)
        y_l = y_l.view(-1, o.O, o.dim_y_l)
        y_l = y_l.transpose(1, 2)
        X_s = y_l.bmm(X_s).clamp(max=1)
        X_a = y_l.bmm(X_a)
        if o.task == 'mnist':
            X_a = X_a.clamp(max=1)
        X_s_split = torch.unbind(X_s.view(-1, o.dim_y_l, 1, o.H, o.W), 1)
        X_a_split = torch.unbind(X_a.view(-1, o.dim_y_l, o.D, o.H, o.W), 1)
        X_r = Y_b if Y_b is not None else X_a_split[0].data.clone().zero_()
        for i in range(0, o.dim_y_l):
            X_r = X_r * (1 - X_s_split[i]) + X_a_split[i]
        X_r = X_r.view(-1, o.T, o.D, o.H, o.W)
        return X_r, area

    def get_sampling_grid(self, y_e, y_p):
        """
        y_e: N * dim_y_e * 1 * 1
        y_p: N * dim_y_p (scale_x, scale_y, trans_x, trans_y)
        """
        o = self.o
        scale, ratio, trans_x, trans_y = y_p.split(1, 1)
        scale = 1 + o.zeta_s * scale
        ratio = o.zeta_r[0] + o.zeta_r[1] * ratio
        ratio_sqrt = ratio.sqrt()
        area = scale * scale
        h_new = o.h * scale * ratio_sqrt
        w_new = o.w * scale / ratio_sqrt
        scale_x = o.W / w_new
        scale_y = o.H / h_new
        if o.bg == 0:
            trans_x = (1 - o.w * 2 / 3 / o.W) * trans_x
            trans_y = (1 - o.h * 2 / 3 / o.H) * trans_y
        zero = trans_x.data.clone().zero_()
        trans_mat = torch.cat((scale_x, zero, scale_x * trans_x, zero, scale_y, scale_y * trans_y), 1).view(-1, 2, 3)
        if o.metric == 1 and o.v == 0:
            bb_conf = y_e.data.view(-1, o.dim_y_e)
            bb_h = h_new.data
            bb_w = w_new.data
            bb_center_y = (-trans_y.data + 1) / 2 * (o.H - 1) + 1
            bb_center_x = (-trans_x.data + 1) / 2 * (o.W - 1) + 1
            bb_top = bb_center_y - (bb_h - 1) / 2
            bb_left = bb_center_x - (bb_w - 1) / 2
            bb = torch.cat((bb_left, bb_top, bb_w, bb_h, bb_conf), dim=1)
            torch.save(bb.view(-1, o.T, o.O, 5), path.join(o.result_metric_dir, str(self.i) + '.pt'))
            self.i += 1
        grid = nn.functional.affine_grid(trans_mat, torch.Size((trans_mat.size(0), o.D, o.H, o.W)), align_corners=False)
        return grid, area


class NTMCell(nn.Module):

    def __init__(self, o):
        super(NTMCell, self).__init__()
        self.o = o
        self.linear_k = nn.Linear(o.dim_h_o, o.dim_C2_2)
        self.linear_b = nn.Linear(o.dim_h_o, 1)
        self.linear_e = nn.Linear(o.dim_h_o, o.dim_C2_2)
        self.linear_v = nn.Linear(o.dim_h_o, o.dim_C2_2)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
        self.rnn_cell = nn.GRUCell(o.dim_C2_2, o.dim_h_o)
        self.ha = int(round(np.sqrt(o.dim_C2_1 * o.H / o.W)))
        self.wa = int(round(np.sqrt(o.dim_C2_1 * o.W / o.H)))
        self.att = torch.Tensor(o.O, self.ha, self.wa)
        self.mem = torch.Tensor(o.O, self.ha, self.wa)
        self.i = 0
        self.t = 0
        self.n = 0

    def forward(self, h_o_prev, C):
        """
        h_o_prev: N * dim_h_o
        C: N * C2_1 * C2_2
        """
        o = self.o
        n = self.n
        if o.v > 0:
            if self.i == 0:
                self.att.fill_(0.5)
                self.mem.fill_(0.5)
            self.mem[self.i].copy_(C.data[n].mean(1).view(self.ha, self.wa))
        k = self.linear_k(h_o_prev)
        k_expand = k.unsqueeze(1).expand_as(C)
        beta_pre = self.linear_b(h_o_prev)
        beta_pos = beta_pre.clamp(min=0)
        beta_neg = beta_pre.clamp(max=0)
        beta = beta_neg.exp().log1p() + beta_pos + (-beta_pos).exp().log1p() + (1 - np.log(2))
        C_cos = smd.Identity()(C)
        smd.norm_grad(C_cos, 1)
        s = self.cosine_similarity(C_cos, k_expand).view(-1, o.dim_C2_1)
        w = self.softmax(s * beta)
        w1 = w.unsqueeze(1)
        smd.norm_grad(w1, 1)
        r = w1.bmm(C).squeeze(1)
        h_o = self.rnn_cell(r, h_o_prev)
        if 'no_mem' not in o.exp_config:
            e = self.linear_e(h_o).sigmoid().unsqueeze(1)
            v = self.linear_v(h_o).unsqueeze(1)
            w2 = w.unsqueeze(2)
            C = C * (1 - w2.bmm(e)) + w2.bmm(v)
        if o.v > 0:
            self.att[self.i].copy_(w.data[n].view(self.ha, self.wa))
        return h_o, C, k, r


class NTM(nn.Module):

    def __init__(self, o):
        super(NTM, self).__init__()
        self.o = o
        dim_y = o.dim_y_e + o.dim_y_l + o.dim_y_p + o.dim_Y_s + o.dim_Y_a
        fcn_params = [o.dim_h_o] + o.fcn + [dim_y]
        self.fcn = smd.FCN(fcn_params, hid_trans='relu', out_trans=None)
        self.softmax = nn.Softmax(dim=1)
        self.st_gumbel_sigmoid = smd.STGumbelSigmoid()
        self.st_gumbel_softmax = smd.STGumbelSoftmax(1)
        self.permutation_matrix_calculator = smd.PermutationMatrixCalculator()
        self.ntm_cell = NTMCell(o)
        self.t = 0
        self.att = torch.Tensor(o.T, o.O, self.ntm_cell.ha, self.ntm_cell.wa)
        self.mem = torch.Tensor(o.T, o.O, self.ntm_cell.ha, self.ntm_cell.wa)

    def forward(self, h_o_prev, y_e_prev, C_o):
        """
        h_o_prev: N * O * dim_h_o
        y_e_prev: N * O * dim_y_e
        C_o:      N * C2_1 * C2_2
        """
        o = self.o
        if 'no_tem' in o.exp_config:
            h_o_prev = torch.zeros_like(h_o_prev)
            y_e_prev = torch.zeros_like(y_e_prev)
        if 'no_rep' not in o.exp_config:
            delta = torch.arange(0, o.O).float().unsqueeze(0) * 0.0001
            y_e_prev_mdf = y_e_prev.squeeze(2).round() - delta
            perm_mat = self.permutation_matrix_calculator(y_e_prev_mdf)
            h_o_prev = perm_mat.bmm(h_o_prev)
            y_e_prev = perm_mat.bmm(y_e_prev)
        h_o_prev_split = torch.unbind(h_o_prev, 1)
        h_o_split = {}
        k_split = {}
        r_split = {}
        for i in range(0, o.O):
            self.ntm_cell.i = i
            h_o_split[i], C_o, k_split[i], r_split[i] = self.ntm_cell(h_o_prev_split[i], C_o)
        h_o = torch.stack(tuple(h_o_split.values()), dim=1)
        k = torch.stack(tuple(k_split.values()), dim=1)
        r = torch.stack(tuple(r_split.values()), dim=1)
        att = self.ntm_cell.att
        mem = self.ntm_cell.mem
        if 'no_rep' not in o.exp_config:
            perm_mat_inv = perm_mat.transpose(1, 2)
            h_o = perm_mat_inv.bmm(h_o)
            k = perm_mat_inv.bmm(k)
            r = perm_mat_inv.bmm(r)
            att = perm_mat_inv.data[self.ntm_cell.n].mm(att.view(o.O, -1)).view(o.O, -1, self.ntm_cell.wa)
            mem = perm_mat_inv.data[self.ntm_cell.n].mm(mem.view(o.O, -1)).view(o.O, -1, self.ntm_cell.wa)
        if o.v > 0:
            self.att[self.t].copy_(att)
            self.mem[self.t].copy_(mem)
        a = self.fcn(h_o.view(-1, o.dim_h_o))
        a_e = a[:, 0:o.dim_y_e]
        a_l = a[:, o.dim_y_e:o.dim_y_e + o.dim_y_l]
        a_p = a[:, o.dim_y_e + o.dim_y_l:o.dim_y_e + o.dim_y_l + o.dim_y_p]
        a_s = a[:, o.dim_y_e + o.dim_y_l + o.dim_y_p:o.dim_y_e + o.dim_y_l + o.dim_y_p + o.dim_Y_s]
        a_a = a[:, o.dim_y_e + o.dim_y_l + o.dim_y_p + o.dim_Y_s:o.dim_y_e + o.dim_y_l + o.dim_y_p + o.dim_Y_s + o.dim_Y_a]
        y_e = a_e.tanh().abs()
        y_e = y_e.view(-1, o.O, o.dim_y_e)
        y_l = self.softmax(a_l)
        smd.norm_grad(y_l, 10)
        y_l = self.st_gumbel_softmax(y_l)
        y_l = y_l.view(-1, o.O, o.dim_y_l)
        y_p = a_p.tanh()
        y_p = y_p.view(-1, o.O, o.dim_y_p)
        Y_s = a_s.sigmoid()
        Y_s = self.st_gumbel_sigmoid(Y_s)
        Y_s = Y_s.view(-1, o.O, 1, o.h, o.w)
        Y_a = a_a.sigmoid()
        Y_a = Y_a.view(-1, o.O, o.D, o.h, o.w)
        if 'act' in o.exp_config:
            y_e_perm = perm_mat.bmm(y_e).round()
            y_e_mask = y_e_prev.round() + y_e_perm
            y_e_mask = y_e_mask.lt(0.5).type_as(y_e_mask)
            y_e_mask = y_e_mask.cumsum(1)
            y_e_mask = y_e_mask.lt(0.5).type_as(y_e_mask)
            ones = torch.ones(y_e_mask.size(0), 1, o.dim_y_e)
            y_e_mask = torch.cat((ones, y_e_mask[:, 0:o.O - 1]), dim=1)
            y_e_mask = perm_mat_inv.bmm(y_e_mask)
            h_o = y_e_mask * (h_o - h_o_prev) + h_o_prev
            y_e = y_e_mask * y_e
            y_p = y_e_mask * y_p
            Y_a = y_e_mask.view(-1, o.O, o.dim_y_e, 1, 1) * Y_a
        if self.t == o.T - 1:
            None
        return h_o, y_e, y_l, y_p, Y_s, Y_a


class TrackerArray(nn.Module):

    def __init__(self, o):
        super(TrackerArray, self).__init__()
        self.o = o
        self.ntm = NTM(o)

    def forward(self, h_o_prev, y_e_prev, C_o_seq):
        """
        h_o_prev: N * O * dim_h_o
        y_e_prev: N * O * dim_y_e
        C_o_seq:  N * T * C2_1 * C2_2
        """
        o = self.o
        C_o = torch.unbind(C_o_seq, dim=1)
        h_o, y_e, y_l, y_p, Y_s, Y_a = {}, {}, {}, {}, {}, {}
        for t in range(0, o.T):
            self.ntm.t = t
            self.ntm.ntm_cell.t = t
            h_o[t], y_e[t], y_l[t], y_p[t], Y_s[t], Y_a[t] = self.ntm(h_o_prev, y_e_prev, C_o[t])
            h_o_prev, y_e_prev = h_o[t], y_e[t]
        h_o_seq = torch.stack(tuple(h_o.values()), dim=1)
        y_e_seq = torch.stack(tuple(y_e.values()), dim=1)
        y_l_seq = torch.stack(tuple(y_l.values()), dim=1)
        y_p_seq = torch.stack(tuple(y_p.values()), dim=1)
        Y_s_seq = torch.stack(tuple(Y_s.values()), dim=1)
        Y_a_seq = torch.stack(tuple(Y_a.values()), dim=1)
        return h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq


class Net(nn.Module):

    def __init__(self, o):
        super(Net, self).__init__()
        self.o = o
        device_ids = np.arange(0, o.G).tolist()
        self.feature_extractor = nn.DataParallel(FeatureExtractor(o), device_ids)
        self.tracker_array = TrackerArray(o)
        self.renderer = nn.DataParallel(Renderer(o), device_ids)
        self.renderer_vis = Renderer(o)
        self.loss_calculator = LossCalculator(o)
        zeros = torch.Tensor(o.N, o.T, 1, o.H, o.W).fill_(0)
        dh, dw = 2 / (o.H - 1), 2 / (o.W - 1)
        coor_y = torch.arange(-1, 1 + 1e-05, dh).view(1, 1, 1, o.H, 1)
        coor_x = torch.arange(-1, 1 + 1e-05, dw).view(1, 1, 1, 1, o.W)
        coor_y = coor_y + zeros
        coor_x = coor_x + zeros
        self.coor = torch.cat((coor_y, coor_x), 2)
        self.states = {}
        self.states['h_o_prev'] = torch.Tensor(o.N, o.O, o.dim_h_o)
        self.states['y_e_prev'] = torch.Tensor(o.N, o.O, o.dim_y_e)
        self.reset_states()
        self.n = 0

    def forward(self, X_seq, **kwargs):
        o = self.o
        if 'X_bg_seq' in kwargs.keys():
            Y_b_seq = kwargs['X_bg_seq']
        X_seq_cat = torch.cat((X_seq, self.coor.clone()), 2)
        C_o_seq = self.feature_extractor(X_seq_cat)
        C_o_seq = smd.CheckBP('C_o_seq')(C_o_seq)
        h_o_prev, y_e_prev = self.load_states('h_o_prev', 'y_e_prev')
        h_o_seq, y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq = self.tracker_array(h_o_prev, y_e_prev, C_o_seq)
        if o.r == 1:
            self.save_states(h_o_prev=h_o_seq, y_e_prev=y_e_seq)
        ka = {}
        if o.bg == 1:
            ka['Y_b'] = Y_b_seq
        X_r_seq, area = self.renderer(y_e_seq, y_l_seq, y_p_seq, Y_s_seq, Y_a_seq, **ka)
        ka = {'y_e': y_e_seq}
        if o.bg == 0:
            ka['Y_a'] = Y_a_seq
        else:
            ka['Y_b'] = Y_b_seq
            if o.metric == 0:
                ka['y_p'] = y_p_seq
        loss = self.loss_calculator(X_r_seq, X_seq, area, **ka)
        loss = loss.sum() / (o.N * o.T)
        if o.v > 0:
            ka = {'X': X_seq, 'X_r': X_r_seq, 'y_e': y_e_seq, 'y_l': y_l_seq, 'y_p': y_p_seq, 'Y_s': Y_s_seq, 'Y_a': Y_a_seq}
            if o.bg == 1:
                ka['Y_b'] = Y_b_seq
                if o.metric == 1:
                    ka['X_org'] = kwargs['X_org_seq']
            self.visualize(**ka)
        return loss

    def visualize(self, **kwargs):
        o = self.o
        im_scale = 1
        obj_scale = 1
        n = 0
        self.n = (self.n + 1) % o.N
        H, W = o.H * im_scale, o.W * im_scale
        h, w = o.h * obj_scale, o.w * obj_scale
        if o.v == 2:
            save_dir = path.join(o.pic_dir, str(n))
        show_dict = {'input': kwargs['X'], 'input_recon': kwargs['X_r']}
        if o.bg == 1:
            if o.metric == 1:
                show_dict['org'] = kwargs['X_org']
        att_hor = 1
        if att_hor == 1:
            att = self.tracker_array.ntm.att.permute(0, 2, 1, 3).reshape(o.T, self.tracker_array.ntm.ntm_cell.ha, -1)
            mem = self.tracker_array.ntm.mem.permute(0, 2, 1, 3).reshape(o.T, self.tracker_array.ntm.ntm_cell.ha, -1)
        else:
            att = self.tracker_array.ntm.att.view(o.T, -1, self.tracker_array.ntm.ntm_cell.wa)
            mem = self.tracker_array.ntm.mem.view(o.T, -1, self.tracker_array.ntm.ntm_cell.wa)
        mem_max = 1.8 if o.task == 'mnist' else 3.8
        mem_min = 0
        mem = (mem - mem_min) / (mem_max - mem_min + 1e-20)
        for t in range(0, o.T):
            tao = o.batch_id * o.T + t
            for img_kw, img_arg in show_dict.items():
                img = img_arg.data[n, t].permute(1, 2, 0).clamp(0, 1)
                if o.v == 1:
                    utils.imshow(img, H, W, img_kw)
                elif img_kw == 'input' or img_kw == 'org':
                    utils.mkdir(path.join(save_dir, img_kw))
                    utils.imwrite(img, path.join(save_dir, img_kw, '%05d' % tao))
            if o.metric == 1 and 'no_mem' not in o.exp_config:
                y_e = kwargs['y_e'].data[n:n + 1].clone().round()
            else:
                y_e = kwargs['y_e'].data[n:n + 1].clone()
            y_e_vis = y_e
            y_l = kwargs['y_l'].data[n:n + 1].clone()
            y_p = kwargs['y_p'].data[n:n + 1].clone()
            Y_s = kwargs['Y_s'].data[n:n + 1].clone()
            Y_a = kwargs['Y_a'].data[n:n + 1].clone()
            Y_s.data[:, :, :, :, 0, :].fill_(1)
            Y_s.data[:, :, :, :, -1, :].fill_(1)
            Y_s.data[:, :, :, :, :, 0].fill_(1)
            Y_s.data[:, :, :, :, :, -1].fill_(1)
            Y_a.data[:, :, :, :, 0, :].fill_(1)
            Y_a.data[:, :, :, :, -1, :].fill_(1)
            Y_a.data[:, :, :, :, :, 0].fill_(1)
            Y_a.data[:, :, :, :, :, -1].fill_(1)
            if o.bg == 0:
                X_r_vis, _a = self.renderer_vis(y_e_vis, y_l, y_p, Y_s, Y_a)
            else:
                Y_b = kwargs['Y_b'].data[n:n + 1].clone()
                X_r_vis, _a = self.renderer_vis(y_e_vis, y_l, y_p, Y_s, Y_a, Y_b=Y_b)
            img = X_r_vis.data[0, t, 0:o.D].permute(1, 2, 0).clamp(0, 1)
            if o.v == 1:
                utils.imshow(img, H, W, 'X_r_vis')
            else:
                utils.mkdir(path.join(save_dir, 'X_r_vis'))
                utils.imwrite(img, path.join(save_dir, 'X_r_vis', '%05d' % tao))
            y_e, Y_s, Y_a = y_e.data[0, t], Y_s.data[0, t], Y_a.data[0, t]
            if o.task == 'mnist':
                Y_o = (y_e.view(-1, 1, 1, 1) * Y_a).permute(2, 0, 3, 1).reshape(o.h, o.O * o.w, o.D)
                Y_o_v = (y_e.view(-1, 1, 1, 1) * Y_a).permute(0, 2, 3, 1).reshape(o.O * o.h, o.w, o.D)
            else:
                Y_o = (y_e.view(-1, 1, 1, 1) * Y_s * Y_a).permute(2, 0, 3, 1).reshape(o.h, o.O * o.w, o.D)
                Y_o_v = (y_e.view(-1, 1, 1, 1) * Y_a * Y_a).permute(0, 2, 3, 1).reshape(o.O * o.h, o.w, o.D)
            if o.v == 1:
                utils.imshow(Y_o, h, w * o.O, 'Y_o', 1)
            else:
                utils.mkdir(path.join(save_dir, 'Y_o'))
                utils.imwrite(Y_o, path.join(save_dir, 'Y_o', '%05d' % tao))
            if o.task != 'duke':
                cmap = 'hot'
                att_c = utils.heatmap(att[t], cmap)
                mem_c = utils.heatmap(mem[t], cmap)
                if o.v == 1:
                    sa = 10
                    utils.imshow(att_c, att_c.size(0) * sa, att_c.size(1) * sa, 'att')
                    utils.imshow(mem_c, mem_c.size(0) * sa, mem_c.size(1) * sa, 'mem')
                else:
                    utils.mkdir(path.join(save_dir, 'att'))
                    utils.mkdir(path.join(save_dir, 'mem'))
                    utils.imwrite(att_c, path.join(save_dir, 'att', '%05d' % tao))
                    utils.imwrite(mem_c, path.join(save_dir, 'mem', '%05d' % tao))

    def reset_states(self):
        for state in self.states.values():
            state.fill_(0)

    def load_states(self, *args):
        states = [self.states[arg].clone() for arg in args]
        return states if len(states) > 1 else states[0]

    def save_states(self, **kwargs):
        for kw, arg in kwargs.items():
            self.states[kw].copy_(arg.data[:, -1])


parser = argparse.ArgumentParser()


arg = parser.parse_args()


metric_dir = 'metric'


class CheckBP(nn.Module):

    def __init__(self, label='a', show=1):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        return F.CheckBP.apply(input, self.label, self.show)


class Identity(nn.Module):

    def forward(self, input):
        return F.Identity.apply(input)


class Log(nn.Module):

    def __init__(self, eps=1e-20):
        super(Log, self).__init__()
        self.eps = eps

    def forward(self, input):
        return (input + self.eps).log()


class Round(nn.Module):
    """
    The round operater which is similar to the deterministic Straight-Through Estimator
    It forwards by rounding the input, and backwards with the original output gradients
    """

    def forward(self, input):
        return F.Round.apply(input)


class StraightThrough(nn.Module):
    """
    The stochastic Straight-Through Estimator
    It forwards by sampling from the input probablilities, and backwards with the original output gradients
    """

    def forward(self, input):
        return F.StraightThrough.apply(input)


class ArgMax(nn.Module):
    """
    Input: N * K matrix, where N is the batch size
    Output: N * K matrix, the one-hot encoding of arg_max(input) along the last dimension
    """

    def forward(self, input):
        assert input.dim() == 2, 'only support 2D arg max'
        return F.ArgMax.apply(input)


class STGumbelSigmoid(nn.Module):

    def __init__(self, tao=1.0):
        super(STGumbelSigmoid, self).__init__()
        self.tao = tao
        self.log = Log()
        self.round = Round()

    def forward(self, mu):
        log = self.log
        u1 = torch.rand(mu.size())
        u2 = torch.rand(mu.size())
        a = (log(mu) - log(-log(u1)) - log(1 - mu) + log(-log(u2))) / self.tao
        return self.round(a.sigmoid())


class STGumbelSoftmax(nn.Module):

    def __init__(self, tao=1.0):
        super(STGumbelSoftmax, self).__init__()
        self.tao = tao
        self.log = Log()
        self.softmax = nn.Softmax(dim=1)
        self.arg_max = ArgMax()

    def forward(self, mu):
        log = self.log
        u = torch.rand(mu.size())
        a = (log(mu) - log(-log(u))) / self.tao
        return self.arg_max(self.softmax(a))


class GaussianSampler(nn.Module):

    def forward(self, mu, log_var):
        standard_normal = torch.randn(mu.size())
        return mu + (log_var * 0.5).exp() * standard_normal


class PermutationMatrixCalculator(nn.Module):
    """
    Input: N * K matrix, where N is the batch size
    Output: N * K * K tensor, with each K * K matrix to sort the corresponding row of the input 
    """

    def __init__(self, descend=True):
        super(PermutationMatrixCalculator, self).__init__()
        self.descend = descend

    def forward(self, input):
        assert input.dim() == 2, 'only support 2D input'
        return F.PermutationMatrixCalculator.apply(input, self.descend)


def func(func_name):
    if func_name is None:
        return None
    elif func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    else:
        assert False, 'Invalid func_name.'


class Conv(nn.Module):

    def __init__(self, conv_features, conv_kernels, out_sizes, bn=0, dp=0):
        super(Conv, self).__init__()
        self.layer_num = len(conv_features) - 1
        self.out_sizes = out_sizes
        assert self.layer_num == len(conv_kernels) == len(out_sizes) > 0, 'Invalid conv parameters'
        self.bn = bn
        self.dp = dp
        for i in range(0, self.layer_num):
            setattr(self, 'conv' + str(i), nn.Conv2d(conv_features[i], conv_features[i + 1], (conv_kernels[i][0], conv_kernels[i][1]), stride=1, padding=(conv_kernels[i][0] // 2, conv_kernels[i][1] // 2)))
            if bn == 1:
                setattr(self, 'bn' + str(i), nn.BatchNorm2d(conv_features[i + 1]))
            setattr(self, 'pool' + str(i), nn.AdaptiveMaxPool2d(tuple(out_sizes[i])))
            if dp == 1:
                setattr(self, 'dp' + str(i), nn.Dropout2d(0.2))
        self.tranform = func('relu')

    def forward(self, X):
        H = X
        for i in range(0, self.layer_num):
            H = getattr(self, 'conv' + str(i))(H)
            if self.bn == 1:
                H = getattr(self, 'bn' + str(i))(H)
            H = getattr(self, 'pool' + str(i))(H)
            if self.dp == 1:
                H = getattr(self, 'dp' + str(i))(H)
            H = self.tranform(H)
        return H


class DeConv(nn.Module):

    def __init__(self, scales, conv_features, conv_kernels, conv_paddings, out_trans=None, bn=0, dp=0):
        super(DeConv, self).__init__()
        self.layer_num = len(conv_features) - 1
        self.scales = scales
        assert self.layer_num == len(scales) == len(conv_kernels) == len(conv_paddings) > 0, 'Invalid deconv parameters'
        self.bn = bn
        self.dp = dp
        for i in range(0, self.layer_num):
            if scales[i] > 1:
                setattr(self, 'unpool' + str(i), nn.Upsample(scale_factor=scales[i], mode='nearest'))
            setattr(self, 'conv' + str(i), nn.Conv2d(conv_features[i], conv_features[i + 1], conv_kernels[i], stride=1, padding=tuple(conv_paddings[i])))
            if bn == 1:
                setattr(self, 'bn' + str(i), nn.BatchNorm2d(conv_features[i + 1]))
            if dp == 1:
                setattr(self, 'dp' + str(i), nn.Dropout2d(0.2))
        self.transform = func('relu')
        self.out_trans_func = func(out_trans)

    def forward(self, X):
        H = X
        for i in range(0, self.layer_num):
            if self.scales[i] > 1:
                H = getattr(self, 'unpool' + str(i))(H)
            H = getattr(self, 'conv' + str(i))(H)
            if self.bn == 1:
                H = getattr(self, 'bn' + str(i))(H)
            if self.dp == 1:
                H = getattr(self, 'dp' + str(i))(H)
            if i < self.layer_num - 1:
                H = self.transform(H)
        if self.out_trans_func is not None:
            H = self.out_trans_func(H)
        return H


class FCN(nn.Module):

    def __init__(self, features, hid_trans='tanh', out_trans=None, hid_bn=0, out_bn=0):
        super(FCN, self).__init__()
        self.layer_num = len(features) - 1
        assert self.layer_num > 0, 'Invalid fc parameters'
        self.hid_bn = hid_bn
        self.out_bn = out_bn
        for i in range(0, self.layer_num):
            setattr(self, 'fc' + str(i), nn.Linear(features[i], features[i + 1]))
            if hid_bn == 1:
                setattr(self, 'hid_bn_func' + str(i), nn.BatchNorm1d(features[i + 1]))
        if out_bn == 1:
            self.out_bn_func = nn.BatchNorm1d(features[-1])
        self.hid_trans_func = func(hid_trans)
        self.out_trans_func = func(out_trans)

    def forward(self, X):
        H = X
        for i in range(0, self.layer_num):
            H = getattr(self, 'fc' + str(i))(H)
            if i < self.layer_num - 1:
                if self.hid_bn == 1:
                    H = getattr(self, 'hid_bn_func' + str(i))(H)
                H = self.hid_trans_func(H)
        if self.out_bn == 1:
            H = self.out_bn_func(H)
        if self.out_trans_func is not None:
            H = self.out_trans_func(H)
        return H


class CNN(nn.Module):

    def __init__(self, params):
        super(CNN, self).__init__()
        self.conv = Conv(params['conv_features'], params['conv_kernels'], params['out_sizes'], bn=params['bn'])
        self.fcn = FCN(params['fc_features'], hid_trans='relu', out_trans=params['out_trans'], hid_bn=params['bn'], out_bn=params['bn'])

    def forward(self, X):
        H = self.conv(X)
        H = H.view(H.size(0), -1)
        H = self.fcn(H)
        return H


class DCN(nn.Module):

    def __init__(self, params):
        super(DCN, self).__init__()
        self.fcn = FCN(params['fc_features'], hid_trans='relu', out_trans='relu', hid_bn=params['bn'], out_bn=params['bn'])
        self.deconv = DeConv(params['scales'], params['conv_features'], params['conv_kernels'], params['conv_paddings'], out_trans=params['out_trans'], bn=params['bn'])
        self.H_in, self.W_in = params['H_in'], params['W_in']

    def forward(self, X):
        H = self.fcn(X)
        H = H.view(H.size(0), -1, self.H_in, self.W_in)
        H = self.deconv(H)
        return H


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FCN,
     lambda: ([], {'features': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianSampler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Log,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_zhen_he_tracking_by_animation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

