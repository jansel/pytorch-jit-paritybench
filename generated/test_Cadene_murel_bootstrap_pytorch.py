import sys
_module = sys.modules[__name__]
del sys
murel = _module
__version__ = _module
compare_tdiuc_test = _module
compare_tdiuc_val = _module
compare_vqa2_val = _module
datasets = _module
factory = _module
vqacp2 = _module
models = _module
criterions = _module
metrics = _module
networks = _module
factory = _module
murel_cell = _module
murel_net = _module
pairwise = _module
factory = _module
setup = _module
tests = _module
test_run_tdiuc_options = _module
test_run_vqa2_options = _module
test_run_vqacp2_options = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import copy


import torch


import numpy as np


import torch.nn as nn


from copy import deepcopy


import math


import torch.nn.functional as F


import itertools


import scipy


class Pairwise(nn.Module):

    def __init__(self, residual=True, fusion_coord={}, fusion_feat={}, agg={}):
        super(Pairwise, self).__init__()
        self.residual = residual
        self.fusion_coord = fusion_coord
        self.fusion_feat = fusion_feat
        self.agg = agg
        if self.fusion_coord:
            self.f_coord_module = block.factory_fusion(self.fusion_coord)
        if self.fusion_feat:
            self.f_feat_module = block.factory_fusion(self.fusion_feat)
        self.buffer = None

    def set_buffer(self):
        self.buffer = {}

    def forward(self, mm, coords=None):
        bsize = mm.shape[0]
        nregion = mm.shape[1]
        Rij = 0
        if self.fusion_coord:
            assert coords is not None
            coords_l = coords[:, :, (None), :]
            coords_l = coords_l.expand(bsize, nregion, nregion, coords.shape[-1])
            coords_l = coords_l.contiguous()
            coords_l = coords_l.view(bsize * nregion * nregion, coords.shape[-1])
            coords_r = coords[:, (None), :, :]
            coords_r = coords_r.expand(bsize, nregion, nregion, coords.shape[-1])
            coords_r = coords_r.contiguous()
            coords_r = coords_r.view(bsize * nregion * nregion, coords.shape[-1])
            Rij += self.f_coord_module([coords_l, coords_r])
        if self.fusion_feat:
            mm_l = mm[:, :, (None), :]
            mm_l = mm_l.expand(bsize, nregion, nregion, mm.shape[-1])
            mm_l = mm_l.contiguous()
            mm_l = mm_l.view(bsize * nregion * nregion, mm.shape[-1])
            mm_r = mm[:, (None), :, :]
            mm_r = mm_r.expand(bsize, nregion, nregion, mm.shape[-1])
            mm_r = mm_r.contiguous()
            mm_r = mm_r.view(bsize * nregion * nregion, mm.shape[-1])
            Rij += self.f_feat_module([mm_l, mm_r])
        Rij = Rij.view(bsize, nregion, nregion, -1)
        if self.agg['type'] == 'max':
            mm_new, argmax = Rij.max(2)
        else:
            mm_new = getattr(Rij, self.agg['type'])(2)
        if self.buffer is not None:
            self.buffer['mm'] = mm.data.cpu()
            self.buffer['mm_new'] = mm.data.cpu()
            self.buffer['argmax'] = argmax.data.cpu()
            L1_regions = torch.norm(mm_new.data, 1, 2)
            L2_regions = torch.norm(mm_new.data, 2, 2)
            self.buffer['L1_max'] = L1_regions.max(1)[0].cpu()
            self.buffer['L2_max'] = L2_regions.max(1)[0].cpu()
        if self.residual:
            mm_new += mm
        return mm_new


class MuRelCell(nn.Module):

    def __init__(self, residual=False, fusion={}, pairwise={}):
        super(MuRelCell, self).__init__()
        self.residual = residual
        self.fusion = fusion
        self.pairwise = pairwise
        self.fusion_module = block.factory_fusion(self.fusion)
        if self.pairwise:
            self.pairwise_module = Pairwise(**pairwise)

    def forward(self, q_expand, mm, coords=None):
        mm_new = self.process_fusion(q_expand, mm)
        if self.pairwise:
            mm_new = self.pairwise_module(mm_new, coords)
        if self.residual:
            mm_new = mm_new + mm
        return mm_new

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]
        mm = mm.contiguous().view(bsize * n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm


class MuRelNet(nn.Module):

    def __init__(self, txt_enc={}, self_q_att=False, n_step=3, shared=False, cell={}, agg={}, classif={}, wid_to_word={}, word_to_wid={}, aid_to_ans=[], ans_to_aid={}):
        super(MuRelNet, self).__init__()
        self.self_q_att = self_q_att
        self.n_step = n_step
        self.shared = shared
        self.cell = cell
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)
        if self.shared:
            self.cell = MuRelCell(**cell)
        else:
            self.cells = nn.ModuleList([MuRelCell(**cell) for i in range(self.n_step)])
        if 'fusion' in self.classif:
            self.classif_module = block.factory_fusion(self.classif['fusion'])
        elif 'mlp' in self.classif:
            self.classif_module = MLP(self.classif['mlp'])
        else:
            raise ValueError(self.classif.keys())
        Logger().log_value('nparams', sum(p.numel() for p in self.parameters() if p.requires_grad), should_print=True)
        Logger().log_value('nparams_txt_enc', self.get_nparams_txt_enc(), should_print=True)
        self.buffer = None

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def set_buffer(self):
        self.buffer = {}
        if self.shared:
            self.cell.pairwise.set_buffer()
        else:
            for i in range(self.n_step):
                self.cell[i].pairwise.set_buffer()

    def set_pairs_ids(self, n_regions, bsize, device='cuda'):
        if self.shared and self.cell.pairwise:
            self.cell.pairwise_module.set_pairs_ids(n_regions, bsize, device=device)
        else:
            for i in self.n_step:
                if self.cells[i].pairwise:
                    self.cells[i].pairwise_module.set_pairs_ids(n_regions, bsize, device=device)

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        q = self.process_question(q, l)
        bsize = q.shape[0]
        n_regions = v.shape[1]
        q_expand = q[:, (None), :].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize * n_regions, -1)
        mm = v
        for i in range(self.n_step):
            cell = self.cell if self.shared else self.cells[i]
            mm = cell(q_expand, mm, c)
            if self.buffer is not None:
                self.buffer[i] = deepcopy(cell.pairwise.buffer)
        if self.agg['type'] == 'max':
            mm = torch.max(mm, 1)[0]
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)
        if 'fusion' in self.classif:
            logits = self.classif_module([q, mm])
        elif 'mlp' in self.classif:
            logits = self.classif_module(mm)
        out = {'logits': logits}
        return out

    def process_question(self, q, l):
        q_emb = self.txt_enc.embedding(q)
        q, _ = self.txt_enc.rnn(q_emb)
        if self.self_q_att:
            q_att = self.q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = self.q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att * q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            l = list(l.data[:, (0)])
            q = self.txt_enc._select_last(q, l)
        return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out

