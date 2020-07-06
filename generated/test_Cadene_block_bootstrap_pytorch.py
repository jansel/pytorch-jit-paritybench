import sys
_module = sys.modules[__name__]
del sys
block = _module
__version__ = _module
compare_tdiuc_test = _module
compare_tdiuc_val = _module
compare_vqa2_val = _module
compare_vrd_test = _module
compare_vrd_val = _module
datasets = _module
factory = _module
create_vrd = _module
tdiuc = _module
vg = _module
vqa2 = _module
vqa2vg = _module
vqa_utils = _module
vrd = _module
vqaEvalDemo = _module
vqaEvaluation = _module
vqaEval = _module
vqaDemo = _module
vqaTools = _module
vqa = _module
models = _module
criterions = _module
kl_divergence = _module
vqa_cross_entropy = _module
vrd_bce = _module
metrics = _module
compute_oe_accuracy = _module
vqa_accuracies = _module
vqa_accuracy = _module
vrd_predicate = _module
vrd_rel_phrase = _module
vrd_utils = _module
networks = _module
factory = _module
fusions = _module
compactbilinearpooling = _module
fusions = _module
mlp = _module
vqa_net = _module
vrd_net = _module
optimizers = _module
factory = _module
lr_scheduler = _module
setup = _module
tests = _module
conftest = _module
test_fusions = _module
test_run_tdiuc_options = _module
test_run_vqa2_options = _module
test_run_vrd_options = _module

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


import torch


import torch.utils.data as data


import numpy as np


import random


import copy


import re


from collections import Counter


import itertools


from torch.utils.data.sampler import WeightedRandomSampler


import torch.nn as nn


from scipy import stats


from collections import defaultdict


import torch.nn.functional as F


import types


from torch.autograd import Function


import numbers


from torch.autograd import Variable


import inspect


class VQACrossEntropyLoss(nn.Module):

    def __init__(self):
        super(VQACrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        out = {}
        out['loss'] = self.loss(net_out['logits'], batch['class_id'].squeeze(1))
        return out


class VRDBCELoss(nn.Module):

    def __init__(self):
        super(VRDBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, net_output, target):
        y_true = target['target_oh']
        cost = self.loss(net_output['rel_scores'], y_true)
        out = {}
        out['loss'] = cost
        return out


class VQAAccuracy(nn.Module):

    def __init__(self, topk=[1, 5]):
        super(VQAAccuracy, self).__init__()
        self.topk = topk

    def __call__(self, cri_out, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        class_id = batch['class_id'].data.cpu()
        acc_out = accuracy(logits, class_id, topk=self.topk)
        for i, k in enumerate(self.topk):
            out['accuracy_top{}'.format(k)] = acc_out[i]
        return out


class VQAAccuracies(nn.Module):

    def __init__(self, engine=None, mode='eval', open_ended=True, tdiuc=True, dir_exp='', dir_vqa=''):
        super(VQAAccuracies, self).__init__()
        self.engine = engine
        self.mode = mode
        self.open_ended = open_ended
        self.tdiuc = tdiuc
        self.dir_exp = dir_exp
        self.dir_vqa = dir_vqa
        self.dataset = engine.dataset[mode]
        self.ans_to_aid = self.dataset.ans_to_aid
        self.results = None
        self.results_testdev = None
        self.dir_rslt = None
        self.path_rslt = None
        if self.tdiuc or self.dataset.split != 'test':
            self.accuracy = VQAAccuracy()
        else:
            self.accuracy = None
        if self.open_ended:
            engine.register_hook('{}_on_start_epoch'.format(mode), self.reset_oe)
            engine.register_hook('{}_on_end_epoch'.format(mode), self.compute_oe_accuracy)
        if self.tdiuc:
            engine.register_hook('{}_on_start_epoch'.format(mode), self.reset_tdiuc)
            engine.register_hook('{}_on_end_epoch'.format(mode), self.compute_tdiuc_metrics)

    def reset_oe(self):
        self.results = []
        self.dir_rslt = os.path.join(self.dir_exp, 'results', self.dataset.split, 'epoch,{}'.format(self.engine.epoch))
        os.system('mkdir -p ' + self.dir_rslt)
        self.path_rslt = os.path.join(self.dir_rslt, 'OpenEnded_mscoco_{}_model_results.json'.format(self.dataset.get_subtype()))
        if self.dataset.split == 'test':
            self.results_testdev = []
            self.path_rslt_testdev = os.path.join(self.dir_rslt, 'OpenEnded_mscoco_{}_model_results.json'.format(self.dataset.get_subtype(testdev=True)))
            self.path_logits = os.path.join(self.dir_rslt, 'logits.pth')
            os.system('mkdir -p ' + os.path.dirname(self.path_logits))
            self.logits = {}
            self.logits['aid_to_ans'] = self.engine.model.network.aid_to_ans
            self.logits['qid_to_idx'] = {}
            self.logits['tensor'] = None
            self.idx = 0
            path_aid_to_ans = os.path.join(self.dir_rslt, 'aid_to_ans.json')
            with open(path_aid_to_ans, 'w') as f:
                json.dump(self.engine.model.network.aid_to_ans, f)

    def save_logits(self):
        torch.save(self.logits, self.path_logits)

    def reset_tdiuc(self):
        self.pred_aids = []
        self.gt_aids = []
        self.gt_types = []
        self.gt_aid_not_found = 0
        self.res_by_type = defaultdict(list)

    def forward(self, cri_out, net_out, batch):
        out = {}
        if self.accuracy is not None:
            out = self.accuracy(cri_out, net_out, batch)
        if self.open_ended and self.dataset.split == 'test':
            logits = torch.nn.functional.softmax(net_out['logits'], dim=1).data.cpu()
        net_out = self.engine.model.network.process_answers(net_out)
        batch_size = len(batch['index'])
        for i in range(batch_size):
            if self.open_ended:
                pred_item = {'question_id': batch['question_id'][i], 'answer': net_out['answers'][i]}
                self.results.append(pred_item)
                if self.dataset.split == 'test':
                    if 'is_testdev' in batch and batch['is_testdev'][i]:
                        self.results_testdev.append(pred_item)
                    if self.logits['tensor'] is None:
                        self.logits['tensor'] = torch.FloatTensor(len(self.dataset), logits.size(1))
                    self.logits['tensor'][self.idx] = logits[i]
                    self.logits['qid_to_idx'][batch['question_id'][i]] = self.idx
                    self.idx += 1
            if self.tdiuc:
                qid = batch['question_id'][i]
                pred_aid = net_out['answer_ids'][i]
                self.pred_aids.append(pred_aid)
                gt_aid = batch['answer_id'][i]
                gt_ans = batch['answer'][i]
                gt_type = batch['question_type'][i]
                self.gt_types.append(gt_type)
                self.res_by_type[gt_type + '_pred'].append(pred_aid)
                if gt_ans in self.ans_to_aid:
                    self.gt_aids.append(gt_aid)
                    self.res_by_type[gt_type + '_gt'].append(gt_aid)
                    if gt_aid == pred_aid:
                        self.res_by_type[gt_type + '_t'].append(pred_aid)
                    else:
                        self.res_by_type[gt_type + '_f'].append(pred_aid)
                else:
                    self.gt_aids.append(-1)
                    self.res_by_type[gt_type + '_gt'].append(-1)
                    self.res_by_type[gt_type + '_f'].append(pred_aid)
                    self.gt_aid_not_found += 1
        return out

    def compute_oe_accuracy(self):
        with open(self.path_rslt, 'w') as f:
            json.dump(self.results, f)
        if self.dataset.split == 'test':
            with open(self.path_rslt_testdev, 'w') as f:
                json.dump(self.results_testdev, f)
        if 'test' not in self.dataset.split:
            call_to_prog = 'python -m block.models.metrics.compute_oe_accuracy ' + '--dir_vqa {} --dir_exp {} --dir_rslt {} --epoch {} --split {} &'.format(self.dir_vqa, self.dir_exp, self.dir_rslt, self.engine.epoch, self.dataset.split)
            Logger()('`' + call_to_prog + '`')
            os.system(call_to_prog)

    def compute_tdiuc_metrics(self):
        Logger()('{} of validation answers were not found in ans_to_aid'.format(self.gt_aid_not_found))
        accuracy = float(100 * np.mean(np.array(self.pred_aids) == np.array(self.gt_aids)))
        Logger()('Overall Traditional Accuracy is {:.2f}'.format(accuracy))
        Logger().log_value('{}_epoch.tdiuc.accuracy'.format(self.mode), accuracy, should_print=False)
        types = list(set(self.gt_types))
        sum_acc = []
        eps = 1e-10
        Logger()('---------------------------------------')
        Logger()('Not using per-answer normalization...')
        for tp in types:
            acc = 100 * (len(self.res_by_type[tp + '_t']) / len(self.res_by_type[tp + '_t'] + self.res_by_type[tp + '_f']))
            sum_acc.append(acc + eps)
            Logger()("Accuracy for class '{}' is {:.2f}".format(tp, acc))
            Logger().log_value('{}_epoch.tdiuc.perQuestionType.{}'.format(self.mode, tp), acc, should_print=False)
        acc_mpt_a = float(np.mean(np.array(sum_acc)))
        Logger()('Arithmetic MPT Accuracy is {:.2f}'.format(acc_mpt_a))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_a'.format(self.mode), acc_mpt_a, should_print=False)
        acc_mpt_h = float(stats.hmean(sum_acc))
        Logger()('Harmonic MPT Accuracy is {:.2f}'.format(acc_mpt_h))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_h'.format(self.mode), acc_mpt_h, should_print=False)
        Logger()('---------------------------------------')
        Logger()('Using per-answer normalization...')
        for tp in types:
            per_ans_stat = defaultdict(int)
            for g, p in zip(self.res_by_type[tp + '_gt'], self.res_by_type[tp + '_pred']):
                per_ans_stat[str(g) + '_gt'] += 1
                if g == p:
                    per_ans_stat[str(g)] += 1
            unq_acc = 0
            for unq_ans in set(self.res_by_type[tp + '_gt']):
                acc_curr_ans = per_ans_stat[str(unq_ans)] / per_ans_stat[str(unq_ans) + '_gt']
                unq_acc += acc_curr_ans
            acc = 100 * unq_acc / len(set(self.res_by_type[tp + '_gt']))
            sum_acc.append(acc + eps)
            Logger()("Accuracy for class '{}' is {:.2f}".format(tp, acc))
            Logger().log_value('{}_epoch.tdiuc.perQuestionType_norm.{}'.format(self.mode, tp), acc, should_print=False)
        acc_mpt_a = float(np.mean(np.array(sum_acc)))
        Logger()('Arithmetic MPT Accuracy is {:.2f}'.format(acc_mpt_a))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_a_norm'.format(self.mode), acc_mpt_a, should_print=False)
        acc_mpt_h = float(stats.hmean(sum_acc))
        Logger()('Harmonic MPT Accuracy is {:.2f}'.format(acc_mpt_h))
        Logger().log_value('{}_epoch.tdiuc.acc_mpt_h_norm'.format(self.mode), acc_mpt_h, should_print=False)


class VRDPredicate(nn.Module):

    def __init__(self, engine=None, split='test', nb_classes=71):
        super(VRDPredicate, self).__init__()
        self.split = split
        self.k = nb_classes
        self.reset()
        if engine:
            engine.register_hook('%s_on_end_epoch' % split, self.calculate_metrics)

    def reset(self):
        self.tps = {(50): [], (100): []}
        self.fps = {(50): [], (100): []}
        self.scores = {(50): [], (100): []}
        self.total_num_gts = {(50): 0, (100): 0}

    def forward(self, cri_out, net_out, batch):
        out = {}
        nb_classes = net_out['rel_scores'].size(1)
        out['rel_scores'] = net_out['rel_scores']
        det_labels, det_boxes = [], []
        gt_labels, gt_boxes = [], []
        rel_scores = torch.sigmoid(net_out['rel_scores'])
        acc_out = accuracy(rel_scores[batch['target_cls_id'] > 0], batch['target_cls_id'][batch['target_cls_id'] > 0], topk=[1, 5])
        out['accuracy_top1'] = acc_out[0].item()
        out['accuracy_top5'] = acc_out[1].item()
        for batch_id in range(len(batch['rels'])):
            _index = (batch['batch_id'] == batch_id) * (batch['target_cls_id'] > 0)
            n_index = int(_index.int().sum().cpu().data)
            oid_to_box = {obj['object_id']: [obj['x'], obj['y'], obj['w'], obj['h']] for obj in batch['objects'][batch_id]}
            rel_pred = rel_scores[_index]
            subj_pred_boxes = batch['subject_boxes_raw'][_index]
            obj_pred_boxes = batch['object_boxes_raw'][_index]
            subj_gt_boxes = np.array([oid_to_box[rel['subject_id']] for rel in batch['rels'][batch_id]])
            obj_gt_boxes = np.array([oid_to_box[rel['object_id']] for rel in batch['rels'][batch_id]])
            subj_gt_boxes[:, (2)] += subj_gt_boxes[:, (0)]
            obj_gt_boxes[:, (2)] += obj_gt_boxes[:, (0)]
            subj_gt_boxes[:, (3)] += subj_gt_boxes[:, (1)]
            obj_gt_boxes[:, (3)] += obj_gt_boxes[:, (1)]
            _gt_boxes = np.concatenate([subj_gt_boxes[:, (None), :], obj_gt_boxes[:, (None), :]], 1)
            _gt_labels = torch.cat([batch['subject_cls_id'][_index][:, (None)], batch['target_cls_id'][_index][:, (None)], batch['object_cls_id'][_index][:, (None)]], 1).cpu().data.numpy()
            top_score, top_pred = rel_pred.topk(self.k)
            top_score = top_score.cpu().data.numpy()
            top_pred = top_pred.cpu().data.numpy()
            _det_labels, _det_boxes = [], []
            for i in range(n_index):
                s = _gt_labels[i, 0]
                o = _gt_labels[i, 2]
                box_s = _gt_boxes[i, 0]
                box_o = _gt_boxes[i, 1]
                _det_labels.append(np.concatenate([np.ones((self.k, 1)), top_score[i][:, (None)], np.ones((self.k, 1)), s * np.ones((self.k, 1)), top_pred[i][:, (None)], [o] * np.ones((self.k, 1))], 1))
                _det_boxes.append(np.tile(_gt_boxes[i][(None), :, :], (self.k, 1, 1)))
            det_labels.append(np.vstack(_det_labels))
            det_boxes.append(np.vstack(_det_boxes))
            gt_labels.append(_gt_labels)
            gt_boxes.append(_gt_boxes)
        for R in [50, 100]:
            _tp, _fp, _score, _num_gts = vrd_utils.eval_batch([det_labels, det_boxes], [gt_labels, gt_boxes], num_dets=R)
            self.total_num_gts[R] += _num_gts
            self.tps[R] += _tp
            self.fps[R] += _fp
            self.scores[R] += _score
        return out

    def calculate_metrics(self):
        for R in [50, 100]:
            top_recall = vrd_utils.calculate_recall(R, self.tps, self.fps, self.scores, self.total_num_gts)
            Logger().log_value(f'{self.split}_epoch.predicate.R_{R}', top_recall, should_print=True)
        self.reset()


class VRDRelationshipPhrase(nn.Module):

    def __init__(self, engine=None, split='test'):
        super(VRDRelationshipPhrase, self).__init__()
        self.split = split
        self.activation = torch.sigmoid
        self.reset()
        self.dataset = engine.dataset['eval']
        if engine:
            engine.register_hook(f'{split}_on_end_epoch', self.calculate_metrics)

    def reset(self):
        self.tps = {(50): [], (100): []}
        self.fps = {(50): [], (100): []}
        self.scores = {(50): [], (100): []}
        self.total_num_gts = {(50): 0, (100): 0}
        self.tps_union = {(50): [], (100): []}
        self.fps_union = {(50): [], (100): []}
        self.scores_union = {(50): [], (100): []}
        self.total_num_gts_union = {(50): 0, (100): 0}

    def forward(self, cri_out, net_out, batch):
        det_labels, det_boxes, gt_labels, gt_boxes = [], [], [], []
        npairs_count = 0
        nboxes_count = 0
        batch_size = len(batch['idx'])
        total_npairs = net_out['rel_scores'].shape[0]
        nclasses = net_out['rel_scores'].shape[1]
        rel_scores = self.activation(net_out['rel_scores'])
        for idx in range(batch_size):
            image_id = batch['image_id'][idx]
            annot = self.dataset.json_raw[image_id]
            _gt_labels, _gt_boxes = vrd_utils.annot_to_gt(annot)
            nboxes = batch['n_boxes'][idx]
            npairs = batch['n_pairs'][idx]
            if nboxes == 0:
                _det_labels = np.zeros((0, 6))
                _det_boxes = np.zeros((0, 2, 4))
            else:
                begin_ = npairs_count
                end_ = begin_ + npairs
                rel_score = rel_scores[begin_:end_].view(nboxes, nboxes, nclasses)
                rel_score = rel_score.data.cpu()
                rel_prob = rel_score.topk(10)
                begin_ = nboxes_count
                end_ = begin_ + nboxes
                item = {'rel_score': rel_score, 'rel_prob': rel_prob, 'rois': batch['rois_nonorm'][begin_:end_].cpu(), 'cls': batch['cls'][begin_:end_].cpu(), 'cls_scores': batch['cls_scores'][begin_:end_].cpu()}
                _det_labels, _det_boxes = vrd_utils.item_to_det(item)
            npairs_count += npairs
            nboxes_count += nboxes
            det_labels.append(_det_labels)
            det_boxes.append(_det_boxes)
            gt_labels.append(_gt_labels)
            gt_boxes.append(_gt_boxes)
        for R in [50, 100]:
            _tp, _fp, _score, _num_gts = vrd_utils.eval_batch([det_labels, det_boxes], [gt_labels, gt_boxes], num_dets=R)
            self.total_num_gts[R] += _num_gts
            self.tps[R] += _tp
            self.fps[R] += _fp
            self.scores[R] += _score
            _tp, _fp, _score, _num_gts = vrd_utils.eval_batch_union([det_labels, det_boxes], [gt_labels, gt_boxes], num_dets=R)
            self.total_num_gts_union[R] += _num_gts
            self.tps_union[R] += _tp
            self.fps_union[R] += _fp
            self.scores_union[R] += _score
        return {}

    def calculate_metrics(self):
        for R in [50, 100]:
            top_recall = vrd_utils.calculate_recall(R, self.tps, self.fps, self.scores, self.total_num_gts)
            Logger().log_value(f'{self.split}_epoch.rel.R_{R}', top_recall, should_print=True)
            top_recall = vrd_utils.calculate_recall(R, self.tps_union, self.fps_union, self.scores_union, self.total_num_gts_union)
            Logger().log_value(f'{self.split}_epoch.phrase.R_{R}', top_recall, should_print=True)
        self.reset()


def CountSketchFn_backward(h, s, x_size, grad_output):
    s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)
    s = s.view(s_view)
    h = h.view(s_view).expand(x_size)
    grad_x = grad_output.gather(-1, h)
    grad_x = grad_x * s
    return grad_x


def CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add=False):
    x_size = tuple(x.size())
    s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)
    out_size = x_size[:-1] + (output_size,)
    s = s.view(s_view)
    xs = x * s
    h = h.view(s_view).expand(x_size)
    if force_cpu_scatter_add:
        out = x.new(*out_size).zero_().cpu()
        return out.scatter_add_(-1, h.cpu(), xs.cpu())
    else:
        out = x.new(*out_size).zero_()
        return out.scatter_add_(-1, h, xs)


class CountSketchFn(Function):

    @staticmethod
    def forward(ctx, h, s, output_size, x, force_cpu_scatter_add=False):
        x_size = tuple(x.size())
        ctx.save_for_backward(h, s)
        ctx.x_size = tuple(x.size())
        return CountSketchFn_forward(h, s, output_size, x, force_cpu_scatter_add)

    @staticmethod
    def backward(ctx, grad_output):
        h, s = ctx.saved_variables
        grad_x = CountSketchFn_backward(h, s, ctx.x_size, grad_output)
        return None, None, None, grad_x


class CountSketch(nn.Module):
    """Compute the count sketch over an input signal.

    .. math::

        out_j = \\sum_{i : j = h_i} s_i x_i

    Args:
        input_size (int): Number of channels in the input array
        output_size (int): Number of channels in the output sketch
        h (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s (array, optional): Optional array of size input_size of -1 and 1.

    .. note::

        If h and s are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input: (...,input_size)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input_size, output_size, h=None, s=None):
        super(CountSketch, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        if h is None:
            h = torch.LongTensor(input_size).random_(0, output_size)
        if s is None:
            s = 2 * torch.Tensor(input_size).random_(0, 2) - 1

        def identity(self):
            return self
        h.float = types.MethodType(identity, h)
        h.double = types.MethodType(identity, h)
        self.register_buffer('h', h)
        self.register_buffer('s', s)

    def forward(self, x):
        x_size = list(x.size())
        assert x_size[-1] == self.input_size
        return CountSketchFn.apply(self.h, self.s, self.output_size, x)


def ComplexMultiply_forward(X_re, X_im, Y_re, Y_im):
    Z_re = torch.addcmul(X_re * Y_re, -1, X_im, Y_im)
    Z_im = torch.addcmul(X_re * Y_im, 1, X_im, Y_re)
    return Z_re, Z_im


class CompactBilinearPoolingFn(Function):

    @staticmethod
    def forward(ctx, h1, s1, h2, s2, output_size, x, y, force_cpu_scatter_add=False):
        ctx.save_for_backward(h1, s1, h2, s2, x, y)
        ctx.x_size = tuple(x.size())
        ctx.y_size = tuple(y.size())
        ctx.force_cpu_scatter_add = force_cpu_scatter_add
        ctx.output_size = output_size
        px = CountSketchFn_forward(h1, s1, output_size, x, force_cpu_scatter_add)
        fx = torch.rfft(px, 1)
        re_fx = fx.select(-1, 0)
        im_fx = fx.select(-1, 1)
        del px
        py = CountSketchFn_forward(h2, s2, output_size, y, force_cpu_scatter_add)
        fy = torch.rfft(py, 1)
        re_fy = fy.select(-1, 0)
        im_fy = fy.select(-1, 1)
        del py
        re_prod, im_prod = ComplexMultiply_forward(re_fx, im_fx, re_fy, im_fy)
        re = torch.irfft(torch.stack((re_prod, im_prod), re_prod.dim()), 1, signal_sizes=(output_size,))
        return re

    @staticmethod
    def backward(ctx, grad_output):
        h1, s1, h2, s2, x, y = ctx.saved_tensors
        px = CountSketchFn_forward(h1, s1, ctx.output_size, x, ctx.force_cpu_scatter_add)
        py = CountSketchFn_forward(h2, s2, ctx.output_size, y, ctx.force_cpu_scatter_add)
        grad_output = grad_output.contiguous()
        grad_prod = torch.rfft(grad_output, 1)
        grad_re_prod = grad_prod.select(-1, 0)
        grad_im_prod = grad_prod.select(-1, 1)
        fy = torch.rfft(py, 1)
        re_fy = fy.select(-1, 0)
        im_fy = fy.select(-1, 1)
        del py
        grad_re_fx = torch.addcmul(grad_re_prod * re_fy, 1, grad_im_prod, im_fy)
        grad_im_fx = torch.addcmul(grad_im_prod * re_fy, -1, grad_re_prod, im_fy)
        grad_fx = torch.irfft(torch.stack((grad_re_fx, grad_im_fx), grad_re_fx.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_x = CountSketchFn_backward(h1, s1, ctx.x_size, grad_fx)
        del re_fy, im_fy, grad_re_fx, grad_im_fx, grad_fx
        fx = torch.rfft(px, 1)
        re_fx = fx.select(-1, 0)
        im_fx = fx.select(-1, 1)
        del px
        grad_re_fy = torch.addcmul(grad_re_prod * re_fx, 1, grad_im_prod, im_fx)
        grad_im_fy = torch.addcmul(grad_im_prod * re_fx, -1, grad_re_prod, im_fx)
        grad_fy = torch.irfft(torch.stack((grad_re_fy, grad_im_fy), grad_re_fy.dim()), 1, signal_sizes=(ctx.output_size,))
        grad_y = CountSketchFn_backward(h2, s2, ctx.y_size, grad_fy)
        del re_fx, im_fx, grad_re_fy, grad_im_fy, grad_fy
        return None, None, None, None, None, grad_x, grad_y, None


class CompactBilinearPooling(nn.Module):
    """Compute the compact bilinear pooling between two input array x and y

    .. math::

        out = \\Psi (x,h_1,s_1) \\ast \\Psi (y,h_2,s_2)

    Args:
        input_size1 (int): Number of channels in the first input array
        input_size2 (int): Number of channels in the second input array
        output_size (int): Number of channels in the output array
        h1 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s1 (array, optional): Optional array of size input_size of -1 and 1.
        h2 (array, optional): Optional array of size input_size of indices in the range [0,output_size]
        s2 (array, optional): Optional array of size input_size of -1 and 1.
        force_cpu_scatter_add (boolean, optional): Force the scatter_add operation to run on CPU for testing purposes

    .. note::

        If h1, s1, s2, h2 are None, they will be automatically be generated using LongTensor.random_.

    Shape:
        - Input 1: (...,input_size1)
        - Input 2: (...,input_size2)
        - Output: (...,output_size)

    References:
        Yang Gao et al. "Compact Bilinear Pooling" in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
        Akira Fukui et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding", arXiv:1606.01847 (2016).
    """

    def __init__(self, input1_size, input2_size, output_size, h1=None, s1=None, h2=None, s2=None, force_cpu_scatter_add=False):
        super(CompactBilinearPooling, self).__init__()
        self.add_module('sketch1', CountSketch(input1_size, output_size, h1, s1))
        self.add_module('sketch2', CountSketch(input2_size, output_size, h2, s2))
        self.output_size = output_size
        self.force_cpu_scatter_add = force_cpu_scatter_add

    def forward(self, x, y=None):
        if y is None:
            y = x
        return CompactBilinearPoolingFn.apply(self.sketch1.h, self.sketch1.s, self.sketch2.h, self.sketch2.s, self.output_size, x, y, self.force_cpu_scatter_add)


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


class Block(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20, rank=15, shared=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0, pos_norm='before_cat'):
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size * rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)), self.merge_linears0, self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c)
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class BlockTucker(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20, shared=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0, pos_norm='before_cat'):
        super(BlockTucker, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if self.shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        bilinears = []
        for size in self.sizes_list:
            bilinears.append(nn.Bilinear(size, size, size))
        self.bilinears = nn.ModuleList(bilinears)
        self.linear_out = nn.Linear(self.mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, bilinear in enumerate(self.bilinears):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            z = bilinear(x0_c, x1_c)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class Mutan(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, rank=15, shared=False, normalize=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0):
        super(Mutan, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = rank
        self.output_dim = output_dim
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.normalize = normalize
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim * rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim * rank)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        m0 = self.merge_linear0(x0)
        m1 = self.merge_linear1(x1)
        m = m0 * m1
        m = m.view(-1, self.rank, self.mm_dim)
        z = torch.sum(m, 1)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class Tucker(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, shared=False, normalize=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0):
        super(Tucker, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.bilinear = nn.Bilinear(mm_dim, mm_dim, mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = self.bilinear(x0, x1)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MLB(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, activ_input='relu', activ_output='relu', normalize=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0):
        super(MLB, self).__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 * x1
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MFB(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, factor=2, activ_input='relu', activ_output='relu', normalize=False, dropout_input=0.0, dropout_pre_norm=0.0, dropout_output=0.0):
        super(MFB, self).__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 * x1
        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)
        z = z.view(z.size(0), self.mm_dim, self.factor)
        z = z.sum(2)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MFH(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, factor=2, activ_input='relu', activ_output='relu', normalize=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0):
        super(MFH, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim * 2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0_0(x[0])
        x1 = self.linear1_0(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z_0_skip = x0 * x1
        if self.dropout_pre_lin:
            z_0_skip = F.dropout(z_0_skip, p=self.dropout_pre_lin, training=self.training)
        z_0 = z_0_skip.view(z_0_skip.size(0), self.mm_dim, self.factor)
        z_0 = z_0.sum(2)
        if self.normalize:
            z_0 = torch.sqrt(F.relu(z_0)) - torch.sqrt(F.relu(-z_0))
            z_0 = F.normalize(z_0, p=2)
        x0 = self.linear0_1(x[0])
        x1 = self.linear1_1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z_1 = x0 * x1 * z_0_skip
        if self.dropout_pre_lin > 0:
            z_1 = F.dropout(z_1, p=self.dropout_pre_lin, training=self.training)
        z_1 = z_1.view(z_1.size(0), self.mm_dim, self.factor)
        z_1 = z_1.sum(2)
        if self.normalize:
            z_1 = torch.sqrt(F.relu(z_1)) - torch.sqrt(F.relu(-z_1))
            z_1 = F.normalize(z_1, p=2)
        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class MCB(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=16000, activ_output='relu', dropout_output=0.0):
        super(MCB, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_output = activ_output
        self.dropout_output = dropout_output
        self.mcb = cbp.CompactBilinearPooling(input_dims[0], input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        z = self.mcb(x[0], x[1])
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class LinearSum(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, activ_input='relu', activ_output='relu', normalize=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0):
        super(LinearSum, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 + x1
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


class ConcatMLP(nn.Module):

    def __init__(self, input_dims, output_dim, dimensions=[500, 500], activation='relu', dropout=0.0):
        super(ConcatMLP, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        self.mlp = mlp.MLP(self.input_dim, self.dimensions, self.activation, self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if x[0].dim() == 3 and x[1].dim() == 2:
            x[1] = x[1].unsqueeze(1).reshape_as(x[0])
        if x[1].dim() == 3 and x[0].dim() == 2:
            x[0] = x[0].unsqueeze(1).reshape_as(x[1])
        z = torch.cat(x, dim=x[0].dim() - 1)
        z = self.mlp(z)
        return z


class MLP(nn.Module):

    def __init__(self, input_dim, dimensions, activation='relu', dropout=0.0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


class Attention(nn.Module):

    def __init__(self, mlp_glimpses=0, fusion={}):
        super(Attention, self).__init__()
        self.mlp_glimpses = mlp_glimpses
        self.fusion = factory_fusion(fusion)
        if self.mlp_glimpses > 0:
            self.linear0 = nn.Linear(fusion['output_dim'], 512)
            self.linear1 = nn.Linear(512, mlp_glimpses)

    def forward(self, q, v):
        alpha = self.process_attention(q, v)
        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)
        alpha = F.softmax(alpha, dim=1)
        if alpha.size(2) > 1:
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                alpha = alpha.unsqueeze(2).expand_as(v)
                v_out = alpha * v
                v_out = v_out.sum(1)
                v_outs.append(v_out)
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha * v
            v_out = v_out.sum(1)
        return v_out

    def process_attention(self, q, v):
        batch_size = q.size(0)
        n_regions = v.size(1)
        q = q[:, (None), :].expand(q.size(0), n_regions, q.size(1))
        alpha = self.fusion([q.contiguous().view(batch_size * n_regions, -1), v.contiguous().view(batch_size * n_regions, -1)])
        alpha = alpha.view(batch_size, n_regions, -1)
        return alpha


def factory_text_enc(vocab_words, opt):
    list_words = [vocab_words[i + 1] for i in range(len(vocab_words))]
    if opt['name'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'], list_words, dropout=opt['dropout'], fixed_emb=opt['fixed_emb'])
    else:
        raise NotImplementedError
    return seq2vec


def mask_softmax(x, lengths):
    mask = torch.zeros_like(x)
    t_lengths = lengths[:, :, (None)].expand_as(mask)
    arange_id = torch.arange(mask.size(1))
    arange_id = arange_id[(None), :, (None)].expand_as(mask)
    mask[arange_id < t_lengths] = 1
    x = torch.exp(x)
    x = x * mask
    x = x / torch.sum(x, dim=1, keepdim=True).expand_as(x)
    return x


class VQANet(nn.Module):

    def __init__(self, txt_enc={}, self_q_att=False, attention={}, classif={}, wid_to_word={}, word_to_wid={}, aid_to_ans=[], ans_to_aid={}):
        super(VQANet, self).__init__()
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)
        self.attention = Attention(**attention)
        self.fusion = factory_fusion(classif['fusion'])

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths']
        q = self.process_question(q, l)
        v = self.attention(q, v)
        logits = self.fusion([q, v])
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


class VRDNet(nn.Module):

    def __init__(self, opt):
        super(VRDNet, self).__init__()
        self.opt = opt
        self.classeme_embedding = nn.Embedding(self.opt['nb_classeme'], self.opt['classeme_dim'])
        self.fusion_c = factory_fusion(self.opt['classeme'])
        self.fusion_s = factory_fusion(self.opt['spatial'])
        self.fusion_f = factory_fusion(self.opt['feature'])
        self.predictor = MLP(**self.opt['predictor'])

    def forward(self, batch):
        bsize = batch['subject_boxes'].size(0)
        x_c = [self.classeme_embedding(batch['subject_cls_id']), self.classeme_embedding(batch['object_cls_id'])]
        x_s = [batch['subject_boxes'], batch['object_boxes']]
        x_f = [batch['subject_features'], batch['object_features']]
        x_c = self.fusion_c(x_c)
        x_s = self.fusion_s(x_s)
        x_f = self.fusion_f(x_f)
        x = torch.cat([x_c, x_s, x_f], -1)
        if 'aggreg_dropout' in self.opt:
            x = F.dropout(x, self.opt['aggreg_dropout'], training=self.training)
        y = self.predictor(x)
        out = {'rel_scores': y}
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BlockTucker,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CompactBilinearPooling,
     lambda: ([], {'input1_size': 4, 'input2_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CountSketch,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearSum,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MFB,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MFH,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MLB,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mutan,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Cadene_block_bootstrap_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

