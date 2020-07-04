import sys
_module = sys.modules[__name__]
del sys
master = _module
Language_Model = _module
MSDN = _module
MSDN_base = _module
RPN = _module
faster_rcnn = _module
datasets = _module
visual_genome_loader = _module
fast_rcnn = _module
bbox_transform = _module
config = _module
hierarchical_message_passing_structure = _module
mps_base = _module
nms_wrapper = _module
network = _module
nms = _module
py_cpu_nms = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
roi_pooling = _module
_ext = _module
build = _module
functions = _module
roi_pool = _module
modules = _module
roi_pool = _module
roi_pool_py = _module
rpn_msr = _module
anchor_target_layer = _module
generate = _module
generate_anchors = _module
proposal_layer = _module
proposal_target_layer = _module
proposal_target_layer_hdn = _module
setup = _module
HDN_utils = _module
utils = _module
blob = _module
boxes_grid = _module
timer = _module
vgg16 = _module
preprocessing_data = _module
train_hdn = _module
train_rpn = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import numpy.random as npr


import torch.utils.model_zoo as model_zoo


import torchvision.models as models


import math


from torch.nn import Parameter


from torch.nn.modules.module import Module


class Img_Encoder_Structure(nn.Module):

    def __init__(self, ninput, nembed, nhidden, nlayers, bias, dropout):
        super(Img_Encoder_Structure, self).__init__()
        self.image_encoder = FC(ninput, nembed, relu=True)
        self.rnn = nn.LSTM(nembed, nhidden, nlayers, bias=bias, dropout=dropout
            )

    def forward(self, feat_im_with_seq_dim):
        feat_im = self.image_encoder(feat_im_with_seq_dim[0])
        output, feat_im = self.rnn(feat_im.unsqueeze(0))
        return output, feat_im


class Language_Model(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, nimg, nhidden, nembed, nlayers,
        nseq, voc_sign, bias=False, dropout=0.0):
        super(Language_Model, self).__init__()
        self.encoder = nn.Embedding(ntoken, nembed)
        if rnn_type == 'LSTM_im':
            self.lstm_im = nn.LSTM(nimg, nhidden, nlayers, bias=bias,
                dropout=dropout)
            self.lstm_word = nn.LSTM(nembed, nhidden, nlayers, bias=bias,
                dropout=dropout)
        elif rnn_type == 'LSTM_normal':
            self.lstm_im = nn.LSTM(nimg, nhidden, nlayers, bias=bias,
                dropout=dropout)
            self.lstm_word = self.lstm_im
        elif rnn_type == 'LSTM_baseline':
            self.lstm_im = Img_Encoder_Structure(nimg, nembed, nhidden,
                nlayers, bias=bias, dropout=dropout)
            self.lstm_word = self.lstm_im.rnn
        else:
            raise Exception('Cannot recognize LSTM type')
        self.decoder = nn.Linear(nhidden, ntoken, bias=bias)
        self.nseq = nseq
        self.end = voc_sign['end']
        self.null = voc_sign['null']
        self.start = voc_sign['start']
        self.word_weight = torch.ones(ntoken)
        self.word_weight[self.null] = 0.0
        self.word_weight[self.end] = 0.1
        self.ntoken = ntoken
        self.bias = bias
        self.init_weights()
        self.nlayers = nlayers
        self.nhidden = nhidden
        self.nembed = nembed

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, seq=None):
        if self.training:
            seq = torch.t(seq)
            im_batch_size = input.size()[0]
            im_feature_size = input.size()[1]
            seq_batch_size = seq.size()[1]
            seq_len = [(np.where(seq[:, (i)].cpu().data.numpy() == self.end
                )[0][0] + 1) for i in range(seq.size(1))]
            input_seq = seq[:max(seq_len) - 1]
            target_seq = seq[1:max(seq_len)].clone()
            output_mask = input_seq.eq(self.end)
            target_seq[output_mask] = self.null
            seq_embed = self.encoder(input_seq)
            hidden_feat = self.lstm_im(input.view(1, im_batch_size,
                im_feature_size).expand(1, seq_batch_size, im_feature_size))[1]
            output, hidden_feat = self.lstm_word(seq_embed, hidden_feat)
            output = self.decoder(output.view(-1, output.size(2)))
            loss = F.cross_entropy(output, target_seq.view(-1), weight=self
                .word_weight)
            return loss
        else:
            batch_size = input.size(0)
            hidden_feat = self.lstm_im(input.view(1, input.size()[0], input
                .size()[1]))[1]
            x = Variable(torch.ones(1, batch_size).type(torch.LongTensor) *
                self.start, requires_grad=False)
            output = []
            scores = torch.zeros(batch_size)
            flag = torch.ones(batch_size)
            for i in range(self.nseq):
                input_x = self.encoder(x.view(1, -1))
                output_feature, hidden_feat = self.lstm_word(input_x,
                    hidden_feat)
                output_t = self.decoder(output_feature.view(-1,
                    output_feature.size(2)))
                output_t = F.log_softmax(output_t)
                logprob, x = output_t.max(1)
                output.append(x)
                scores += logprob.cpu().data * flag
                flag[x.cpu().eq(self.end).data] = 0
                if flag.sum() == 0:
                    break
            output = torch.stack(output, 0).squeeze().transpose(0, 1)
            return output

    def baseline_search(self, input, beam_size=None):
        batch_size = input.size(0)
        hidden_feat = self.lstm_im(input.view(1, input.size()[0], input.
            size()[1]))[1]
        x = Variable(torch.ones(1, batch_size).type(torch.LongTensor) *
            self.start, requires_grad=False)
        output = []
        flag = torch.ones(batch_size)
        for i in range(self.nseq):
            input_x = self.encoder(x.view(1, -1))
            output_feature, hidden_feat = self.lstm_word(input_x, hidden_feat)
            output_t = self.decoder(output_feature.view(-1, output_feature.
                size(2)))
            output_t = F.log_softmax(output_t)
            logprob, x = output_t.max(1)
            output.append(x)
            flag[x.cpu().eq(self.end).data] = 0
            if flag.sum() == 0:
                break
        output = torch.stack(output, 0).squeeze().transpose(0, 1).cpu().data
        return output

    def beamsearch(self, img_feat, beam_size):
        raise NotImplementedError
        feature_len = img_feat.size(1)
        batch_size = img_feat.size(0)
        hidden_feat = self.lstm_im(img_feat.unsqueeze(0))[1]
        batch_size = img_feat.size(0)
        seq = torch.LongTensor(self.nseq, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(batch_size)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = hidden_feat[0][:, k:k + 1].expand(self.nlayers,
                beam_size, self.nhidden).contiguous(), hidden_feat[1][:, k:
                k + 1].expand(self.nlayers, beam_size, self.nhidden
                ).contiguous()
            beam_seq = torch.LongTensor(self.nseq, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.nseq, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            beam_end_state = torch.zeros(beam_size).type(torch.ByteTensor)
            for t in range(self.nseq + 1):
                if t == 0:
                    it = Variable(torch.ones(beam_size).type(torch.
                        LongTensor) * self.start, requires_grad=False)
                    xt = self.encoder(it)
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.cpu().data
                    ys, ix = logprobsf.topk(beam_size, 1, True)
                    candidates = []
                    assert ys.size(1) == beam_size
                    cols = beam_size
                    rows = beam_size
                    if t == 1:
                        rows = 1
                    for q in range(rows):
                        for c in range(cols):
                            if beam_end_state[q]:
                                local_logprob = 0.0
                                candidate_logprob = beam_logprobs_sum[q]
                                candidates.append({'c': self.end, 'q': q,
                                    'p': candidate_logprob, 'r': local_logprob}
                                    )
                                break
                            else:
                                local_logprob = ys[q, c]
                                candidate_logprob = (beam_logprobs_sum[q] *
                                    (t - 1) + local_logprob) / float(t)
                                candidates.append({'c': ix[q, c], 'q': q,
                                    'p': candidate_logprob, 'r': local_logprob}
                                    )
                    candidates = sorted(candidates, key=lambda x: -x['p'])
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1
                            ].clone()
                    beam_end_state_prev = beam_end_state.clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        beam_end_state[vix] = True if v['c'
                            ] == self.end else beam_end_state_prev[v['q']]
                        if t > 1:
                            beam_seq[:t - 1, (vix)] = beam_seq_prev[:, (v['q'])
                                ]
                            beam_seq_logprobs[:t - 1, (vix)
                                ] = beam_seq_logprobs_prev[:, (v['q'])]
                        for state_ix in range(len(new_state)):
                            new_state[state_ix][0, vix] = state[state_ix][0,
                                v['q']]
                        beam_seq[t - 1, vix] = v['c']
                        beam_seq_logprobs[t - 1, vix] = v['r']
                        try:
                            beam_logprobs_sum[vix] = v['p']
                        except Exception:
                            pdb.set_trace()
                    if beam_end_state.all() or t == self.nseq:
                        for vix in range(beam_size):
                            self.done_beams[k].append({'seq': beam_seq[:, (
                                vix)].clone(), 'logps': beam_seq_logprobs[:,
                                (vix)].clone(), 'p': beam_logprobs_sum[vix]})
                        break
                    it = beam_seq[t - 1]
                    xt = self.encoder(Variable(it))
                if t >= 1:
                    state = new_state
                output, state = self.lstm_word(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.decoder(output.squeeze(0)))
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -
                x['p'])
            seq[:, (k)] = self.done_beams[k][0]['seq']
            seqLogprobs[k] = self.done_beams[k][0]['p']
        return seq.transpose(0, 1)

    def init_hidden(self, bsz):
        return Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()
            ), Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()
            )


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


_global_config['TRAIN'] = 4


def bbox_transform_inv_hdn(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, (2)] - boxes[:, (0)] + 1.0
    heights = boxes[:, (3)] - boxes[:, (1)] + 1.0
    ctr_x = boxes[:, (0)] + 0.5 * widths
    ctr_y = boxes[:, (1)] + 0.5 * heights
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        deltas = deltas * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS) + np.array(
            cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    pred_ctr_x = dx * widths[:, (np.newaxis)] + ctr_x[:, (np.newaxis)]
    pred_ctr_y = dy * heights[:, (np.newaxis)] + ctr_y[:, (np.newaxis)]
    pred_w = np.exp(dw) * widths[:, (np.newaxis)]
    pred_h = np.exp(dh) * heights[:, (np.newaxis)]
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1.0
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1.0
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.
        float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[(i), 0:im.shape[0], 0:im.shape[1], :] = im
    return blob


DEBUG = False


class RPN(nn.Module):
    _feat_stride = [16]
    anchor_scales_kmeans = [19.944, 9.118, 35.648, 42.102, 23.476, 15.882, 
        6.169, 9.702, 6.072, 32.254, 3.294, 10.148, 22.443, 13.831, 16.25, 
        27.969, 14.181, 27.818, 34.146, 29.812, 14.219, 22.309, 20.36, 
        24.025, 40.593]
    anchor_ratios_kmeans = [2.631, 2.304, 0.935, 0.654, 0.173, 0.72, 0.553,
        0.374, 1.565, 0.463, 0.985, 0.914, 0.734, 2.671, 0.209, 1.318, 
        1.285, 2.717, 0.369, 0.718, 0.319, 0.218, 1.319, 0.442, 1.437]
    anchor_scales_kmeans_region = [18.865, 27.466, 35.138, 9.383, 34.77, 
        31.223, 14.003, 40.663, 20.187, 6.062, 31.354, 21.213, 19.379, 
        9.843, 5.98, 3.271, 14.7, 12.794, 25.936, 24.221, 9.69, 27.328, 
        41.85, 16.087, 23.949]
    anchor_ratios_kmeans_region = [2.796, 2.81, 0.981, 0.416, 0.381, 0.422,
        2.358, 1.445, 1.298, 1.69, 0.68, 0.201, 0.636, 0.979, 0.59, 1.006, 
        0.956, 0.327, 0.872, 0.455, 2.201, 1.478, 0.657, 0.224, 0.181]
    anchor_scales_normal = [2, 4, 8, 16, 32, 64]
    anchor_ratios_normal = [0.25, 0.5, 1, 2, 4]
    anchor_scales_normal_region = [4, 8, 16, 32, 64]
    anchor_ratios_normal_region = [0.25, 0.5, 1, 2, 4]

    def __init__(self, use_kmeans_anchors=False):
        super(RPN, self).__init__()
        if use_kmeans_anchors:
            None
            self.anchor_scales = self.anchor_scales_kmeans
            self.anchor_ratios = self.anchor_ratios_kmeans
            self.anchor_scales_region = self.anchor_scales_kmeans_region
            self.anchor_ratios_region = self.anchor_ratios_kmeans_region
        else:
            None
            self.anchor_scales, self.anchor_ratios = np.meshgrid(self.
                anchor_scales_normal, self.anchor_ratios_normal, indexing='ij')
            self.anchor_scales = self.anchor_scales.reshape(-1)
            self.anchor_ratios = self.anchor_ratios.reshape(-1)
            self.anchor_scales_region, self.anchor_ratios_region = np.meshgrid(
                self.anchor_scales_normal_region, self.
                anchor_ratios_normal_region, indexing='ij')
            self.anchor_scales_region = self.anchor_scales_region.reshape(-1)
            self.anchor_ratios_region = self.anchor_ratios_region.reshape(-1)
        self.anchor_num = len(self.anchor_scales)
        self.anchor_num_region = len(self.anchor_scales_region)
        self.features = models.vgg16(pretrained=True).features
        self.features.__delattr__('30')
        network.set_trainable_param(list(self.features.parameters())[:8],
            requires_grad=False)
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, self.anchor_num * 2, 1, relu=False,
            same_padding=False)
        self.bbox_conv = Conv2d(512, self.anchor_num * 4, 1, relu=False,
            same_padding=False)
        self.conv1_region = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv_region = Conv2d(512, self.anchor_num_region * 2, 1,
            relu=False, same_padding=False)
        self.bbox_conv_region = Conv2d(512, self.anchor_num_region * 4, 1,
            relu=False, same_padding=False)
        self.cross_entropy = None
        self.loss_box = None
        self.cross_entropy_region = None
        self.loss_box_region = None
        self.initialize_parameters()

    def initialize_parameters(self, normal_method='normal'):
        if normal_method == 'normal':
            normal_fun = network.weights_normal_init
        elif normal_method == 'MSRA':
            normal_fun = network.weights_MSRA_init
        else:
            raise Exception
        normal_fun(self.conv1, 0.025)
        normal_fun(self.score_conv, 0.025)
        normal_fun(self.bbox_conv, 0.01)
        normal_fun(self.conv1_region, 0.025)
        normal_fun(self.score_conv_region, 0.025)
        normal_fun(self.bbox_conv_region, 0.01)

    @property
    def loss(self):
        return (self.cross_entropy + self.loss_box * 0.5 + self.
            cross_entropy_region + 1.0 * self.loss_box_region)

    def forward(self, im_data, im_info, gt_objects=None, gt_regions=None,
        dontcare_areas=None):
        im_data = Variable(im_data)
        features = self.features(im_data)
        rpn_conv1 = self.conv1(features)
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, self.
            anchor_num * 2)
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        rpn_conv1_region = self.conv1_region(features)
        rpn_cls_score_region = self.score_conv(rpn_conv1_region)
        rpn_cls_score_region_reshape = self.reshape_layer(rpn_cls_score_region,
            2)
        rpn_cls_prob_region = F.softmax(rpn_cls_score_region_reshape)
        rpn_cls_prob_region_reshape = self.reshape_layer(rpn_cls_prob_region,
            self.anchor_num * 2)
        rpn_bbox_pred_region = self.bbox_conv(rpn_conv1_region)
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred,
            im_info, cfg_key, self._feat_stride, self.anchor_scales, self.
            anchor_ratios, is_region=False)
        region_rois = self.proposal_layer(rpn_cls_prob_region_reshape,
            rpn_bbox_pred_region, im_info, cfg_key, self._feat_stride, self
            .anchor_scales_region, self.anchor_ratios_region, is_region=True)
        if self.training:
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_objects,
                dontcare_areas, im_info, self.anchor_scales, self.
                anchor_ratios, self._feat_stride)
            rpn_data_region = self.anchor_target_layer(rpn_cls_score_region,
                gt_regions[:, :4], dontcare_areas, im_info, self.
                anchor_scales_region, self.anchor_ratios_region, self.
                _feat_stride, is_region=True)
            if DEBUG:
                None
                None
            self.cross_entropy, self.loss_box = self.build_loss(
                rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
            self.cross_entropy_region, self.loss_box_region = self.build_loss(
                rpn_cls_score_region_reshape, rpn_bbox_pred_region,
                rpn_data_region, is_region=True)
        return features, rois, region_rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data,
        is_region=False):
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous(
            ).view(-1, 2)
        rpn_label = rpn_data[0]
        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze())
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)
        fg_cnt = torch.sum(rpn_label.data.ne(0))
        bg_cnt = rpn_label.data.numel() - fg_cnt
        _, predict = torch.max(rpn_cls_score.data, 1)
        error = torch.sum(torch.abs(predict - rpn_label.data))
        if predict.size()[0] < 256:
            None
            None
            None
        if is_region:
            self.tp_region = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:
                fg_cnt]))
            self.tf_region = torch.sum(predict[fg_cnt:].eq(rpn_label.data[
                fg_cnt:]))
            self.fg_cnt_region = fg_cnt
            self.bg_cnt_region = bg_cnt
            if DEBUG:
                None
        else:
            self.tp = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:fg_cnt]))
            self.tf = torch.sum(predict[fg_cnt:].eq(rpn_label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt
            if DEBUG:
                None
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
        (rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
            ) = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)
        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets,
            size_average=False) / (fg_cnt + 0.0001)
        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] *
            input_shape[2]) / float(d)), input_shape[3])
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
        cfg_key, _feat_stride, anchor_scales, anchor_ratios, is_region):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
            cfg_key, _feat_stride, anchor_scales, anchor_ratios, is_region=
            is_region)
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, dontcare_areas,
        im_info, _feat_stride, anchor_scales, anchor_rotios, is_region=False):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        #gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        (rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,
            rpn_bbox_outside_weights) = (anchor_target_layer_py(
            rpn_cls_score, gt_boxes, dontcare_areas, im_info, _feat_stride,
            anchor_scales, anchor_rotios, is_region=is_region))
        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype
            =torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda
            =True)
        rpn_bbox_inside_weights = network.np_to_variable(
            rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(
            rpn_bbox_outside_weights, is_cuda=True)
        return (rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,
            rpn_bbox_outside_weights)

    def load_from_npz(self, params):
        self.features.load_from_npz(params)
        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv':
            'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in list(pairs.items()):
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(
                3, 2, 0, 1)
            own_dict[key].copy_(param)
            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)


class Message_Passing_Unit_v2(nn.Module):

    def __init__(self, fea_size, filter_size=128):
        super(Message_Passing_Unit_v2, self).__init__()
        self.w = nn.Linear(fea_size, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.
                size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.
                size()[1])
        gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
        gate = F.sigmoid(gate.sum(1))
        output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])
        return output


class Message_Passing_Unit_v1(nn.Module):

    def __init__(self, fea_size, filter_size=128):
        super(Message_Passing_Unit_v1, self).__init__()
        self.w = nn.Linear(fea_size * 2, filter_size, bias=True)
        self.fea_size = fea_size
        self.filter_size = filter_size

    def forward(self, unary_term, pair_term):
        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.
                size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.
                size()[1])
        gate = torch.cat([unary_term, pair_term], 1)
        gate = F.relu(gate)
        gate = F.sigmoid(self.w(gate)).mean(1)
        output = pair_term * gate.view(-1, 1).expand(gate.size()[0],
            pair_term.size()[1])
        return output


class Gated_Recurrent_Unit(nn.Module):

    def __init__(self, fea_size, dropout):
        super(Gated_Recurrent_Unit, self).__init__()
        self.wih = nn.Linear(fea_size, fea_size, bias=True)
        self.whh = nn.Linear(fea_size, fea_size, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output


class Hierarchical_Message_Passing_Structure_base(nn.Module):

    def __init__(self, fea_size, dropout=False, gate_width=128, use_region=
        True, use_kernel_function=False):
        super(Hierarchical_Message_Passing_Structure_base, self).__init__()
        if use_kernel_function:
            Message_Passing_Unit = Message_Passing_Unit_v2
        else:
            Message_Passing_Unit = Message_Passing_Unit_v1
        self.gate_sub2pred = Message_Passing_Unit(fea_size, gate_width)
        self.gate_obj2pred = Message_Passing_Unit(fea_size, gate_width)
        self.gate_pred2sub = Message_Passing_Unit(fea_size, gate_width)
        self.gate_pred2obj = Message_Passing_Unit(fea_size, gate_width)
        self.GRU_object = Gated_Recurrent_Unit(fea_size, dropout)
        self.GRU_phrase = Gated_Recurrent_Unit(fea_size, dropout)
        if use_region:
            self.gate_pred2reg = Message_Passing_Unit(fea_size, gate_width)
            self.gate_reg2pred = Message_Passing_Unit(fea_size, gate_width)
            self.GRU_region = Gated_Recurrent_Unit(fea_size, dropout)

    def forward(self, feature_obj, feature_phrase, feature_region,
        mps_object, mps_phrase, mps_region):
        raise Exception('Please implement the forward function')

    def prepare_message(self, target_features, source_features, select_mat,
        gate_module):
        feature_data = []
        transfer_list = np.where(select_mat > 0)
        source_indices = Variable(torch.from_numpy(transfer_list[1]).type(
            torch.LongTensor))
        target_indices = Variable(torch.from_numpy(transfer_list[0]).type(
            torch.LongTensor))
        source_f = torch.index_select(source_features, 0, source_indices)
        target_f = torch.index_select(target_features, 0, target_indices)
        transferred_features = gate_module(target_f, source_f)
        for f_id in range(target_features.size()[0]):
            if len(np.where(select_mat[(f_id), :] > 0)[0]) > 0:
                feature_indices = np.where(transfer_list[0] == f_id)[0]
                indices = Variable(torch.from_numpy(feature_indices).type(
                    torch.LongTensor))
                features = torch.index_select(transferred_features, 0, indices
                    ).mean(0).view(-1)
                feature_data.append(features)
            else:
                temp = Variable(torch.zeros(target_features.size()[1:]),
                    requires_grad=True).type(torch.FloatTensor)
                feature_data.append(temp)
        return torch.stack(feature_data, 0)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):

    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RoIPoolFunction(Function):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.output = None
        self.argmax = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, num_channels, self.pooled_height,
            self.pooled_width)
        argmax = torch.IntTensor(num_rois, num_channels, self.pooled_height,
            self.pooled_width).zero_()
        if not features.is_cuda:
            assert False, 'feature not in CUDA'
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(self.pooled_height, self.
                pooled_width, self.spatial_scale, _features, rois, output)
        else:
            output = output.cuda()
            argmax = argmax.cuda()
            roi_pooling.roi_pooling_forward_cuda(self.pooled_height, self.
                pooled_width, self.spatial_scale, features, rois, output,
                argmax)
            self.output = output
            self.argmax = argmax
            self.rois = rois
            self.feature_size = features.size()
        return output

    def backward(self, grad_output):
        assert self.feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = torch.zeros(batch_size, num_channels, data_height,
            data_width).cuda()
        roi_pooling.roi_pooling_backward_cuda(self.pooled_height, self.
            pooled_width, self.spatial_scale, grad_output, self.rois,
            grad_input, self.argmax)
        return grad_input, None


class RoIPool(Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.
            spatial_scale)(features, rois)


class RoIPool(nn.Module):

    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(RoIPool, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        outputs = Variable(torch.zeros(num_rois, num_channels, self.
            pooled_height, self.pooled_width))
        for roi_ind, roi in enumerate(rois):
            batch_ind = int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(roi[1
                :].data.cpu().numpy() * self.spatial_scale).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))
                    is_empty = hend <= hstart or wend <= wstart
                    if is_empty:
                        outputs[(roi_ind), :, (ph), (pw)] = 0
                    else:
                        data = features[batch_ind]
                        outputs[(roi_ind), :, (ph), (pw)] = torch.max(torch
                            .max(data[:, hstart:hend, wstart:wend], 1)[0], 2)[0
                            ].view(-1)
        return outputs


class VGG16(nn.Module):

    def __init__(self, bn=False):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=
            bn), Conv2d(64, 64, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn
            =bn), Conv2d(128, 128, 3, same_padding=True, bn=bn), nn.
            MaxPool2d(2))
        network.set_trainable(self.conv1, requires_grad=False)
        network.set_trainable(self.conv2, requires_grad=False)
        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True,
            bn=bn), Conv2d(256, 256, 3, same_padding=True, bn=bn), Conv2d(
            256, 256, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True,
            bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(
            512, 512, 3, same_padding=True, bn=bn), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True,
            bn=bn), Conv2d(512, 512, 3, same_padding=True, bn=bn), Conv2d(
            512, 512, 3, same_padding=True, bn=bn))

    def forward(self, im_data):
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def load_from_npz(self, params):
        own_dict = self.state_dict()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}/{}:0'.format(i, j, ptype)
            param = torch.from_numpy(params[key])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)

    def load_from_npy_file(self, fname):
        own_dict = self.state_dict()
        params = np.load(fname).item()
        for name, val in own_dict.items():
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}'.format(i, j)
            param = torch.from_numpy(params[key][ptype])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yikang_li_MSDN(_paritybench_base):
    pass
    def test_000(self):
        self._check(Conv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(FC(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Gated_Recurrent_Unit(*[], **{'fea_size': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Message_Passing_Unit_v1(*[], **{'fea_size': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_004(self):
        self._check(Message_Passing_Unit_v2(*[], **{'fea_size': 4}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

