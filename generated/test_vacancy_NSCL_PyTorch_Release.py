import sys
_module = sys.modules[__name__]
del sys
desc_nscl_derender = _module
nscl = _module
configs = _module
common = _module
datasets = _module
clevr = _module
definition = _module
program_translator = _module
filterable = _module
program_analysis = _module
program_executor = _module
scene_annotation = _module
vocab = _module
factory = _module
models = _module
reasoning_v1 = _module
utils = _module
nn = _module
embedding = _module
losses = _module
concept_embedding = _module
concept_embedding_ls = _module
losses = _module
quasi_symbolic = _module
quasi_symbolic_debug = _module
scene_graph = _module
functional = _module
object_repr = _module
scene_graph = _module
scene_graph_groundtruth = _module
trainval = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch


import torch.nn.functional as F


import time


import torch.backends.cudnn as cudnn


import torch.cuda as cuda


def get_global_definition():
    global _GLOBAL_DEF
    assert _GLOBAL_DEF is not None
    return _GLOBAL_DEF


class GlobalDefinitionWrapper(object):

    def __getattr__(self, item):
        return getattr(get_global_definition(), item)

    def __setattr__(self, key, value):
        raise AttributeError('Cannot set the attr of `gdef`.')


gdef = GlobalDefinitionWrapper()


class ReasoningV1Model(nn.Module):

    def __init__(self, vocab, configs):
        super().__init__()
        self.vocab = vocab
        self.resnet = resnet.resnet34(pretrained=True, incl_gap=False, num_classes=None)
        self.resnet.layer4 = jacnn.Identity()
        self.scene_graph = sng.SceneGraph(256, configs.model.sg_dims, 16)
        self.reasoning = qs.DifferentiableReasoning(self._make_vse_concepts(configs.model.vse_large_scale, configs.model.vse_known_belong), self.scene_graph.output_dims, configs.model.vse_hidden_dims)
        self.scene_loss = vqa_losses.SceneParsingLoss(gdef.all_concepts, add_supervision=configs.train.scene_add_supervision)
        self.qa_loss = vqa_losses.QALoss(add_supervision=configs.train.qa_add_supervision)

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, large_scale, known_belong):
        if large_scale:
            return {'attribute_ls': {'attributes': list(gdef.ls_attributes), 'concepts': list(gdef.ls_concepts)}, 'relation_ls': {'attributes': None, 'concepts': list(gdef.ls_relational_concepts)}, 'embeddings': gdef.get_ls_concept_embeddings()}
        return {'attribute': {'attributes': list(gdef.attribute_concepts.keys()) + ['others'], 'concepts': [(v, k if known_belong else None) for k, vs in gdef.attribute_concepts.items() for v in vs]}, 'relation': {'attributes': list(gdef.relational_concepts.keys()) + ['others'], 'concepts': [(v, k if known_belong else None) for k, vs in gdef.relational_concepts.items() for v in vs]}}


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).

    Adapted from: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/learned_positional_embedding.py.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, left_pad=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1


class SigmoidCrossEntropy(nn.Module):

    def __init__(self, one_hot=False):
        super().__init__()
        self.one_hot = one_hot
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        if not self.one_hot:
            target = jactorch.one_hot_nd(target, input.size(-1))
        return self.bce(input, target).sum(dim=-1).mean()


class MultilabelSigmoidCrossEntropy(nn.Module):

    def __init__(self, one_hot=False):
        super().__init__()
        self.one_hot = one_hot
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, labels):
        if type(labels) in (tuple, list):
            labels = torch.tensor(labels, dtype=torch.int64, device=input.device)
        assert input.dim() == 1
        if not self.one_hot:
            with torch.no_grad():
                mask = torch.zeros_like(input)
                if labels.size(0) > 0:
                    ones = torch.ones_like(labels, dtype=torch.float32)
                    mask.scatter_(0, labels, ones)
            labels = mask
        return self.bce(input, labels).sum(dim=-1).mean()


class MultitaskLossBase(nn.Module):

    def __init__(self):
        super().__init__()
        self._sigmoid_xent_loss = SigmoidCrossEntropy()
        self._multilabel_sigmoid_xent_loss = MultilabelSigmoidCrossEntropy()

    def _mse_loss(self, pred, label):
        return (pred - label).abs()

    def _bce_loss(self, pred, label):
        return -(jactorch.log_sigmoid(pred) * label + jactorch.log_sigmoid(-pred) * (1 - label)).mean()

    def _xent_loss(self, pred, label):
        logp = F.log_softmax(pred, dim=-1)
        return -logp[label].mean()


class AttributeBlock(nn.Module):
    """Attribute as a neural operator."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.map = jacnn.LinearLayer(input_dim, output_dim, activation=None)


class ConceptBlock(nn.Module):
    """
    Concept as an embedding in the corresponding attribute space.
    """

    def __init__(self, embedding_dim, nr_attributes, attribute_agnostic=False):
        """

        Args:
            embedding_dim (int): dimension of the embedding.
            nr_attributes (int): number of known attributes.
            attribute_agnostic (bool): if the embedding in different embedding spaces are shared or not.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nr_attributes = nr_attributes
        self.attribute_agnostic = attribute_agnostic
        if self.attribute_agnostic:
            self.embedding = nn.Parameter(torch.randn(embedding_dim))
        else:
            self.embedding = nn.Parameter(torch.randn(nr_attributes, embedding_dim))
        self.belong = nn.Parameter(torch.randn(nr_attributes) * 0.1)
        self.known_belong = False

    def set_belong(self, belong_id):
        """
        Set the attribute that this concept belongs to.

        Args:
            belong_id (int): the id of the attribute.

        """
        self.belong.data.fill_(-100)
        self.belong.data[belong_id] = 100
        self.belong.requires_grad = False
        self.known_belong = True

    @property
    def normalized_embedding(self):
        """L2-normalized embedding in all spaces."""
        embedding = self.embedding / self.embedding.norm(2, dim=-1, keepdim=True)
        if self.attribute_agnostic:
            return jactorch.broadcast(embedding.unsqueeze(0), 0, self.nr_attributes)
        return embedding

    @property
    def log_normalized_belong(self):
        """Log-softmax-normalized belong vector."""
        return F.log_softmax(self.belong, dim=-1)

    @property
    def normalized_belong(self):
        """Softmax-normalized belong vector."""
        return F.softmax(self.belong, dim=-1)


_apply_self_mask = {'relate': True, 'relate_ae': True}


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + -10 * self_mask


class ConceptQuantizationContext(nn.Module):

    def __init__(self, attribute_taxnomy, relation_taxnomy, training=False, quasi=False):
        """
        Args:
            attribute_taxnomy: attribute-level concept embeddings.
            relation_taxnomy: relation-level concept embeddings.
            training (bool): training mode or not.
            quasi(bool): if False, quantize the results as 0/1.

        """
        super().__init__()
        self.attribute_taxnomy = attribute_taxnomy
        self.relation_taxnomy = relation_taxnomy
        self.quasi = quasi
        super().train(training)

    def forward(self, f_sng):
        batch_size = len(f_sng)
        output_list = [dict() for i in range(batch_size)]
        for i in range(batch_size):
            f = f_sng[i][1]
            nr_objects = f.size(0)
            output_list[i]['filter'] = dict()
            for concept in self.attribute_taxnomy.all_concepts:
                scores = self.attribute_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['filter'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['filter'][concept] = (scores > 0).nonzero().squeeze(-1).cpu().tolist()
            output_list[i]['relate_ae'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                cross_scores = self.attribute_taxnomy.cross_similarity(f, attr)
                if _apply_self_mask['relate_ae']:
                    cross_scores = do_apply_self_mask(cross_scores)
                if self.quasi:
                    output_list[i]['relate_ae'][attr] = cross_scores.detach().cpu().numpy()
                else:
                    cross_scores = cross_scores > 0
                    output_list[i]['relate_ae'][attr] = cross_scores.nonzero().cpu().tolist()
            output_list[i]['query'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                scores, word2idx = self.attribute_taxnomy.query_attribute(f, attr)
                idx2word = {v: k for k, v in word2idx.items()}
                if self.quasi:
                    output_list[i]['query'][attr] = scores.detach().cpu().numpy(), idx2word
                else:
                    argmax = scores.argmax(-1)
                    output_list[i]['query'][attr] = [idx2word[v] for v in argmax.cpu().tolist()]
            f = f_sng[i][2]
            output_list[i]['relate'] = dict()
            for concept in self.relation_taxnomy.all_concepts:
                scores = self.relation_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['relate'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['relate'][concept] = (scores > 0).nonzero().cpu().tolist()
            output_list[i]['nr_objects'] = nr_objects
        return output_list


class ProgramExecutorContext(nn.Module):

    def __init__(self, attribute_taxnomy, relation_taxnomy, features, parameter_resolution, training=True):
        super().__init__()
        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy]
        self._concept_groups_masks = [None, None, None]
        self._attribute_groups_masks = None
        self._attribute_query_masks = None
        self._attribute_query_ls_masks = None
        self._attribute_query_ls_mc_masks = None
        self.train(training)

    def filter(self, selected, group, concept_groups):
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_most(self, selected, group, concept_groups):
        mask = self._get_concept_groups_masks(concept_groups, 2)
        mask = torch.min(mask, selected.unsqueeze(-1).unsqueeze(0)).max(dim=-2)[0]
        mask = torch.min(selected, -mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def relate(self, selected, group, concept_groups):
        mask = self._get_concept_groups_masks(concept_groups, 2)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def relate_ae(self, selected, group, attribute_groups):
        mask = self._get_attribute_groups_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def unique(self, selected):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            return jacf.general_softmax(selected, impl='standard', training=self.training)
        return jacf.general_softmax(selected, impl='gumbel_hard', training=self.training)

    def intersect(self, selected1, selected2):
        return torch.min(selected1, selected2)

    def union(self, selected1, selected2):
        return torch.max(selected1, selected2)

    def exist(self, selected):
        return selected.max(dim=-1)[0]

    def belong_to(self, selected1, selected2):
        return (selected1 * selected2).sum(dim=-1)

    def count(self, selected):
        if self.training:
            return torch.sigmoid(selected).sum(dim=-1)
        else:
            if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                return (selected > 0).float().sum()
            return torch.sigmoid(selected).sum(dim=-1).round()
    _count_margin = 0.25
    _count_tau = 0.25

    def count_greater(self, selected1, selected2):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            a = torch.sigmoid(selected1).sum(dim=-1)
            b = torch.sigmoid(selected2).sum(dim=-1)
            return (a - b - 1 + 2 * self._count_margin) / self._count_tau
        else:
            return -10 + 20 * (self.count(selected1) > self.count(selected2)).float()

    def count_less(self, selected1, selected2):
        return self.count_greater(selected2, selected1)

    def count_equal(self, selected1, selected2):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            a = torch.sigmoid(selected1).sum(dim=-1)
            b = torch.sigmoid(selected2).sum(dim=-1)
            return (2 * self._count_margin - (a - b).abs()) / (2 * self._count_margin) / self._count_tau
        else:
            return -10 + 20 * (self.count(selected1) == self.count(selected2)).float()

    def query(self, selected, group, attribute_groups):
        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_ls(self, selected, group, attribute_groups):
        """large-scale query"""
        mask, word2idx = self._get_attribute_query_ls_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_ls_mc(self, selected, group, attribute_groups, concepts):
        mask, word2idx = self._get_attribute_query_ls_mc_masks(attribute_groups, concepts)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def query_is(self, selected, group, concept_groups):
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = (mask * selected.unsqueeze(0)).sum(dim=-1)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def query_ae(self, selected1, selected2, group, attribute_groups):
        mask = self._get_attribute_groups_masks(attribute_groups)
        mask = (mask * selected1.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        mask = (mask * selected2.unsqueeze(0)).sum(dim=-1)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def _get_concept_groups_masks(self, concept_groups, k):
        if self._concept_groups_masks[k] is None:
            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[k].similarity(self.features[k], c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_attribute_groups_masks(self, attribute_groups):
        if self._attribute_groups_masks is None:
            masks = list()
            for attribute in attribute_groups:
                mask = self.taxnomy[1].cross_similarity(self.features[1], attribute)
                if _apply_self_mask['relate_ae']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._attribute_groups_masks = torch.stack(masks, dim=0)
        return self._attribute_groups_masks

    def _get_attribute_query_masks(self, attribute_groups):
        if self._attribute_query_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                if word2idx is not None:
                    for k in word2idx:
                        assert word2idx[k] == this_word2idx[k]
                word2idx = this_word2idx
            self._attribute_query_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_masks

    def _get_attribute_query_ls_masks(self, attribute_groups):
        if self._attribute_query_ls_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                word2idx = this_word2idx
            self._attribute_query_ls_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_masks

    def _get_attribute_query_ls_mc_masks(self, attribute_groups, concepts):
        if self._attribute_query_ls_mc_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute_mc(self.features[1], attribute, concepts)
                masks.append(mask)
                word2idx = this_word2idx
            self._attribute_query_ls_mc_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_mc_masks


class DifferentiableReasoning(nn.Module):

    def __init__(self, used_concepts, input_dims, hidden_dims, parameter_resolution='deterministic', vse_attribute_agnostic=False):
        super().__init__()
        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution
        for i, nr_vars in enumerate(['attribute', 'relation']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]
            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for v, b in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)
        for i, nr_vars in enumerate(['attribute_ls', 'relation_ls']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars.replace('_ls', ''), concept_embedding_ls.ConceptEmbeddingLS(self.input_dims[1 + i], self.hidden_dims[1 + i], self.hidden_dims[1 + i]))
            tax = getattr(self, 'embedding_' + nr_vars.replace('_ls', ''))
            rec = self.used_concepts[nr_vars]
            if rec['attributes'] is not None:
                tax.init_attributes(rec['attributes'], self.used_concepts['embeddings'])
            if rec['concepts'] is not None:
                tax.init_concepts(rec['concepts'], self.used_concepts['embeddings'])

    def forward(self, batch_features, progs, fd=None):
        assert len(progs) == len(batch_features)
        programs = []
        buffers = []
        result = []
        for i, (features, prog) in enumerate(zip(batch_features, progs)):
            buffer = []
            buffers.append(buffer)
            programs.append(prog)
            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, features, parameter_resolution=self.parameter_resolution, training=self.training)
            for block_id, block in enumerate(prog):
                op = block['op']
                if op == 'scene':
                    buffer.append(10 + torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device))
                    continue
                inputs = []
                for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                    inp = buffer[inp]
                    if inp_type == 'object':
                        inp = ctx.unique(inp)
                    inputs.append(inp)
                if op == 'filter':
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'filter_scene':
                    inputs = [10 + torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device)]
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'filter_most':
                    buffer.append(ctx.filter_most(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                elif op == 'relate':
                    buffer.append(ctx.relate(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                elif op == 'relate_attribute_equal':
                    buffer.append(ctx.relate_ae(*inputs, block['attribute_idx'], block['attribute_values']))
                elif op == 'intersect':
                    buffer.append(ctx.intersect(*inputs))
                elif op == 'union':
                    buffer.append(ctx.union(*inputs))
                else:
                    assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                    if op == 'query':
                        buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'query_ls':
                        buffer.append(ctx.query_ls(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'query_ls_mc':
                        buffer.append(ctx.query_ls_mc(*inputs, block['attribute_idx'], block['attribute_values'], block['multiple_choices']))
                    elif op == 'query_is':
                        buffer.append(ctx.query_is(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'query_attribute_equal':
                        buffer.append(ctx.query_ae(*inputs, block['attribute_idx'], block['attribute_values']))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'belong_to':
                        buffer.append(ctx.belong_to(*inputs))
                    elif op == 'count':
                        buffer.append(ctx.count(*inputs))
                    elif op == 'count_greater':
                        buffer.append(ctx.count_greater(*inputs))
                    elif op == 'count_less':
                        buffer.append(ctx.count_less(*inputs))
                    elif op == 'count_equal':
                        buffer.append(ctx.count_equal(*inputs))
                    else:
                        raise NotImplementedError('Unsupported operation: {}.'.format(op))
                if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                    if block_id != len(prog) - 1:
                        buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
            result.append((op, buffer[-1]))
            quasi_symbolic_debug.embed(self, i, buffer, result, fd)
        return programs, buffers, result


class ObjectBasedRepresentation(nn.Module):

    def __init__(self, feature_dim, downsample_rate, pool_size=7):
        super().__init__()
        self.pool_size = pool_size
        self.feature_dim = feature_dim
        self.downsample_rate = downsample_rate
        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

    def forward(self, input, objects, objects_length):
        object_features = input
        context_features = input
        outputs = list()
        objects_index = 0
        for i in range(input.size(0)):
            box = objects[objects_index:objects_index + objects_length[i].item()]
            objects_index += objects_length[i].item()
            with torch.no_grad():
                batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
                image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                image_box = torch.cat([torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device), torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device), image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device), image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)], dim=-1)
                box_context_imap = functional.generate_intersection_map(box, image_box, self.pool_size)
            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1)[:1])
            this_object_features = torch.cat([self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1))], dim=1)
            outputs.append((this_object_features, this_context_features[0], box_context_imap))
        return outputs

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)


class SceneGraph(nn.Module):

    def __init__(self, feature_dim, output_dims, downsample_rate):
        super().__init__()
        self.pool_size = 7
        self.feature_dim = feature_dim
        self.output_dims = output_dims
        self.downsample_rate = downsample_rate
        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        if not DEBUG:
            self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
            self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)
            self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
            self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)
            self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))
            self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))
            self.reset_parameters()
        else:

            def gen_replicate(n):

                def rep(x):
                    return torch.cat([x for _ in range(n)], dim=1)
                return rep
            self.pool_size = 32
            self.object_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.relation_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_feature_extract = gen_replicate(2)
            self.relation_feature_extract = gen_replicate(3)
            self.object_feature_fuse = jacnn.Identity()
            self.relation_feature_fuse = jacnn.Identity()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input, objects, objects_length):
        object_features = input
        context_features = self.context_feature_extract(input)
        relation_features = self.relation_feature_extract(input)
        outputs = list()
        objects_index = 0
        for i in range(input.size(0)):
            box = objects[objects_index:objects_index + objects_length[i].item()]
            objects_index += objects_length[i].item()
            with torch.no_grad():
                batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)
                image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                image_box = torch.cat([torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device), torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device), image_w + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device), image_h + torch.zeros(box.size(0), 1, dtype=box.dtype, device=box.device)], dim=-1)
                sub_id, obj_id = jactorch.meshgrid(torch.arange(box.size(0), dtype=torch.int64, device=box.device), dim=0)
                sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)
                sub_box, obj_box = jactorch.meshgrid(box, dim=0)
                sub_box = sub_box.contiguous().view(box.size(0) ** 2, 4)
                obj_box = obj_box.contiguous().view(box.size(0) ** 2, 4)
                union_box = functional.generate_union_box(sub_box, obj_box)
                rel_batch_ind = i + torch.zeros(union_box.size(0), 1, dtype=box.dtype, device=box.device)
                box_context_imap = functional.generate_intersection_map(box, image_box, self.pool_size)
                sub_union_imap = functional.generate_intersection_map(sub_box, union_box, self.pool_size)
                obj_union_imap = functional.generate_intersection_map(obj_box, union_box, self.pool_size)
            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
            x, y = this_context_features.chunk(2, dim=1)
            this_object_features = self.object_feature_fuse(torch.cat([self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1)), x, y * box_context_imap], dim=1))
            this_relation_features = self.relation_roi_pool(relation_features, torch.cat([rel_batch_ind, union_box], dim=-1))
            x, y, z = this_relation_features.chunk(3, dim=1)
            this_relation_features = self.relation_feature_fuse(torch.cat([this_object_features[sub_id], this_object_features[obj_id], x, y * sub_union_imap, z * obj_union_imap], dim=1))
            if DEBUG:
                outputs.append([None, this_object_features, this_relation_features])
            else:
                outputs.append([None, self._norm(self.object_feature_fc(this_object_features.view(box.size(0), -1))), self._norm(self.relation_feature_fc(this_relation_features.view(box.size(0) * box.size(0), -1)).view(box.size(0), box.size(0), -1))])
        return outputs

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)


class SceneGraphGroundtruth(nn.Module):

    def __init__(self, vocab, used_concepts):
        super().__init__()
        self.vocab = vocab
        self.used_concepts = used_concepts
        self.output_dims = [None, 0, 4]
        self.register_buffer('global2local', torch.zeros(len(self.vocab), dtype=torch.int64))
        for k, v in self.used_concepts.items():
            if v['type'] != 'attribute':
                continue
            self.output_dims[1] += len(v['values'])
            v = v['values']
            self.register_buffer('local2global_{}'.format(k), torch.zeros(len(v), dtype=torch.int64))
            for i, vv in enumerate(v):
                self.global2local[vocab.word2idx[vv]] = i
                getattr(self, 'local2global_{}'.format(k))[i] = vocab.word2idx[vv]

    def forward(self, input, objects, objects_length, feed_dict):
        objects_index = 0
        relation_index = 0
        outputs = []
        for i in range(input.size(0)):
            nr_objects = objects_length[i].item()
            object_features = []
            for attribute, info in self.used_concepts.items():
                if info['type'] == 'attribute':
                    values = feed_dict['objects_' + attribute][objects_index:objects_index + nr_objects]
                    mapped_values = self._valmap(self.global2local, values)
                    object_features.append(jactorch.one_hot(mapped_values, len(info['values'])))
            object_features = torch.cat(object_features, dim=-1)
            object_features = object_features.float()
            relation_features = feed_dict['relations_spatial_relation'][relation_index:relation_index + nr_objects * nr_objects]
            relation_features = relation_features.float()
            outputs.append((None, object_features, relation_features.view(nr_objects, nr_objects, 4)))
            objects_index += nr_objects
            relation_index += nr_objects * nr_objects
        return outputs

    @staticmethod
    def _valmap(a, i):
        return a[i.view(-1)].view_as(i)

