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


import torch.nn as nn


import torch


import torch.nn.functional as F


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
        positions = positions - mask.size(1) + mask.long().sum(dim=1
            ).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).

    Adapted from: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/learned_positional_embedding.py.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=0,
        left_pad=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            positions = input.data.new(1, 1).fill_(self.padding_idx + input
                .size(1))
        else:
            positions = make_positions(input.data, self.padding_idx, self.
                left_pad)
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
            labels = torch.tensor(labels, dtype=torch.int64, device=input.
                device)
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
        return -(jactorch.log_sigmoid(pred) * label + jactorch.log_sigmoid(
            -pred) * (1 - label)).mean()

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
            self.embedding = nn.Parameter(torch.randn(nr_attributes,
                embedding_dim))
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
        embedding = self.embedding / self.embedding.norm(2, dim=-1, keepdim
            =True)
        if self.attribute_agnostic:
            return jactorch.broadcast(embedding.unsqueeze(0), 0, self.
                nr_attributes)
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

    def __init__(self, attribute_taxnomy, relation_taxnomy, training=False,
        quasi=False):
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
                    output_list[i]['filter'][concept] = scores.detach().cpu(
                        ).numpy()
                else:
                    output_list[i]['filter'][concept] = (scores > 0).nonzero(
                        ).squeeze(-1).cpu().tolist()
            output_list[i]['relate_ae'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                cross_scores = self.attribute_taxnomy.cross_similarity(f, attr)
                if _apply_self_mask['relate_ae']:
                    cross_scores = do_apply_self_mask(cross_scores)
                if self.quasi:
                    output_list[i]['relate_ae'][attr] = cross_scores.detach(
                        ).cpu().numpy()
                else:
                    cross_scores = cross_scores > 0
                    output_list[i]['relate_ae'][attr] = cross_scores.nonzero(
                        ).cpu().tolist()
            output_list[i]['query'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                scores, word2idx = self.attribute_taxnomy.query_attribute(f,
                    attr)
                idx2word = {v: k for k, v in word2idx.items()}
                if self.quasi:
                    output_list[i]['query'][attr] = scores.detach().cpu(
                        ).numpy(), idx2word
                else:
                    argmax = scores.argmax(-1)
                    output_list[i]['query'][attr] = [idx2word[v] for v in
                        argmax.cpu().tolist()]
            f = f_sng[i][2]
            output_list[i]['relate'] = dict()
            for concept in self.relation_taxnomy.all_concepts:
                scores = self.relation_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['relate'][concept] = scores.detach().cpu(
                        ).numpy()
                else:
                    output_list[i]['relate'][concept] = (scores > 0).nonzero(
                        ).cpu().tolist()
            output_list[i]['nr_objects'] = nr_objects
        return output_list


class ObjectBasedRepresentation(nn.Module):

    def __init__(self, feature_dim, downsample_rate, pool_size=7):
        super().__init__()
        self.pool_size = pool_size
        self.feature_dim = feature_dim
        self.downsample_rate = downsample_rate
        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.
            pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.
            pool_size, 1.0 / downsample_rate)

    def forward(self, input, objects, objects_length):
        object_features = input
        context_features = input
        outputs = list()
        objects_index = 0
        for i in range(input.size(0)):
            box = objects[objects_index:objects_index + objects_length[i].
                item()]
            objects_index += objects_length[i].item()
            with torch.no_grad():
                batch_ind = i + torch.zeros(box.size(0), 1, dtype=box.dtype,
                    device=box.device)
                image_h, image_w = input.size(2
                    ) * self.downsample_rate, input.size(3
                    ) * self.downsample_rate
                image_box = torch.cat([torch.zeros(box.size(0), 1, dtype=
                    box.dtype, device=box.device), torch.zeros(box.size(0),
                    1, dtype=box.dtype, device=box.device), image_w + torch
                    .zeros(box.size(0), 1, dtype=box.dtype, device=box.
                    device), image_h + torch.zeros(box.size(0), 1, dtype=
                    box.dtype, device=box.device)], dim=-1)
                box_context_imap = functional.generate_intersection_map(box,
                    image_box, self.pool_size)
            this_context_features = self.context_roi_pool(context_features,
                torch.cat([batch_ind, image_box], dim=-1)[:1])
            this_object_features = torch.cat([self.object_roi_pool(
                object_features, torch.cat([batch_ind, box], dim=-1))], dim=1)
            outputs.append((this_object_features, this_context_features[0],
                box_context_imap))
        return outputs

    def _norm(self, x):
        return x / x.norm(2, dim=-1, keepdim=True)


class SceneGraphGroundtruth(nn.Module):

    def __init__(self, vocab, used_concepts):
        super().__init__()
        self.vocab = vocab
        self.used_concepts = used_concepts
        self.output_dims = [None, 0, 4]
        self.register_buffer('global2local', torch.zeros(len(self.vocab),
            dtype=torch.int64))
        for k, v in self.used_concepts.items():
            if v['type'] != 'attribute':
                continue
            self.output_dims[1] += len(v['values'])
            v = v['values']
            self.register_buffer('local2global_{}'.format(k), torch.zeros(
                len(v), dtype=torch.int64))
            for i, vv in enumerate(v):
                self.global2local[vocab.word2idx[vv]] = i
                getattr(self, 'local2global_{}'.format(k))[i] = vocab.word2idx[
                    vv]

    def forward(self, input, objects, objects_length, feed_dict):
        objects_index = 0
        relation_index = 0
        outputs = []
        for i in range(input.size(0)):
            nr_objects = objects_length[i].item()
            object_features = []
            for attribute, info in self.used_concepts.items():
                if info['type'] == 'attribute':
                    values = feed_dict['objects_' + attribute][objects_index
                        :objects_index + nr_objects]
                    mapped_values = self._valmap(self.global2local, values)
                    object_features.append(jactorch.one_hot(mapped_values,
                        len(info['values'])))
            object_features = torch.cat(object_features, dim=-1)
            object_features = object_features.float().to(input.device)
            relation_features = feed_dict['relations_spatial_relation'][
                relation_index:relation_index + nr_objects * nr_objects]
            relation_features = relation_features.float().to(input.device)
            outputs.append((None, object_features, relation_features.view(
                nr_objects, nr_objects, 4)))
            objects_index += nr_objects
            relation_index += nr_objects * nr_objects
        return outputs

    @staticmethod
    def _valmap(a, i):
        return a[i.view(-1)].view_as(i)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_vacancy_NSCL_PyTorch_Release(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(LearnedPositionalEmbedding(*[], **{'num_embeddings': 4, 'embedding_dim': 4}), [torch.zeros([4, 4], dtype=torch.int64)], {})

