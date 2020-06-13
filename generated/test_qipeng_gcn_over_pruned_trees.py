import sys
_module = sys.modules[__name__]
del sys
loader = _module
eval = _module
gcn = _module
trainer = _module
tree = _module
prepare_vocab = _module
train = _module
constant = _module
helper = _module
scorer = _module
torch_utils = _module
vocab = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import random


import torch.optim as optim


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        idx += [t.idx]
        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children
    if not directed:
        ret = ret + ret.T
    if self_loop:
        for i in idx:
            ret[i, i] = 1
    return ret


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, tokens, len_, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None
    if prune < 0:
        nodes = [Tree() for _ in head]
        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    else:
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]
        cas = None
        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h - 1]
                subj_ancestors.add(h - 1)
                h = head[h - 1]
            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)
        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = head[h - 1]
            cas.intersection_update(tmp)
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: (0) for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break
        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)
        dist = [(-1 if i not in path_nodes else 0) for i in range(len_)]
        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)
                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(10000.0)
        highest_node = lca
        nodes = [(Tree() if dist[i] <= prune else None) for i in range(len_)]
        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h - 1] is not None
                nodes[h - 1].add_child(nodes[i])
        root = nodes[highest_node]
    assert root is not None
    return root


class GCNRelationModel(nn.Module):

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'],
            padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']
            ) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']
            ) if opt['ner_dim'] > 0 else None
        embeddings = self.emb, self.pos_emb, self.ner_emb
        self.init_embeddings()
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.
                ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['topn'] <= 0:
            None
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            None
            self.emb.weight.register_hook(lambda x: torch_utils.
                keep_partial_grad(x, self.opt['topn']))
        else:
            None

    def forward(self, inputs):
        (words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type,
            obj_type) = inputs
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu(
                ).numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[
                i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=
                False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj) if self.opt['cuda'] else Variable(adj)
        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt[
            'prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask = self.gcn(adj, inputs)
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0
            ).eq(0).unsqueeze(2)
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True,
    use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = total_layers, batch_size, hidden_dim
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.emb, self.pos_emb, self.ner_emb = embeddings
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt[
                'rnn_layers'], batch_first=True, dropout=opt['rnn_dropout'],
                bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.
            opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens,
            batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs,
            batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        (words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type,
            obj_type) = inputs
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks,
                words.size()[0]))
        else:
            gcn_inputs = embs
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs, mask


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_qipeng_gcn_over_pruned_trees(_paritybench_base):
    pass
