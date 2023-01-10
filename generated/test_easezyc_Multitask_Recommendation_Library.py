import sys
_module = sys.modules[__name__]
del sys
preprocess = _module
aliexpress = _module
main = _module
aitm = _module
layers = _module
metaheac = _module
mmoe = _module
omoe = _module
ple = _module
sharedbottom = _module
singletask = _module
test = _module

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


import pandas as pd


import torch


from sklearn.metrics import roc_auc_score


from torch.utils.data import DataLoader


class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class AITMModel(torch.nn.Module):
    """
    A pytorch implementation of Adaptive Information Transfer Multi-task Model.

    Reference:
        Xi, Dongbo, et al. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising. KDD 2021.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]
        self.g = torch.nn.ModuleList([torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1]) for i in range(task_num - 1)])
        self.h1 = torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h2 = torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h3 = torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.bottom = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        fea = [self.bottom[i](emb) for i in range(self.task_num)]
        for i in range(1, self.task_num):
            p = self.g[i - 1](fea[i - 1]).unsqueeze(1)
            q = fea[i].unsqueeze(1)
            x = torch.cat([p, q], dim=1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            fea[i] = torch.sum(torch.nn.functional.softmax(torch.sum(K * Q, 2, True) / np.sqrt(self.hidden_dim), dim=1) * V, 1)
        results = [torch.sigmoid(self.tower[i](fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results


class Meta_Linear(torch.nn.Linear):

    def __init__(self, in_features, out_features):
        super(Meta_Linear, self).__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = torch.nn.functional.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Meta_Linear, self).forward(x)
        return out


class Meta_Embedding(torch.nn.Embedding):

    def __init__(self, num_embedding, embedding_dim):
        super(Meta_Embedding, self).__init__(num_embedding, embedding_dim)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = torch.nn.functional.embedding(x, self.weight.fast, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            out = torch.nn.functional.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out


class HeacModel(torch.nn.Module):
    """
    A pytorch implementation of Hybrid Expert and Critic Model.

    Reference:
        Zhu, Yongchun, et al. Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising. KDD 2021.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, critic_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_embedding = Meta_Embedding(task_num, embed_dim)
        self.task_num = task_num
        self.expert_num = expert_num
        self.critic_num = critic_num
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.critic = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(critic_num)])
        self.expert_gate = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, expert_num), torch.nn.Softmax(dim=1))
        self.critic_gate = torch.nn.Sequential(torch.nn.Linear(embed_dim * 2, critic_num), torch.nn.Softmax(dim=1))

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1)
        batch_size = emb.size(0)
        gate_input_emb = []
        for i in range(self.task_num):
            idxs = torch.tensor([i for j in range(batch_size)]).view(-1, 1)
            task_emb = self.task_embedding(idxs).squeeze(1)
            gate_input_emb.append(torch.cat([task_emb, torch.mean(emb, dim=1)], dim=1).view(batch_size, -1))
        emb = emb.view(-1, self.embed_output_dim)
        expert_gate_value = [self.expert_gate(gate_input_emb[i]).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1)
        task_fea = [torch.bmm(expert_gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        critic_gate_value = [self.critic_gate(gate_input_emb[i]) for i in range(self.task_num)]
        results = []
        for i in range(self.task_num):
            output = [torch.sigmoid(self.critic[j](task_fea[i])) for j in range(self.critic_num)]
            output = torch.cat(output, dim=1)
            results.append(torch.mean(critic_gate_value[i] * output, dim=1))
        return results


class MetaHeacModel(torch.nn.Module):

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, critic_num, dropout):
        super(MetaHeacModel, self).__init__()
        self.model = HeacModel(categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, critic_num, dropout)
        self.local_lr = 0.0002
        self.criterion = torch.nn.BCELoss()

    def forward(self, categorical_x, numerical_x):
        return self.model(categorical_x, numerical_x)

    def local_update(self, support_set_categorical, support_set_numerical, support_set_y):
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        support_set_y_pred = self.model(support_set_categorical, support_set_numerical)
        loss_list = [self.criterion(support_set_y_pred[j], support_set_y[:, j].float()) for j in range(support_set_y.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        self.model.zero_grad()
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
        fast_parameters = []
        for k, weight in enumerate(self.model.parameters()):
            if grad[k] is None:
                continue
            if weight.fast is None:
                weight.fast = weight - self.local_lr * grad[k]
            else:
                weight.fast = weight.fast - self.local_lr * grad[k]
            fast_parameters.append(weight.fast)
        return loss

    def global_update(self, list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y):
        batch_sz = len(list_sup_categorical)
        losses_q = []
        for i in range(batch_sz):
            loss_sup = self.local_update(list_sup_categorical[i], list_sup_numerical[i], list_sup_y[i])
            query_set_y_pred = self.model(list_qry_categorical[i], list_qry_numerical[i])
            loss_list = [self.criterion(query_set_y_pred[j], list_qry_y[i][:, j].float()) for j in range(list_qry_y[i].size(1))]
            loss = 0
            for item in loss_list:
                loss += item
            loss /= len(loss_list)
            losses_q.append(loss)
        losses_q = torch.stack(losses_q).mean(0)
        return losses_q


class MMoEModel(torch.nn.Module):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results


class OMoEModel(torch.nn.Module):
    """
    A pytorch implementation of one-gate MoE Model.

    Reference:
        Jacobs, Robert A., et al. "Adaptive mixtures of local experts." Neural computation 3.1 (1991): 79-87.
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, expert_num), torch.nn.Softmax(dim=1))

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = self.gate(emb).unsqueeze(1)
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1)
        fea = torch.bmm(gate_value, fea).squeeze(1)
        results = [torch.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        return results


class PLEModel(torch.nn.Module):
    """
    A pytorch implementation of PLE Model.

    Reference:
        Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, shared_expert_num, specific_expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)
        self.task_experts = [([0] * self.task_num) for _ in range(self.layers_num)]
        self.task_gates = [([0] * self.task_num) for _ in range(self.layers_num)]
        self.share_experts = [0] * self.layers_num
        self.share_gates = [0] * self.layers_num
        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i - 1]
            self.share_experts[i] = torch.nn.ModuleList([MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for k in range(self.shared_expert_num)])
            self.share_gates[i] = torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + task_num * specific_expert_num), torch.nn.Softmax(dim=1))
            for j in range(task_num):
                self.task_experts[i][j] = torch.nn.ModuleList([MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False) for k in range(self.specific_expert_num)])
                self.task_gates[i][j] = torch.nn.Sequential(torch.nn.Linear(input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1))
            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])
        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        task_fea = [emb for i in range(self.task_num + 1)]
        for i in range(self.layers_num):
            share_output = [expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]
            task_output_list = []
            for j in range(self.task_num):
                task_output = [expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_ouput).squeeze(1)
            if i != self.layers_num - 1:
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                task_fea[-1] = torch.bmm(gate_value, mix_ouput).squeeze(1)
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results


class SharedBottomModel(torch.nn.Module):
    """
    A pytorch implementation of Shared-Bottom Model.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.bottom = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        fea = self.bottom(emb)
        results = [torch.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        return results


class SingleTaskModel(torch.nn.Module):
    """
    A pytorch implementation of Single Task Model.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        self.embedding = torch.nn.ModuleList([EmbeddingLayer(categorical_field_dims, embed_dim) for i in range(task_num)])
        self.numerical_layer = torch.nn.ModuleList([torch.nn.Linear(numerical_num, embed_dim) for i in range(task_num)])
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.bottom = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        results = list()
        for i in range(self.task_num):
            categorical_emb = self.embedding[i](categorical_x)
            numerical_emb = self.numerical_layer[i](numerical_x).unsqueeze(1)
            emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
            fea = self.bottom[i](emb)
            results.append(torch.sigmoid(self.tower[i](fea).squeeze(1)))
        return results


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Meta_Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiLayerPerceptron,
     lambda: ([], {'input_dim': 4, 'embed_dims': [4, 4], 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_easezyc_Multitask_Recommendation_Library(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

