import sys
_module = sys.modules[__name__]
del sys
src = _module
config = _module
data_utils = _module
dataloader = _module
dataset_base = _module
dataset_pretrain = _module
dataset_video_qa = _module
dataset_video_retrieval = _module
dataset_vqa = _module
decoder = _module
e2e_model = _module
grid_feat = _module
grid_feats = _module
build_loader = _module
dataset_mapper = _module
roi_heads = _module
visual_genome = _module
modeling = _module
transformers = _module
adamw = _module
sched = _module
utils = _module
file2lmdb = _module
lmdb_utils = _module
run_pretrain = _module
run_msrvtt_mc = _module
run_video_qa = _module
run_video_retrieval = _module
run_vqa = _module
basic_utils = _module
distributed = _module
load_save = _module
logger = _module
misc = _module

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


import torch


import random


import torchvision.transforms as transforms


from torchvision.transforms.functional import pad as img_pad


from torchvision.transforms.functional import resize as img_resize


from torch.nn.functional import interpolate as img_tensor_resize


from torch.nn.functional import pad as img_tensor_pad


from torch.nn.modules.utils import _quadruple


import numbers


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


import copy


import math


from torch import nn


import logging


import torch.utils.data


from torch.nn import functional as F


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.optim import Optimizer


from torch.optim import Adam


from torch.optim import Adamax


from torch.optim import SGD


import time


from torch.nn.utils import clip_grad_norm_


from collections import defaultdict


from torch.utils.data.distributed import DistributedSampler


from typing import Any


from typing import Dict


from typing import Union


logger = logging.getLogger(__name__)


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += pointer.shape, array.shape
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_random_sample_indices(seq_len, num_samples=100, device=torch.device('cpu')):
    """
    Args:
        seq_len: int, the sampled indices will be in the range [0, seq_len-1]
        num_samples: sample size
        device: torch.device

    Returns:
        1D torch.LongTensor consisting of sorted sample indices
        (sort should not affect the results as we use transformers)
    """
    if num_samples >= seq_len:
        sample_indices = np.arange(seq_len)
    else:
        sample_indices = np.random.choice(seq_len, size=num_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    return torch.from_numpy(sample_indices).long()


class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """

    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        self.col_position_embeddings = nn.Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
        bsz, _, _, _, hsz = grid.shape
        grid = grid.mean(1)
        grid = self.add_2d_positional_embeddings(grid)
        visual_tokens = grid.view(bsz, -1, hsz)
        if hasattr(self.config, 'pixel_random_sampling_size') and self.config.pixel_random_sampling_size > 0 and self.training:
            sampled_indices = get_random_sample_indices(seq_len=visual_tokens.shape[1], num_samples=self.config.pixel_random_sampling_size, device=visual_tokens.device)
            visual_tokens = visual_tokens.index_select(dim=1, index=sampled_indices)
        visual_tokens_shape = visual_tokens.shape[:-1]
        device = visual_tokens.device
        token_type_ids = torch.zeros(visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_temporal_postion_embeddings(self, grid):
        """
        Args:
            grid: (B, n_frms, H, W, d)

        Returns:
            (B, n_frms, H, W, d)
        """
        n_frms, height, width, hsz = grid.shape[-4:]
        temporal_position_ids = torch.arange(n_frms, dtype=torch.long, device=grid.device)
        t_position_embeddings = self.temporal_position_embeddings(temporal_position_ids)
        new_shape = 1, n_frms, 1, 1, hsz
        grid = grid + t_position_embeddings.view(*new_shape)
        return grid

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)

        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]
        row_position_ids = torch.arange(height, dtype=torch.long, device=grid.device)
        row_position_embeddings = self.row_position_embeddings(row_position_ids)
        row_shape = (1,) * (len(grid.shape) - 3) + (height, 1, hsz)
        grid = grid + row_position_embeddings.view(*row_shape)
        col_position_ids = torch.arange(width, dtype=torch.long, device=grid.device)
        col_position_embeddings = self.col_position_embeddings(col_position_ids)
        col_shape = (1,) * (len(grid.shape) - 3) + (1, width, hsz)
        grid = grid + col_position_embeddings.view(*col_shape)
        return grid


def add_attribute_config(cfg):
    """
    Add config for attribute prediction.
    """
    cfg.MODEL.ATTRIBUTE_ON = False
    cfg.INPUT.MAX_ATTR_PER_INS = 16
    cfg.MODEL.ROI_ATTRIBUTE_HEAD = CN()
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM = 256
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM = 512
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT = 0.2
    cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES = 400
    """
    Add config for box regression loss adjustment.
    """
    cfg.MODEL.RPN.BBOX_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.BBOX_LOSS_WEIGHT = 1.0


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, init_identity=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    if init_identity:
        """ init as identity kernel, not working yet
        0 0 0
        0 1 0
        0 0 0
        """
        identity_weight = conv.weight.new_zeros(3, 3)
        identity_weight[1, 1] = 1.0 / in_planes
        identity_weight = identity_weight.view(1, 1, 3, 3).expand(conv.weight.size())
        with torch.no_grad():
            conv.weight = nn.Parameter(identity_weight)
    return conv


class GridFeatBackbone(nn.Module):

    def __init__(self, detectron2_model_cfg, config, input_format='BGR'):
        super(GridFeatBackbone, self).__init__()
        self.detectron2_cfg = self.__setup__(detectron2_model_cfg)
        self.feature = build_model(self.detectron2_cfg)
        self.grid_encoder = nn.Sequential(conv3x3(config.backbone_channel_in_size, config.hidden_size), nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(inplace=True))
        self.input_format = input_format
        assert input_format == 'BGR', 'detectron 2 image input format should be BGR'
        self.config = config

    def __setup__(self, config_file):
        """
        Create configs and perform basic setups.
        """
        rank = hvd.rank()
        detectron2_cfg = get_cfg()
        add_attribute_config(detectron2_cfg)
        detectron2_cfg.merge_from_file(config_file)
        detectron2_cfg.MODEL.RESNETS.RES5_DILATION = 1
        detectron2_cfg.MODEL.DEVICE = f'cpu'
        detectron2_cfg.freeze()
        setup_logger(None, distributed_rank=rank, name='fvcore')
        logger = setup_logger(None, distributed_rank=rank)
        return detectron2_cfg

    def load_state_dict(self, model_path):
        if not os.path.exists(model_path):
            None
            DetectionCheckpointer(self.feature).resume_or_load(self.detectron2_cfg.MODEL.WEIGHTS, resume=True)
        else:
            DetectionCheckpointer(self.feature).resume_or_load(model_path, resume=True)

    @property
    def config_file(self):
        return self.detectron2_cfg.dump()

    def train(self, mode=True):
        super(GridFeatBackbone, self).train(mode)

    def forward(self, x):
        bsz, n_frms, c, h, w = x.shape
        x = x.view(bsz * n_frms, c, h, w)
        if self.input_format == 'BGR':
            x = x[:, [2, 1, 0], :, :]
        res5_features = self.feature.backbone(x)
        grid_feat_outputs = self.feature.roi_heads.get_conv5_features(res5_features)
        grid = self.grid_encoder(grid_feat_outputs)
        new_c, new_h, new_w = grid.shape[-3:]
        grid = grid.view(bsz, n_frms, new_c, new_h, new_w)
        grid = grid.permute(0, 1, 3, 4, 2)
        return grid


LOGGER = logging.getLogger('__main__')


def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""
    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(loaded_state_dict_or_path, map_location='cpu')
    else:
        loaded_state_dict = loaded_state_dict_or_path
    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())
    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]
    LOGGER.info('You can ignore the keys with `num_batches_tracked` or from task heads')
    LOGGER.info('Keys in loaded but not in model:')
    diff_keys = load_keys.difference(model_keys)
    LOGGER.info(f'In total {len(diff_keys)}, {sorted(diff_keys)}')
    LOGGER.info('Keys in model but not in loaded:')
    diff_keys = model_keys.difference(load_keys)
    LOGGER.info(f'In total {len(diff_keys)}, {sorted(diff_keys)}')
    LOGGER.info('Keys in model and loaded, but shape mismatched:')
    LOGGER.info(f'In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}')
    model.load_state_dict(toload, strict=False)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def repeat_tensor_rows(raw_tensor, row_repeats):
    """ repeat raw_tensor[i] row_repeats[i] times.
    Args:
        raw_tensor: (B, *)
        row_repeats: list(int), len(row_repeats) == len(raw_tensor)
    """
    assert len(raw_tensor) == len(raw_tensor), 'Has to be the same length'
    if sum(row_repeats) == len(row_repeats):
        return raw_tensor
    else:
        indices = torch.LongTensor(flat_list_of_lists([([i] * r) for i, r in enumerate(row_repeats)]))
        return raw_tensor.index_select(0, indices)


class AttributePredictor(nn.Module):
    """
    Head for attribute prediction, including feature/score computation and
    loss computation.

    """

    def __init__(self, cfg, input_dim):
        super().__init__()
        self.num_objs = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.obj_embed_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM
        self.fc_dim = cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM
        self.num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES
        self.max_attr_per_ins = cfg.INPUT.MAX_ATTR_PER_INS
        self.loss_weight = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT
        self.obj_embed = nn.Embedding(self.num_objs + 1, self.obj_embed_dim)
        input_dim += self.obj_embed_dim
        self.fc = nn.Sequential(nn.Linear(input_dim, self.fc_dim), nn.ReLU())
        self.attr_score = nn.Linear(self.fc_dim, self.num_attributes)
        nn.init.normal_(self.attr_score.weight, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, obj_labels):
        attr_feat = torch.cat((x, self.obj_embed(obj_labels)), dim=1)
        return self.attr_score(self.fc(attr_feat))

    def loss(self, score, label):
        n = score.shape[0]
        score = score.unsqueeze(1)
        score = score.expand(n, self.max_attr_per_ins, self.num_attributes).contiguous()
        score = score.view(-1, self.num_attributes)
        inv_weights = (label >= 0).sum(dim=1).repeat(self.max_attr_per_ins, 1).transpose(0, 1).flatten()
        weights = inv_weights.float().reciprocal()
        weights[weights > 1] = 0.0
        n_valid = len((label >= 0).sum(dim=1).nonzero())
        label = label.view(-1)
        attr_loss = F.cross_entropy(score, label, reduction='none', ignore_index=-1)
        attr_loss = (attr_loss * weights).view(n, -1).sum(dim=1)
        if n_valid > 0:
            attr_loss = attr_loss.sum() * self.loss_weight / n_valid
        else:
            attr_loss = attr_loss.sum() * 0.0
        return {'loss_attr': attr_loss}


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, output_size))

    def forward(self, hidden_states):
        return self.classifier(hidden_states)


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOnlyNSPHead,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, output_attentions=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jayleicn_ClipBERT(_paritybench_base):
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

