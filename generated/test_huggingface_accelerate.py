import sys
_module = sys.modules[__name__]
del sys
big_model_inference = _module
measures_util = _module
automatic_gradient_accumulation = _module
checkpointing = _module
cross_validation = _module
deepspeed_with_config_support = _module
fsdp_with_peak_mem_tracking = _module
gradient_accumulation = _module
megatron_lm_gpt_pretraining = _module
memory = _module
multi_process_metrics = _module
tracking = _module
complete_cv_example = _module
complete_nlp_example = _module
cv_example = _module
nlp_example = _module
stage_1 = _module
stage_2 = _module
stage_3 = _module
stage_4 = _module
stage_5 = _module
setup = _module
accelerate = _module
accelerator = _module
big_modeling = _module
checkpointing = _module
commands = _module
accelerate_cli = _module
config = _module
cluster = _module
config_args = _module
config_utils = _module
default = _module
sagemaker = _module
update = _module
env = _module
launch = _module
menu = _module
cursor = _module
helpers = _module
input = _module
keymap = _module
selection_menu = _module
test = _module
tpu = _module
data_loader = _module
hooks = _module
launchers = _module
logging = _module
memory_utils = _module
optimizer = _module
scheduler = _module
state = _module
test_utils = _module
examples = _module
scripts = _module
external_deps = _module
test_checkpointing = _module
test_metrics = _module
test_peak_memory_usage = _module
test_performance = _module
test_cli = _module
test_distributed_data_loop = _module
test_script = _module
test_sync = _module
testing = _module
training = _module
tracking = _module
utils = _module
constants = _module
dataclasses = _module
deepspeed = _module
environment = _module
imports = _module
launch = _module
megatron_lm = _module
memory = _module
modeling = _module
offload = _module
operations = _module
other = _module
random = _module
rich = _module
torch_xla = _module
tqdm = _module
versions = _module
test_deepspeed = _module
test_fsdp = _module
test_accelerator = _module
test_big_modeling = _module
test_cli = _module
test_cpu = _module
test_data_loader = _module
test_examples = _module
test_grad_sync = _module
test_hooks = _module
test_kwargs_handlers = _module
test_memory_utils = _module
test_metrics = _module
test_modeling_utils = _module
test_multigpu = _module
test_offload = _module
test_optimizer = _module
test_sagemaker = _module
test_scheduler = _module
test_state_checkpointing = _module
test_tpu = _module
test_tracking = _module
test_utils = _module
xla_spawn = _module
log_reports = _module
stale = _module
style_doc = _module

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


import time


import torch


from torch.optim import AdamW


from torch.utils.data import DataLoader


from typing import List


import numpy as np


from sklearn.model_selection import StratifiedKFold


import logging


import math


import random


from itertools import chain


import re


from torch.optim.lr_scheduler import OneCycleLR


from torch.utils.data import Dataset


from torchvision.transforms import Compose


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


import warnings


from functools import wraps


from typing import Optional


from typing import Union


from typing import Dict


import torch.nn as nn


from torch.cuda.amp import GradScaler


from torch.utils.data import BatchSampler


from torch.utils.data import IterableDataset


import functools


from typing import Mapping


import inspect


from copy import deepcopy


from torch.utils.data import TensorDataset


import torch.nn.functional as F


from torch.optim.lr_scheduler import LambdaLR


from functools import partial


from abc import ABCMeta


from abc import abstractmethod


from abc import abstractproperty


from typing import Any


import copy


import enum


import typing


from typing import Callable


from typing import Iterable


from functools import lru_cache


from abc import ABC


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


from collections import defaultdict


from typing import Tuple


from collections.abc import Mapping


from functools import update_wrapper


from torch.distributed import ReduceOp


import itertools


from torch import nn


from collections import UserDict


from collections import namedtuple


class RegressionModel(torch.nn.Module):

    def __init__(self, a=0, b=0, double_output=False):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(a).float())
        self.b = torch.nn.Parameter(torch.tensor(b).float())
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            None
            self.first_batch = False
        return x * self.a + self.b


class AbstractTrainStep(ABC):
    """Abstract class for batching, forward pass and loss handler."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_batch_func(self):
        pass

    def get_forward_step_func(self):
        pass

    def get_loss_func(self):
        pass


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        return type(obj)(*list(generator))


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def recursively_apply(func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs:
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(data, (recursively_apply(func, o, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs) for o in data))
    elif isinstance(data, Mapping):
        return type(data)({k: recursively_apply(func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs) for k, v in data.items()})
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects that satisfy {test_type.__name__}.")
    return data


def send_to_device(tensor, device, non_blocking=False):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """

    def _send_to_device(t, device, non_blocking):
        try:
            return t
        except TypeError:
            return t

    def _has_to_method(t):
        return hasattr(t, 'to')
    return recursively_apply(_send_to_device, tensor, device, non_blocking, test_type=_has_to_method)


class BertTrainStep(AbstractTrainStep):
    """
    Bert train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__('BertTrainStep')
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func(args.pretraining_flag, args.num_labels)
        self.forward_step = self.get_forward_step_func(args.pretraining_flag, args.bert_binary_head)
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = SequenceClassifierOutput

    def get_batch_func(self, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Build the batch."""
            keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)
            tokens = data_b['text'].long()
            types = data_b['types'].long()
            sentence_order = data_b['is_random'].long()
            loss_mask = data_b['loss_mask'].float()
            lm_labels = data_b['labels'].long()
            padding_mask = data_b['padding_mask'].long()
            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())
            tokens = data['input_ids'].long()
            padding_mask = data['attention_mask'].long()
            if 'token_type_ids' in data:
                types = data['token_type_ids'].long()
            else:
                types = None
            if 'labels' in data:
                lm_labels = data['labels'].long()
                loss_mask = data['labels'] != -100
            else:
                lm_labels = None
                loss_mask = None
            if 'next_sentence_label' in data:
                sentence_order = data['next_sentence_label'].long()
            else:
                sentence_order = None
            return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self, pretraining_flag, num_labels):

        def loss_func_pretrain(loss_mask, sentence_order, output_tensor):
            lm_loss_, sop_logits = output_tensor
            lm_loss_ = lm_loss_.float()
            loss_mask = loss_mask.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            if sop_logits is not None:
                sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1)
                sop_loss = sop_loss.float()
                loss = lm_loss + sop_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss, sop_loss])
                return loss, {'lm loss': averaged_losses[0], 'sop loss': averaged_losses[1]}
            else:
                loss = lm_loss
                averaged_losses = average_losses_across_data_parallel_group([lm_loss])
                return loss, {'lm loss': averaged_losses[0]}

        def loss_func_finetune(labels, logits):
            if num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            averaged_losses = average_losses_across_data_parallel_group([loss])
            return loss, {'loss': averaged_losses[0]}
        if pretraining_flag:
            return loss_func_pretrain
        else:
            return loss_func_finetune

    def get_forward_step_func(self, pretraining_flag, bert_binary_head):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, types, sentence_order, loss_mask, labels, padding_mask = self.get_batch(data_iterator)
            if not bert_binary_head:
                types = None
            if pretraining_flag:
                output_tensor = model(tokens, padding_mask, tokentype_ids=types, lm_labels=labels)
                return output_tensor, partial(self.loss_func, loss_mask, sentence_order)
            else:
                logits = model(tokens, padding_mask, tokentype_ids=types)
                return logits, partial(self.loss_func, labels)
        return forward_step


class GPTTrainStep(AbstractTrainStep):
    """
    GPT train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__('GPTTrainStep')
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        self.eod_token = args.padded_vocab_size - 1
        if args.vocab_file is not None:
            tokenizer = get_tokenizer()
            self.eod_token = tokenizer.eod
        self.reset_position_ids = args.reset_position_ids
        self.reset_attention_mask = args.reset_attention_mask
        self.eod_mask_loss = args.eod_mask_loss
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Generate a batch"""
            keys = ['text']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)
            tokens_ = data_b['text'].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss)
            return tokens, labels, loss_mask, attention_mask, position_ids

        def get_batch_transformer(data_iterator):
            data = next(data_iterator)
            data = {'input_ids': data['input_ids']}
            data = send_to_device(data, torch.cuda.current_device())
            tokens_ = data['input_ids'].long()
            padding = torch.zeros((tokens_.shape[0], 1), dtype=tokens_.dtype, device=tokens_.device) + self.eod_token
            tokens_ = torch.concat([tokens_, padding], dim=1)
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, True)
            return tokens, labels, loss_mask, attention_mask, position_ids
        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):
        args = get_args()

        def loss_func(loss_mask, output_tensor):
            if args.return_logits:
                losses, logits = output_tensor
            else:
                losses = output_tensor
            losses = losses.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            averaged_loss = average_losses_across_data_parallel_group([loss])
            output_dict = {'lm loss': averaged_loss[0]}
            if args.return_logits:
                output_dict.update({'logits': logits})
            return loss, output_dict
        return loss_func

    def get_forward_step_func(self):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            return output_tensor, partial(self.loss_func, loss_mask)
        return forward_step


class T5TrainStep(AbstractTrainStep):
    """
    T5 train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__('T5TrainStep')
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = Seq2SeqLMOutput

    @staticmethod
    def attn_mask_postprocess(attention_mask):
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = attention_mask.unsqueeze(2)
        attention_mask_bss = attention_mask_b1s * attention_mask_bs1
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    @staticmethod
    def get_decoder_mask(seq_length, device):
        attention_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device))
        attention_mask = attention_mask < 0.5
        return attention_mask

    @staticmethod
    def get_enc_dec_mask(attention_mask, dec_seq_length, device):
        batch_size, _ = attention_mask.shape
        attention_mask_b1s = attention_mask.unsqueeze(1)
        attention_mask_bs1 = torch.ones((batch_size, dec_seq_length, 1), device=device)
        attention_mask_bss = attention_mask_bs1 * attention_mask_b1s
        extended_attention_mask = attention_mask_bss < 0.5
        return extended_attention_mask

    def get_batch_func(self, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Build the batch."""
            keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)
            tokens_enc = data_b['text_enc'].long()
            tokens_dec = data_b['text_dec'].long()
            labels = data_b['labels'].long()
            loss_mask = data_b['loss_mask'].float()
            enc_mask = data_b['enc_mask'] < 0.5
            dec_mask = data_b['dec_mask'] < 0.5
            enc_dec_mask = data_b['enc_dec_mask'] < 0.5
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask

        def get_batch_transformer(data_iterator):
            """Build the batch."""
            data = next(data_iterator)
            data = send_to_device(data, torch.cuda.current_device())
            tokens_enc = data['input_ids'].long()
            labels = data['labels'].long()
            loss_mask = labels != -100
            if 'decoder_input_ids' in data:
                tokens_dec = data['decoder_input_ids'].long()
            else:
                tokens_dec = labels.new_zeros(labels.shape, device=labels.device, dtype=torch.long)
                tokens_dec[..., 1:] = labels[..., :-1].clone()
                tokens_dec[..., 0] = 0
                tokens_dec.masked_fill_(tokens_dec == -100, 0)
            enc_mask = T5TrainStep.attn_mask_postprocess(data['attention_mask'].long())
            dec_mask = T5TrainStep.get_decoder_mask(tokens_dec.shape[1], tokens_dec.device)
            enc_dec_mask = T5TrainStep.get_enc_dec_mask(data['attention_mask'].long(), tokens_dec.shape[1], tokens_dec.device)
            return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask
        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):

        def loss_func(loss_mask, output_tensor):
            lm_loss_ = output_tensor.float()
            lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
            loss = lm_loss
            averaged_losses = average_losses_across_data_parallel_group([lm_loss])
            return loss, {'lm loss': averaged_losses[0]}
        return loss_func

    def get_forward_step_func(self):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens_enc, tokens_dec, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask = self.get_batch(data_iterator)
            output_tensor = model(tokens_enc, tokens_dec, enc_mask, dec_mask, enc_dec_mask, tokentype_ids=None, lm_labels=lm_labels)
            return output_tensor, partial(self.loss_func, loss_mask)
        return forward_step


class MegatronEngine(torch.nn.Module):
    """
    Megatron-LM model wrapper

    Args:
        accelerator (:class:`~accelerate.Accelerator`): The accelerator object to use.
        model: Megatron-LM model
        optimizer: Megatron-LM optimizer
        lr_scheduler: Megatron-LM lr scheduler
    """

    def __init__(self, accelerator, model, optimizer, scheduler):
        super(MegatronEngine, self).__init__()
        self.module = model
        self.base_model = model[0]
        self.optimizer = optimizer
        self.scheduler = scheduler
        args = get_args()
        if accelerator.state.megatron_lm_plugin.custom_train_step_class is not None:
            self.train_step_handler = accelerator.state.megatron_lm_plugin.custom_train_step_class(args, **accelerator.state.megatron_lm_plugin.custom_train_step_kwargs)
        elif args.model_type_name == 'bert':
            self.train_step_handler = BertTrainStep(args)
        elif args.model_type_name == 'gpt':
            self.train_step_handler = GPTTrainStep(args)
        elif args.model_type_name == 't5':
            self.train_step_handler = T5TrainStep(args)
        else:
            raise ValueError(f'Unsupported model type: {args.model_type_name}')
        self.optimizer.skipped_iter = False
        self.total_loss_dict = {}
        self.eval_total_loss_dict = {}
        self.iteration = 0
        self.report_memory_flag = True
        if args.tensorboard_dir is not None:
            write_args_to_tensorboard()

    def train(self):
        for model_module in self.module:
            model_module.train()
        self.log_eval_results()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

    def train_step(self, **batch_data):
        """
        Training step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to train on.
        """
        args = get_args()
        timers = get_timers()
        if len(batch_data) > 0:
            data_chunks = []
            if args.num_micro_batches > 1:
                for i in range(0, args.num_micro_batches):
                    data_chunks.append({k: v[i * args.micro_batch_size:(i + 1) * args.micro_batch_size] for k, v in batch_data.items()})
            else:
                data_chunks = [batch_data]
        if len(self.module) > 1:
            batch_data_iterator = [iter(data_chunks) for _ in range(len(self.module))] if len(batch_data) > 0 else [None] * len(self.module)
        else:
            batch_data_iterator = iter(data_chunks) if len(batch_data) > 0 else None
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            for partition in self.module:
                partition.zero_grad_buffer()
        self.optimizer.zero_grad()
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(self.train_step_handler.forward_step, batch_data_iterator, self.module, self.optimizer, None, forward_only=False)
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()
        timers('backward-reduce-model-grads').start()
        self.optimizer.reduce_model_grads(args, timers)
        timers('backward-reduce-model-grads').stop()
        timers('optimizer').start()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step(args, timers)
        timers('optimizer').stop()
        if update_successful:
            timers('backward-gather-model-params').start()
            self.optimizer.gather_model_params(args, timers)
            timers('backward-gather-model-params').stop()
        if update_successful:
            if self.scheduler is not None:
                increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                self.scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1
        self.optimizer.skipped_iter = not update_successful
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()
        args.consumed_train_samples += mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def eval_step(self, **batch_data):
        """
        Evaluation step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to evaluate on.
        """
        args = get_args()
        data_chunks = []
        if args.num_micro_batches > 1:
            for i in range(0, args.num_micro_batches):
                data_chunks.append({k: v[i * args.micro_batch_size:(i + 1) * args.micro_batch_size] for k, v in batch_data.items()})
        else:
            data_chunks = [batch_data]
        if len(self.module) > 1:
            batch_data_iterator = [iter(data_chunks) for _ in range(len(self.module))]
        else:
            batch_data_iterator = iter(data_chunks)
        forward_backward_func = get_forward_backward_func()
        loss_dicts = forward_backward_func(self.train_step_handler.forward_step, batch_data_iterator, self.module, optimizer=None, timers=None, forward_only=True)
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()
        args.consumed_valid_samples += mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = {}
            for key in loss_dicts[0]:
                losses_reduced_for_key = [x[key] for x in loss_dicts]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced
        else:
            return {}

    def forward(self, **batch_data):
        args = get_args()
        if self.module[0].training:
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(**batch_data)
            self.iteration += 1
            if args.tensorboard_dir is not None:
                loss_scale = self.optimizer.get_loss_scale().item()
                params_norm = None
                if args.log_params_norm:
                    params_norm = calc_params_l2_norm(self.model)
                self.report_memory_flag = training_log(loss_dict, self.total_loss_dict, self.optimizer.param_groups[0]['lr'], self.iteration, loss_scale, self.report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad)
        else:
            loss_dict = self.eval_step(**batch_data)
            if args.tensorboard_dir is not None:
                for key in loss_dict:
                    self.eval_total_loss_dict[key] = self.eval_total_loss_dict.get(key, torch.FloatTensor([0.0])) + loss_dict[key]
                    self.eval_total_loss_dict[key + '_num_iters'] = self.eval_total_loss_dict.get(key + '_num_iters', torch.FloatTensor([0.0])) + torch.FloatTensor([1.0])
        loss = torch.tensor(0.0, device=args.local_rank)
        for key in loss_dict:
            if len(loss_dict[key].shape) == 0:
                loss += loss_dict[key]
        logits = None
        if 'logits' in loss_dict:
            logits = loss_dict['logits']
        if self.train_step_handler.model_output_class is not None:
            return self.train_step_handler.model_output_class(loss=loss, logits=logits)
        return loss

    def log_eval_results(self):
        args = get_args()
        if args.tensorboard_dir is None or self.iteration == 0:
            return
        args = get_args()
        writer = get_tensorboard_writer()
        string = f'validation loss at iteration {self.iteration} | '
        for key in self.eval_total_loss_dict:
            if key.endswith('_num_iters'):
                continue
            value = self.eval_total_loss_dict[key] / self.eval_total_loss_dict[key + '_num_iters']
            string += f'{key} value: {value} | '
            ppl = math.exp(min(20, value.item()))
            if args.pretraining_flag:
                string += f'{key} PPL: {ppl} | '
            if writer:
                writer.add_scalar(f'{key} validation', value.item(), self.iteration)
                if args.pretraining_flag:
                    writer.add_scalar(f'{key} validation ppl', ppl, self.iteration)
        length = len(string) + 1
        print_rank_last('-' * length)
        print_rank_last(string)
        print_rank_last('-' * length)
        self.eval_total_loss_dict = {}

    def save_checkpoint(self, output_dir):
        self.log_eval_results()
        args = get_args()
        args.save = output_dir
        torch.distributed.barrier()
        save_checkpoint(self.iteration, self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()

    def load_checkpoint(self, input_dir):
        args = get_args()
        args.load = input_dir
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        torch.distributed.barrier()
        iteration = load_checkpoint(self.module, self.optimizer, self.scheduler)
        torch.distributed.barrier()
        self.iteration = iteration
        if args.fp16 and self.iteration == 0:
            self.optimizer.reload_model_params()

    def megatron_generate(self, inputs, attention_mask=None, max_length=None, max_new_tokens=None, num_beams=None, temperature=None, top_k=None, top_p=None, length_penalty=None, **kwargs):
        """
        Generate method for GPT2 model. This method is used for inference. Supports both greedy and beam search along
        with sampling. Refer the Megatron-LM repo for more details

        Args:
            inputs (torch.Tensor): input ids
            attention_mask (torch.Tensor, optional): attention mask. Defaults to None.
            max_length (int, optional): max length of the generated sequence. Defaults to None.
            Either this or max_new_tokens should be provided.
            max_new_tokens (int, optional): max number of tokens to be generated. Defaults to None.
            Either this or max_length should be provided.
            num_beams (int, optional): number of beams to use for beam search. Defaults to None.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            top_k (int, optional): top k tokens to consider for sampling. Defaults to 0.0.
            top_p (float, optional): tokens in top p probability are considered for sampling. Defaults to 0.0.
            length_penalty (float, optional): length penalty for beam search. Defaults to None.
            kwargs: additional key-value arguments
        """
        args = get_args()
        if args.model_type_name != 'gpt':
            raise NotImplementedError('Generate method is not implemented for this model')
        if args.data_parallel_size > 1:
            raise ValueError('Generate method requires data parallelism to be 1')
        if args.sequence_parallel:
            raise ValueError('Generate method requires sequence parallelism to be False')
        if args.recompute_granularity is not None:
            raise ValueError('Checkpoint activations cannot be set for inference')
        if args.vocab_file is None:
            raise ValueError('Vocab file is required for inference')
        if max_length is None and max_new_tokens is None:
            raise ValueError('`max_length` or `max_new_tokens` are required for inference')
        if temperature is None:
            temperature = 1.0
        elif not 0.0 < temperature <= 100.0:
            raise ValueError('temperature must be a positive number less than or equal to 100.0')
        if top_k is None:
            top_k = 0
        elif not 0 <= top_k <= 1000:
            raise ValueError('top_k must be a positive number less than or equal to 1000')
        if top_p is None:
            top_p = 0.0
        elif top_p > 0.0 and top_k > 0.0:
            raise ValueError('top_p and top_k sampling cannot be set together')
        elif not 0.0 <= top_p <= 1.0:
            raise ValueError('top_p must be less than or equal to 1.0')
        top_p_decay = kwargs.get('top_p_decay', 0.0)
        if not 0.0 <= top_p_decay <= 1.0:
            raise ValueError('top_p_decay must be less than or equal to 1.0')
        top_p_bound = kwargs.get('top_p_bound', 0.0)
        if not 0.0 <= top_p_bound <= 1.0:
            raise ValueError('top_p_bound must be less than or equal to 1.0')
        add_BOS = kwargs.get('add_BOS', False)
        if not isinstance(add_BOS, bool):
            raise ValueError('add_BOS must be a boolean')
        beam_width = num_beams
        if beam_width is not None:
            if not isinstance(beam_width, int):
                raise ValueError('beam_width must be an integer')
            if beam_width < 1:
                raise ValueError('beam_width must be greater than 0')
            if inputs.shape[0] > 1:
                return 'When doing beam_search, batch size must be 1'
        tokenizer = get_tokenizer()
        stop_token = kwargs.get('stop_token', tokenizer.eod)
        if stop_token is not None:
            if not isinstance(stop_token, int):
                raise ValueError('stop_token must be an integer')
        if length_penalty is None:
            length_penalty = 1.0
        sizes_list = None
        prompts_tokens_tensor = None
        prompts_length_tensor = None
        if torch.distributed.get_rank() == 0:
            if attention_mask is None:
                prompts_length_tensor = torch.LongTensor([inputs.shape[1]] * inputs.shape[0])
            else:
                prompts_length_tensor = attention_mask.sum(axis=-1)
            if max_new_tokens is None:
                max_new_tokens = max_length - inputs.shape[1]
            if max_new_tokens <= 0:
                raise ValueError('max_new_tokens must be greater than 0')
            if add_BOS:
                max_length = max_new_tokens + inputs.shape[1] + 1
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - (inputs.shape[1] + 1)
                padding = torch.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([torch.unsqueeze(padding[:, 0], axis=-1), inputs, padding], axis=-1)
            else:
                max_length = max_new_tokens + inputs.shape[1]
                max_length = 4 * math.ceil(max_length / 4)
                max_new_tokens = max_length - inputs.shape[1]
                padding = torch.LongTensor([[tokenizer.eod] * max_new_tokens] * inputs.shape[0])
                prompts_tokens_tensor = torch.concat([inputs, padding], axis=-1)
            sizes_list = [prompts_tokens_tensor.size(0), prompts_tokens_tensor.size(1)]
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=0)
        sizes = sizes_tensor.tolist()
        context_tokens_tensor = broadcast_tensor(sizes, torch.int64, tensor=prompts_tokens_tensor, rank=0)
        context_length_tensor = broadcast_tensor(sizes[0], torch.int64, tensor=prompts_length_tensor, rank=0)
        random_seed = kwargs.get('random_seed', 0)
        torch.random.manual_seed(random_seed)
        unwrapped_model = unwrap_model(self.base_model, (torchDDP, LocalDDP, Float16Module))
        if beam_width is not None:
            tokens, _ = beam_search_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, beam_width, stop_token=stop_token, num_return_gen=1, length_penalty=length_penalty)
        else:
            tokens, _, _ = generate_tokens_probs_and_return_on_first_stage(unwrapped_model, context_tokens_tensor, context_length_tensor, return_output_log_probs=False, top_k=top_k, top_p=top_p, top_p_decay=top_p_decay, top_p_bound=top_p_bound, temperature=temperature, use_eod_token_for_early_termination=True)
        return tokens


class ModelForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.batchnorm = nn.BatchNorm1d(4)
        self.linear2 = nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class BiggerModelForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = nn.Linear(5, 6)
        self.linear4 = nn.Linear(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


class ModuleWithUnusedSubModules(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return x @ self.linear.weight.t() + self.linear.bias


class ModelWithUnusedSubModulesForTest(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = ModuleWithUnusedSubModules(3, 4)
        self.linear2 = ModuleWithUnusedSubModules(4, 5)
        self.batchnorm = nn.BatchNorm1d(5)
        self.linear3 = ModuleWithUnusedSubModules(5, 6)
        self.linear4 = ModuleWithUnusedSubModules(6, 5)

    def forward(self, x):
        return self.linear4(self.linear3(self.batchnorm(self.linear2(self.linear1(x)))))


class DummyModel(nn.Module):
    """Simple model to do y=mx+b"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.a + self.b


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DummyModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ModuleWithUnusedSubModules,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_huggingface_accelerate(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

