import sys
_module = sys.modules[__name__]
del sys
conf = _module
dcgan = _module
digit_recognizer = _module
dogs_vs_cats = _module
flower_classification = _module
flower_inference = _module
sorghum_fgvc9 = _module
wheat_efficientdet = _module
imdb_sentiment_classification = _module
natural_questions_qa = _module
toxic_multilabel_classification = _module
setup = _module
tez = _module
callbacks = _module
early_stopping = _module
progress = _module
tensorboard = _module
datasets = _module
generic = _module
image_classification = _module
image_segmentation = _module
enums = _module
logger = _module
model = _module
config = _module
model = _module
tez = _module
utils = _module

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


import torch.nn as nn


from sklearn import metrics


from sklearn import model_selection


import torchvision


from sklearn.model_selection import train_test_split


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from torch.utils.tensorboard import SummaryWriter


import warnings


import time


from typing import Optional


from torch.utils.data import DataLoader


import random


class DigitRecognizerModel(nn.Module):

    def __init__(self, model_name, num_classes, learning_rate, n_train_steps):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        self.model = timm.create_model(model_name, pretrained=True, in_chans=1, num_classes=num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        targets = targets.cpu().detach().numpy()
        acc = metrics.accuracy_score(targets, outputs)
        acc = torch.tensor(acc, device=device)
        return {'accuracy': acc}

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True, mode='max', threshold=0.0001)
        return opt, sch

    def forward(self, image, targets=None):
        x = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}


class CatsDogsModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        f1 = metrics.f1_score(targets, outputs, average='macro')
        return {'f1': torch.tensor(f1, device=device)}

    def optimizer_scheduler(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True, mode='max', threshold=0.0001)
        return opt, sch

    def forward(self, image, targets=None):
        outputs = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, 0, {}


class FlowerModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def forward(self, image, targets=None):
        outputs = self.model(image)
        return outputs, 0, {}


class SorghumModel(nn.Module):

    def __init__(self, model_name, num_classes, learning_rate, n_train_steps):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        self.model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.device.type
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        targets = targets.cpu().detach().numpy()
        acc = metrics.accuracy_score(targets, outputs)
        acc = torch.tensor(acc, device=device)
        return {'accuracy': acc}

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True, mode='max', threshold=0.0001)
        return opt, sch

    def forward(self, image, targets=None):
        x = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}


def create_model(num_classes=1, image_size=512, architecture='tf_efficientnetv2_l'):
    efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(name='tf_efficientnetv2_l', backbone_name='tf_efficientnetv2_l', backbone_args=dict(drop_path_rate=0.2), num_classes=num_classes, url='')
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchTrain(net, config)


class WheatModel(nn.Module):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.base_model = create_model(num_classes=1, image_size=1024, architecture='tf_efficientdet_d1')

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
        return opt, None

    def forward(self, images, targets):
        outputs = self.base_model(images, targets)
        if targets is not None:
            loss = outputs['loss']
            return outputs, loss, {}
        return outputs, 0, {}


class IMDBModel(nn.Module):

    def __init__(self, model_name, num_train_steps, learning_rate):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-07
        config = AutoConfig.from_pretrained(model_name)
        config.update({'output_hidden_states': True, 'hidden_dropout_prob': hidden_dropout_prob, 'layer_norm_eps': layer_norm_eps, 'add_pooling_layer': False, 'num_labels': 1})
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001}, {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        opt = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate)
        sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=self.num_train_steps)
        return opt, sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        device = targets.device.type
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {'accuracy': torch.tensor(accuracy, device=device)}

    def forward(self, ids, mask, token_type_ids, targets=None):
        transformer_out = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
        out = transformer_out.pooler_output
        out = self.dropout(out)
        output = self.output(out)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) ->str:
        return f'AverageMeter(val={self.val}, avg={self.avg}, sum={self.sum}, count={self.count})'


class Callback:

    def on_epoch_start(self, tez_trainer, **kwargs):
        return

    def on_epoch_end(self, tez_trainer, **kwargs):
        return

    def on_train_epoch_start(self, tez_trainer, **kwargs):
        return

    def on_train_epoch_end(self, tez_trainer, **kwargs):
        return

    def on_valid_epoch_start(self, tez_trainer, **kwargs):
        return

    def on_valid_epoch_end(self, tez_trainer, **kwargs):
        return

    def on_train_step_start(self, tez_trainer, **kwargs):
        return

    def on_train_step_end(self, tez_trainer, **kwargs):
        return

    def on_valid_step_start(self, tez_trainer, **kwargs):
        return

    def on_valid_step_end(self, tez_trainer, **kwargs):
        return

    def on_test_step_start(self, tez_trainer, **kwargs):
        return

    def on_test_step_end(self, tez_trainer, **kwargs):
        return

    def on_train_start(self, tez_trainer, **kwargs):
        return

    def on_train_end(self, tez_trainer, **kwargs):
        return


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        """
        Instead of inheriting from nn.Module, you import tez and inherit from tez.Model
        """
        super().__init__(*args, **kwargs)
        self.train_loader = None
        self.valid_loader = None
        self.optimizer = None
        self.scheduler = None
        self.step_scheduler_after = None
        self.step_scheduler_metric = None
        self.current_epoch = 0
        self.current_train_step = 0
        self.current_valid_step = 0
        self._model_state = None
        self._train_state = None
        self.device = None
        self._callback_runner = None
        self.fp16 = False
        self.scaler = None
        self.accumulation_steps = 0
        self.batch_index = 0
        self.metrics = {}
        self.metrics['train'] = {}
        self.metrics['valid'] = {}
        self.metrics['test'] = {}
        self.clip_grad_norm = None
        self.using_tpu = False

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, value):
        self._model_state = value

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, value):
        self._train_state = value
        if self._callback_runner is not None:
            self._callback_runner(value)

    def name_to_metric(self, metric_name):
        if metric_name == 'current_epoch':
            return self.current_epoch
        v_1 = metric_name.split('_')[0]
        v_2 = '_'.join(metric_name.split('_')[1:])
        return self.metrics[v_1][v_2]

    def _init_model(self, device, train_dataset, valid_dataset, train_sampler, valid_sampler, train_bs, valid_bs, n_jobs, callbacks, fp16, train_collate_fn, valid_collate_fn, train_shuffle, valid_shuffle, accumulation_steps, clip_grad_norm):
        if callbacks is None:
            callbacks = list()
        if n_jobs == -1:
            n_jobs = psutil.cpu_count()
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        if next(self.parameters()).device != self.device:
            self
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, num_workers=n_jobs, sampler=train_sampler, shuffle=train_shuffle, collate_fn=train_collate_fn)
        if self.valid_loader is None:
            if valid_dataset is not None:
                self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_bs, num_workers=n_jobs, sampler=valid_sampler, shuffle=valid_shuffle, collate_fn=valid_collate_fn)
        if self.optimizer is None:
            self.optimizer = self.fetch_optimizer()
        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.amp.GradScaler()
        self._callback_runner = CallbackRunner(callbacks, self)
        self.train_state = enums.TrainingState.TRAIN_START

    def monitor_metrics(self, *args, **kwargs):
        return

    def loss(self, *args, **kwargs):
        return

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def model_fn(self, data):
        for key, value in data.items():
            data[key] = value
        if self.fp16:
            with torch.amp.autocast():
                output, loss, metrics = self(**data)
        else:
            output, loss, metrics = self(**data)
        return output, loss, metrics

    def train_one_step(self, data):
        if self.accumulation_steps == 1 and self.batch_index == 0:
            self.zero_grad()
        _, loss, metrics = self.model_fn(data)
        loss = loss / self.accumulation_steps
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        if (self.batch_index + 1) % self.accumulation_steps == 0:
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.using_tpu:
                xm.optimizer_step(self.optimizer, barrier=True)
            else:
                self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == 'batch':
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            if self.batch_index > 0:
                self.zero_grad()
        return loss, metrics

    def validate_one_step(self, data):
        _, loss, metrics = self.model_fn(data)
        return loss, metrics

    def predict_one_step(self, data):
        output, _, _ = self.model_fn(data)
        return output

    def update_metrics(self, losses, monitor):
        self.metrics[self._model_state.value].update(monitor)
        self.metrics[self._model_state.value]['loss'] = losses.avg

    def train_one_epoch(self, data_loader):
        self.train()
        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.batch_index = b_idx
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_one_step(data)
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            losses.update(loss.item() * self.accumulation_steps, data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            self.current_train_step += 1
            if not self.using_tpu:
                tk0.set_postfix(loss=losses.avg, stage='train', **monitor)
            if self.using_tpu:
                None
        if not self.using_tpu:
            tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def validate_one_epoch(self, data_loader):
        self.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                loss, metrics = self.validate_one_step(data)
            self.train_state = enums.TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg
            if not self.using_tpu:
                tk0.set_postfix(loss=losses.avg, stage='valid', **monitor)
            self.current_valid_step += 1
        if not self.using_tpu:
            tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)
        return losses.avg

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output

    def predict(self, dataset, sampler=None, batch_size=16, n_jobs=1, collate_fn=None):
        if next(self.parameters()).device != self.device:
            self
        if n_jobs == -1:
            n_jobs = psutil.cpu_count()
        if batch_size == 1:
            n_jobs = 0
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_jobs, sampler=sampler, collate_fn=collate_fn, pin_memory=True)
        if self.training:
            self.eval()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        for _, data in enumerate(tk0):
            with torch.no_grad():
                out = self.predict_one_step(data)
                out = self.process_output(out)
                yield out
            if not self.using_tpu:
                tk0.set_postfix(stage='test')
        if not self.using_tpu:
            tk0.close()

    def save(self, model_path, weights_only=False):
        model_state_dict = self.state_dict()
        if weights_only:
            if self.using_tpu:
                xm.save(model_state_dict, model_path)
            else:
                torch.save(model_state_dict, model_path)
            return
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict['state_dict'] = model_state_dict
        model_dict['optimizer'] = opt_state_dict
        model_dict['scheduler'] = sch_state_dict
        model_dict['epoch'] = self.current_epoch
        model_dict['fp16'] = self.fp16
        if self.using_tpu:
            xm.save(model_dict, model_path)
        else:
            torch.save(model_dict, model_path)

    def load(self, model_path, weights_only=False, device='cuda'):
        if device == 'tpu':
            if XLA_AVAILABLE is False:
                raise RuntimeError('XLA is not available')
            else:
                self.using_tpu = True
                device = xm.xla_device()
        self.device = device
        if next(self.parameters()).device != self.device:
            self
        model_dict = torch.load(model_path, map_location=torch.device(device))
        if weights_only:
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(model_dict['state_dict'])

    def fit(self, train_dataset, valid_dataset=None, train_sampler=None, valid_sampler=None, device='cuda', epochs=10, train_bs=16, valid_bs=16, n_jobs=8, callbacks=None, fp16=False, train_collate_fn=None, valid_collate_fn=None, train_shuffle=True, valid_shuffle=False, accumulation_steps=1, clip_grad_norm=None):
        None
        """
        The model fit function. Heavily inspired by tf/keras, this function is the core of Tez and this is the only
        function you need to train your models.

        """
        if device == 'tpu':
            if XLA_AVAILABLE is False:
                raise RuntimeError('XLA is not available. Please install pytorch_xla')
            else:
                self.using_tpu = True
                fp16 = False
                device = xm.xla_device()
        self._init_model(device=device, train_dataset=train_dataset, valid_dataset=valid_dataset, train_sampler=train_sampler, valid_sampler=valid_sampler, train_bs=train_bs, valid_bs=valid_bs, n_jobs=n_jobs, callbacks=callbacks, fp16=fp16, train_collate_fn=train_collate_fn, valid_collate_fn=valid_collate_fn, train_shuffle=train_shuffle, valid_shuffle=valid_shuffle, accumulation_steps=accumulation_steps, clip_grad_norm=clip_grad_norm)
        for _ in range(epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                valid_loss = self.validate_one_epoch(self.valid_loader)
                self.train_state = enums.TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == 'epoch':
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == 'end':
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END

