import sys
_module = sys.modules[__name__]
del sys
base = _module
base_data_loader = _module
base_model = _module
base_trainer = _module
classifier = _module
data_loader = _module
data_loaders = _module
CRNN_module = _module
Densenet_module = _module
I3D_module = _module
Resnet_module = _module
Resnetv2_module = _module
Resnext_module = _module
WideResnet_module = _module
densenet = _module
i3d = _module
resnet = _module
resnetv2 = _module
resnext = _module
wideresnet = _module
models = _module
DenseNet_module = _module
Inception_module = _module
InceptionresnetV2_module = _module
ResNet_module = _module
ResNext101_module = _module
ResNext101_module2 = _module
Squeezenet_module = _module
Vgg_module = _module
Xception_module = _module
densenet = _module
inception = _module
inceptionresnetv2 = _module
resnet = _module
resnext = _module
squeezenet = _module
vgg = _module
xception = _module
loss = _module
metric = _module
model_template = _module
train = _module
trainer = _module
trainer = _module
utils = _module
logger = _module
util = _module
visualization = _module

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


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torch.utils.data.sampler import SubsetRandomSampler


import logging


import torch.nn as nn


import math


import torch


from collections import OrderedDict


from collections import Iterable


from torchvision import datasets


from torchvision.datasets import ImageFolder


import torchvision.transforms.functional as TF


import torchvision.transforms as T


import torch.nn.functional as F


from functools import partial


import torch.utils.model_zoo as model_zoo


from functools import reduce


import torch.nn.init as init


from torch.nn import init


import re


from torch import load as TorchLoad


from torchvision.utils import make_grid


from sklearn.metrics import f1_score


from sklearn.metrics import accuracy_score


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.img_size = None

    def forward(self, input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def isValidSize(self, input):
        assert self.img_size == (input.size(2), input.size(3)), '输入数据与model的设定输入尺寸不同！！'

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)


class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)


class WriterTensorboardX:

    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ImportError:
                message = 'Warning: TensorboardX visualization is configured to use, but currently not installed on this machine. ' + "Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."
                logger.warning(message)
        self.step = 0
        self.mode = ''
        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_logger = train_logger
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.verbose_per_epoch = cfg_trainer.get('verbose_per_epoch', 10)
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)
        self.start_epoch = 1
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], config['name'], start_time)
        writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, cfg_trainer['tensorboardX'])
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)
        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There's no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU's configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        not_improved_count = 0
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch + 1):
            result = self._train_epoch(epoch)
            log = OrderedDict({'epoch': epoch})
            for key, value in result.items():
                if key == 'metrics':
                    for i, mtr in enumerate(self.metrics):
                        log['train_' + mtr.__name__] = value[i]
                elif key == 'val_metrics':
                    for i, mtr in enumerate(self.metrics):
                        log['val_' + mtr.__name__] = value[i]
                else:
                    log[key] = value
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best or self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. Training stops.".format(self.early_stop))
                    break
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {'arch': arch, 'epoch': epoch, 'logger': self.train_logger, 'state_dict': self.model.state_dict(), 'monitor_best': self.mnt_best, 'config': self.config}
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info('Saving checkpoint: {} ...'.format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(state, best_path)
            self.logger.info('Saving current best: {} ...'.format('checkpoint_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info('Loading checkpoint: {} ...'.format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        if 'img_size' in checkpoint['config']['arch']['args'] and 'img_size' in self.config['arch']['args']:
            if isinstance(checkpoint['config']['arch']['args']['img_size'], list):
                checkpoint['config']['arch']['args']['img_size'] = tuple(checkpoint['config']['arch']['args']['img_size'])
            if isinstance(self.config['arch']['args']['img_size'], list):
                self.config['arch']['args']['img_size'] = tuple(self.config['arch']['args']['img_size'])
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + 'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, resume, config, data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, tensorboard_image=False):
        """
        steps_update: how many steps to update parameters, use it when memory is not enough 
        """
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.steps_update = self.config['trainer']['steps_update']
        self.steps_to_verb = len(self.data_loader) // self.verbose_per_epoch
        self.steps_to_verb = 1 if self.steps_to_verb <= 0 else self.steps_to_verb
        self.tensorboard_image = tensorboard_image

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(str(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        self.optimizer.zero_grad()
        batch_idx = 0
        for data, target in self.data_loader:
            batch_idx += 1
            data, target = data, target
            isUpdateOptim = self.steps_update == 1 or (batch_idx + 1) % self.steps_update == 0
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            if isUpdateOptim:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)
            if self.verbosity >= 2 and batch_idx % self.steps_to_verb == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, batch_idx * self.data_loader.batch_size, self.data_loader.n_samples, 100.0 * batch_idx / len(self.data_loader), loss.item()))
                if self.tensorboard_image:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        None
        log = OrderedDict()
        log['train_loss'] = total_loss / len(self.data_loader)
        log['metrics'] = (total_metrics / len(self.data_loader)).tolist()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            for key, value in val_log.items():
                log[key] = value
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def cal_f1_score(self, dataset='train'):
        if dataset == 'train':
            data_loader = self.data_loader
        else:
            data_loader = self.valid_data_loader
        self.model.eval()
        y_preds = y_trues = np.array([])
        batches = len(data_loader)
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data, target
                output = self.model(data)
                loss = self.loss(output, target).data.cpu().numpy()
                total_loss += loss
                y_pred = output.data.cpu().max(1)[1].numpy()
                y_true = target.data.cpu().numpy()
                y_preds = np.append(y_preds, y_pred)
                y_trues = np.append(y_trues, y_true)
                if batch_idx % 50 == 0:
                    None
        total_loss = total_loss / len(data_loader)
        res = f1_score(y_trues, y_preds, average='macro')
        acc = accuracy_score(y_trues, y_preds)
        None
        return res

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data, target
                output = self.model(data)
                loss = self.loss(output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                if self.tensorboard_image:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        log = OrderedDict()
        log['val_loss'] = total_val_loss / len(self.valid_data_loader)
        log['val_metrics'] = (total_val_metrics / len(self.valid_data_loader)).tolist()
        return log


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class classifier(BaseModel):
    """ 可以训练 """

    def __init__(self, model='xception', n_classes=10, img_size=(224, 224), data_dir=None, pretrained=False, pretrained_path='./pretrained/', n_gpu=1, default_init=True):
        """ 初始化时只初始化model model可以是string 也可以是自己创建的model """
        super(classifier, self).__init__()
        self.resume = None
        self.data_loader = None
        self.valid_data_loader = None
        self.train_logger = Logger()
        if isinstance(model, str):
            arch = {'type': model, 'args': {'n_class': n_classes, 'img_size': img_size, 'pretrained': pretrained, 'pretrained_path': pretrained_path}}
            self.config = {'name': model, 'arch': arch, 'n_gpu': n_gpu}
            self.model = get_instance(model_zoo, 'arch', self.config)
        elif callable(model):
            model_name = model.__class__.__name__
            arch = {'type': model_name, 'args': {'n_class': n_classes, 'img_size': img_size}}
            self.config = {'name': model_name, 'arch': arch, 'n_gpu': n_gpu}
            self.model = model
        else:
            self.logger.info('input type is invalid, please set model as str or a callable object')
            raise Exception('model: wrong input error')
        if default_init:
            self.config['loss'] = 'cls_loss'
            self.loss = getattr(module_loss, self.config['loss'])
            self.config['metrics'] = ['accuracy']
            self.metrics = [getattr(module_metric, met) for met in self.config['metrics']]
            self.config['optimizer'] = {'type': 'Adam', 'args': {'lr': 0.0003, 'weight_decay': 3e-05}}
            optimizer_params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = get_instance(torch.optim, 'optimizer', self.config, optimizer_params)
            self.config['lr_scheduler'] = {'type': 'StepLR', 'args': {'step_size': 50, 'gamma': 0.2}}
            self.lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config, self.optimizer)
            self.set_trainer()
        if data_dir:
            self.autoset_dataloader(data_dir, batch_size=64, shuffle=True, validation_split=0.2, num_workers=4, transform=None)

    def init_from_config(config_file, resume=None, user_defined_model=None):
        """
        if user_defined_model is callable, init classifier with the given model
        otherwise, use model in modelfeast
        """
        config = json.load(open(config_file))
        model_config = config['arch']['args']
        if callable(user_defined_model):
            kernal_model = user_defined_model
        else:
            kernal_model = config['arch']['type']
        clf = classifier(model=kernal_model, n_classes=model_config['n_class'], img_size=model_config['img_size'], data_dir=None, pretrained=model_config['pretrained'], pretrained_path=model_config['pretrained_path'], default_init=False)
        loss = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        optimizer_params = filter(lambda p: p.requires_grad, clf.model.parameters())
        optimizer = get_instance(torch.optim, 'optimizer', config, optimizer_params)
        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
        if 'args' in config['data_loader']:
            data_cng = config['data_loader']['args']
            transform = None if 'transform' not in data_cng else data_cng['transform']
            if not transform and config['arch']['args']['img_size']:
                config['data_loader']['args']['transform'] = config['arch']['args']['img_size']
        try:
            data_loader = get_instance(module_data, 'data_loader', config)
            valid_data_loader = data_loader.split_validation()
        except AttributeError:
            clf.logger.warning('Can not match data loader in config file!!! please set data loader manually!!!')
            data_loader = None
            valid_data_loader = None
            pass
        clf.config = config
        clf.loss = loss
        clf.resume = resume
        clf.metrics = metrics
        clf.optimizer = optimizer
        clf.lr_scheduler = lr_scheduler
        clf.data_loader = data_loader
        clf.valid_data_loader = valid_data_loader
        return clf

    def train(self):
        assert callable(self.model), 'model is not callable!!'
        assert callable(self.loss), 'loss is not callable!!'
        assert all(callable(met) for met in self.metrics), 'metrics is not callable!!'
        assert 'trainer' in self.config, "trainer hasn't been configured!!"
        assert isinstance(self.data_loader, Iterable), 'data_loader is not iterable!!'
        if hasattr(self.data_loader, 'classes'):
            true_classes = len(self.data_loader.classes)
            model_output = self.config['arch']['args']['n_class']
            assert true_classes == model_output, 'model分类数为{}，可是实际上有{}个类'.format(model_output, true_classes)
        if 'name' not in self.config:
            self.config['name'] = '_'.join(self.config['arch']['type'], self.config['data_loader']['type'])
        self.trainer = Trainer(self.model, self.loss, self.metrics, self.optimizer, resume=self.resume, config=self.config, data_loader=self.data_loader, valid_data_loader=self.valid_data_loader, lr_scheduler=self.lr_scheduler, train_logger=self.train_logger)
        self.trainer.train()

    def train_from(self, resume):
        self.resume = resume
        self.train()

    def set_optimizer(self, name='Adam', **kwargs):
        self.config['optimizer'] = {'type': name, 'args': kwargs}
        optimizer_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = get_instance(torch.optim, 'optimizer', self.config, optimizer_params)

    def set_lr_scheduler(self, name='StepLR', **kwargs):
        self.config['lr_scheduler'] = {'type': name, 'args': kwargs}
        self.lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', self.config, self.optimizer)

    def autoset_dataloader(self, folder, batch_size=16, shuffle=True, validation_split=0.2, num_workers=4, transform=None):
        """automatic generate data-loader from a given folder"""
        assert os.path.exists(folder), "data folder doesn't exit!!!"
        if not transform and self.config['arch']['args']['img_size']:
            transform = self.config['arch']['args']['img_size']
        self.data_loader = module_data.AutoDataLoader(folder, batch_size, shuffle, validation_split, num_workers, transform)
        self.valid_data_loader = self.data_loader.split_validation()
        true_classes = len(self.data_loader.classes)
        model_output = self.config['arch']['args']['n_class']
        assert true_classes == model_output, 'model output {} classes，but there are {} classes in dataset'.format(model_output, true_classes)
        self.config['data_loader'] = {'type': self.data_loader.__class__.__name__}
        self.config['data_loader']['args'] = {'data_dir': folder}
        self.config['data_loader']['args']['batch_size'] = batch_size
        self.config['data_loader']['args']['shuffle'] = shuffle
        self.config['data_loader']['args']['validation_split'] = validation_split
        self.config['data_loader']['args']['validation_split'] = validation_split
        self.config['data_loader']['args']['transform'] = transform

    def set_trainer(self, epochs=20, save_dir='saved/', save_period=2, verbosity=2, verbose_per_epoch=100, monitor='max val_accuracy', early_stop=10, tensorboardX=False, log_dir='saved/runs', steps_update=1):
        self.config['trainer'] = {'epochs': epochs, 'save_dir': save_dir, 'save_period': save_period, 'verbosity': verbosity, 'verbose_per_epoch': verbose_per_epoch, 'monitor': monitor, 'early_stop': early_stop, 'tensorboardX': tensorboardX, 'log_dir': log_dir, 'steps_update': steps_update}


class CNNEncoder(nn.Module):

    def __init__(self, model_name='resnet50', img_size=(32, 32), fc_hidden1=512, fc_hidden2=512, drop_p=0.3, out_channels=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNEncoder, self).__init__()
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        model = getattr(model_arch, model_name)(n_class=2, img_size=img_size)
        modules = list(model.children())[:-1]
        self.kernal = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, out_channels)

    def forward(self, x_3d):
        cnn_embed_seq = []
        with torch.no_grad():
            for t in range(x_3d.size(1)):
                x = self.kernal(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)
                x = self.bn1(self.fc1(x))
                x = F.relu(x)
                x = self.bn2(self.fc2(x))
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_p, training=self.training)
                x = self.fc3(x)
                cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class DecoderRNN(nn.Module):

    def __init__(self, in_channels=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, n_classes=2):
        super(DecoderRNN, self).__init__()
        self.RNN_input_size = in_channels
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.n_classes = n_classes
        self.LSTM = nn.LSTM(input_size=self.RNN_input_size, hidden_size=self.h_RNN, num_layers=h_RNN_layers, batch_first=True)
        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.n_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(BaseModel):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block 有4个block

        num_init_features (int) - 前置卷积层出来能有多少特征
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(30, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, 1000)
        self.conv_features = num_features
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        super(DenseNet, self).isValidSize(x)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def adaptive_set_fc(self, n_class):
        self.classifier = nn.Linear(self.conv_features, n_class)
        None
        None

    def cal_features(self, x):
        super(DenseNet, self).isValidSize(x)
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class _NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        if bn_layer:
            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0), bn(self.in_channels))
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)
        self.theta = None
        self.phi = None
        if mode in ['embedded_gaussian', 'dot_product']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            else:
                self.operation_function = self._dot_product
        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        else:
            raise NotImplementedError('Mode concatenation has not been implemented.')
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """
        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

    def _dot_product(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NONLocalBlock3D(_NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels, inter_channels=inter_channels, dimension=3, mode=mode, sub_sample=sub_sample, bn_layer=bn_layer)


class I3DResNet(nn.Module):

    def __init__(self, block, layers, n_classes=2, in_channels=1):
        self.inplanes = 32 if in_channels == 1 else 64
        super(I3DResNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer_inflat(block, 64, layers[0])
        self.temporalpool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.layer2 = self._make_layer_inflat(block, 128, layers[1], space_stride=2)
        self.layer3 = self._make_layer_inflat(block, 256, layers[2], space_stride=2)
        self.layer4 = self._make_layer_inflat(block, 512, layers[3], space_stride=2)
        self.avgdrop = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer_inflat(self, block, planes, blocks, space_stride=1):
        downsample = None
        if space_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(1, space_stride, space_stride), bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        time_kernel = 3
        layers.append(block(self.inplanes, planes, time_kernel, space_stride, downsample, addnon=None))
        self.inplanes = planes * block.expansion
        if blocks == 3:
            for i in range(1, blocks):
                if i % 2 == 1:
                    time_kernel = 3
                else:
                    time_kernel = 1
                layers.append(block(self.inplanes, planes, time_kernel))
        elif blocks == 4:
            for i in range(1, blocks):
                if i % 2 == 1:
                    time_kernel = 3
                    layers.append(block(self.inplanes, planes, time_kernel, addnon=True))
                else:
                    time_kernel = 1
                    layers.append(block(self.inplanes, planes, time_kernel))
        elif blocks == 23:
            for i in range(1, blocks):
                if i % 2 == 1:
                    time_kernel = 3
                else:
                    time_kernel = 1
                if i % 7 == 6:
                    addnon = True
                    layers.append(block(self.inplanes, planes, time_kernel, addnon=True))
                else:
                    layers.append(block(self.inplanes, planes, time_kernel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.temporalpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(x.size(0), -1)
        x = self.avgdrop(x)
        x = self.fc(x)
        return x

    def cal_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.temporalpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(x.size(0), -1)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    """
    downsample是一个网络， 如果存在的话，可以把输入的深度增加expansion倍，尺寸减小
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(BaseModel):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        zero_init_residual = True
        self.block_expansion = block.expansion
        """结构是7*7的卷积+3*3的池化，步长都是2，所以最终把原图缩小了1/4
        7*7的卷积： 假设输入长宽为x， 输出(x-7+2*3)/2+1 = (x-1)/2+1 
                   原文说是224进去，112出来，确实如此，不过除不尽，所以其实漏了一行和一列没处理，信息丢了
        3*3的池化：假设输入长宽为x， 输出(x-3+2*1)/2+1 = (x-1)/2+1 
                   原文说是112进去，56出来，确实如此，不过除不尽，也是漏了一行一列没做pool操作丢掉了
        最终，输出为原图的1/4，通道为64
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        """总共4组残差块 各组残差块内部，维度一样，分别是 64 128 256 512

        """
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        super(ResNet, self).isValidSize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def cal_features(self, x):
        super(ResNet, self).isValidSize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def adaptive_set_fc(self, n_class, h, w):
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, n_class)
        None
        None


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActivationBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class PreActivationBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.FloatTensor):
        zero_pads = zero_pads
    out = torch.cat([out.data, zero_pads], dim=1)
    return out


class PreActivationResNet(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', n_classes=400, in_channels=3):
        super(PreActivationResNet, self).__init__()
        first_channels = 64 if in_channels == 3 else 32
        self.inplanes = first_channels
        self.conv1 = nn.Conv3d(in_channels, first_channels, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(first_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, first_channels, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def cal_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        return x


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32, n_classes=400, in_channels=3):
        super(ResNeXt, self).__init__()
        first_channels = 64 if in_channels == 3 else 32
        self.inplanes = first_channels
        self.conv1 = nn.Conv3d(in_channels, first_channels, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(first_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class WideBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class WideResNet(nn.Module):

    def __init__(self, block, layers, k=1, shortcut_type='B', n_classes=400, in_channels=3):
        super(WideResNet, self).__init__()
        first_features = 64 if in_channels == 3 else 32
        self.inplanes = first_features
        self.conv1 = nn.Conv3d(in_channels, first_features, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(first_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * k, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128 * k, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256 * k, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512 * k, layers[3], shortcut_type, stride=2)
        self.fc = nn.Linear(512 * k * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3dbl_1(x)
        branch3x3 = self.branch3x3dbl_2(branch3x3)
        branch3x3 = self.branch3x3dbl_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch5x5, branch3x3, branch_pool], dim=1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch7x7x3, branch_pool], 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(BaseModel):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        super(InceptionResNetV2, self).isValidSize(x)
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool_1a(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def adaptive_set_fc(self, n_class):
        self.avgpool_1a = nn.AdaptiveAvgPool2d((1, 1))
        self.last_linear = nn.Linear(1536, n_class)


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):

    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):

    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(BaseModel):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        super(SqueezeNet, self).isValidSize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

    def adaptive_set_classifier(self, num_classes):
        self.num_classes = num_classes
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        None


def adaptive_classifier(features, n_class):
    layer = []
    ratio = features // n_class
    if ratio <= 80:
        layer += [nn.Linear(features, n_class)]
    elif ratio <= 256:
        h1_features = int(ratio ** 0.5) * n_class
        layer += [nn.Linear(features, h1_features), nn.ReLU(True)]
        layer += [nn.Linear(h1_features, n_class)]
    else:
        cube_root = int(ratio ** 0.33)
        h1_features = n_class * cube_root * cube_root
        h2_features = n_class * cube_root
        layer += [nn.Linear(features, h1_features), nn.ReLU(True), nn.Dropout()]
        layer += [nn.Linear(h1_features, h2_features), nn.ReLU(True), nn.Dropout()]
        layer += [nn.Linear(h2_features, n_class)]
    return nn.Sequential(*layer)


def get_Convlayer(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)


def construct_Conv_Block(block_cfg, img_channel):
    layer = []
    in_channel = img_channel
    for v in block_cfg:
        if v == 'M':
            layer += [nn.MaxPool2d(2, stride=2, padding=0)]
        else:
            layer += [get_Convlayer(in_channel, v)]
            layer += [nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channel = v
    return nn.Sequential(*layer)


def init_weight(Netwrk):
    for m in Netwrk.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def pretrained_classifier(n_class):
    return nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, n_class))


class vgg_Net(BaseModel):

    def __init__(self, cfg, param):
        """
        param为字典，包含网络需要的参数
        param['img_height']: image's height, must be 32's multiple
        param['img_width']: image's weight, must be 32's multiple
        """
        super(vgg_Net, self).__init__()
        self.features = construct_Conv_Block(cfg, 3)
        conv_height, conv_width = param['img_height'] // 32, param['img_width'] // 32
        self.fc_in_features = conv_height * conv_width * 512
        self.classifier = pretrained_classifier(1000)
        init_weight(self)

    def forward(self, x):
        super(vgg_Net, self).isValidSize(x)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def adjust_classifier(self, n_class):
        self.classifier = adaptive_classifier(self.fc_in_features, n_class)
        init_weight(self.classifier)
        None
        None


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(BaseModel):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = 1000
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        super(Xception, self).isValidSize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def adaptive_fc(self, n_class):
        self.num_classes = n_class
        self.fc = nn.Linear(2048, n_class)
        None
        None
        return


class Inception3(BaseModel):

    def __init__(self, num_classes, transform_input):
        super(Inception3, self).__init__()
        aux_logits = self.aux_logits = True
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        super(Inception3, self).isValidSize(x)
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training and self.aux_logits:
            return x, aux
        return x


resnext101_32x4d_features = nn.Sequential(nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((3, 3), (2, 2), (1, 1)), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(64, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), nn.Sequential(nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(128), nn.ReLU()), nn.Conv2d(128, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), nn.Sequential(nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), nn.Sequential(nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), nn.Sequential(nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 32, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))


class ResNeXt101_32x4d(BaseModel):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        super(ResNeXt101_32x4d, self).isValidSize(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def adaptive_set_fc(self, n_class):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_linear = nn.Linear(2048, n_class)


resnext101_64x4d_features = nn.Sequential(nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((3, 3), (2, 2), (1, 1)), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), nn.Sequential(nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(256), nn.ReLU()), nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(256)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), nn.Sequential(nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(512), nn.ReLU()), nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(512)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), nn.Sequential(nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(1024), nn.ReLU()), nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(1024)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())), nn.Sequential(nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(), nn.Conv2d(2048, 2048, (3, 3), (2, 2), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(2048), nn.ReLU()), nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), nn.Sequential(nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048))), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(), nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(2048), nn.ReLU()), nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU()), nn.Sequential(LambdaMap(lambda x: x, nn.Sequential(nn.Sequential(nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU(), nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias=False), nn.BatchNorm2d(2048), nn.ReLU()), nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False), nn.BatchNorm2d(2048)), Lambda(lambda x: x)), LambdaReduce(lambda x, y: x + y), nn.ReLU())))


class ResNeXt101_64x4d(BaseModel):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_64x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        super(ResNeXt101_64x4d, self).isValidSize(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def adaptive_set_fc(self, n_class):
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_linear = nn.Linear(2048, n_class)


class MnistModel(BaseModel):

    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {}),
     True),
    (DecoderRNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 300])], {}),
     False),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (PreActivationBasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_daili0015_ModelFeast(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

