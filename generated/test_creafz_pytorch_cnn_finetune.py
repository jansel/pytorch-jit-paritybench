import sys
_module = sys.modules[__name__]
del sys
cnn_finetune = _module
base = _module
contrib = _module
pretrainedmodels = _module
torchvision = _module
shims = _module
utils = _module
examples = _module
cifar10 = _module
setup = _module
tests = _module
conftest = _module
test_base = _module
test_pretrained_models = _module
test_torchvision_models = _module
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


from abc import ABCMeta


from abc import abstractmethod


from collections import namedtuple


import warnings


import torch


from torch import nn


from torch.utils import model_zoo


from torchvision import models as torchvision_models


from torch.autograd import Variable


import torchvision


import torchvision.transforms as transforms


import torch.nn as nn


import torch.optim as optim


import re


import types


import numpy as np


MODEL_REGISTRY = {}


class ModelRegistryMeta(type):
    """Metaclass that registers all model names defined in model_names property
    of a descendant class in the global MODEL_REGISTRY.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if 'model_names' in namespace:
            for model_name in namespace['model_names']:
                if model_name in MODEL_REGISTRY:
                    current_class = "<class '{module}.{qualname}'>".format(module=namespace['__module__'], qualname=namespace['__qualname__'])
                    warnings.warn("{current_class} redefined model_name '{model_name}'that was already registered by {previous_class}".format(current_class=current_class, model_name=model_name, previous_class=MODEL_REGISTRY[model_name]))
                MODEL_REGISTRY[model_name] = cls
        return cls


class ModelWrapperMeta(ABCMeta, ModelRegistryMeta):
    """An intermediate class that allows usage of both
    ABCMeta and ModelRegistryMeta simultaneously
    """
    pass


default = object()


def product(iterable):
    return functools.reduce(operator.mul, iterable)


class ModelWrapperBase(nn.Module, metaclass=ModelWrapperMeta):
    """Base class for all wrappers. To create a new wrapper you should
    subclass it and add model names that are supported by the wrapper to
    the model_names property. Those model names will be automatically
    registered in the global MODEL_REGISTRY upon class initialization.
    """
    flatten_features_output = True

    def __init__(self, *, model_name, num_classes, pretrained, dropout_p, pool, classifier_factory, use_original_classifier, input_size, original_model_state_dict, catch_output_size_exception):
        super().__init__()
        if num_classes < 1:
            raise ValueError('num_classes should be greater or equal to 1')
        if use_original_classifier and classifier_factory:
            raise ValueError("You can't use classifier_factory when use_original_classifier is set to True")
        self.check_args(model_name=model_name, num_classes=num_classes, dropout_p=dropout_p, pretrained=pretrained, pool=pool, classifier_fn=classifier_factory, use_original_classifier=use_original_classifier, input_size=input_size)
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.catch_output_size_exception = catch_output_size_exception
        original_model = self.get_original_model()
        if original_model_state_dict is not None:
            original_model.load_state_dict(original_model_state_dict)
        self._features = self.get_features(original_model)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else None
        self.pool = self.get_pool() if pool is default else pool
        self.input_size = input_size
        if pretrained:
            self.original_model_info = self.get_original_model_info(original_model)
        else:
            self.original_model_info = None
        if input_size:
            classifier_in_features = self.calculate_classifier_in_features(original_model)
        else:
            classifier_in_features = self.get_classifier_in_features(original_model)
        if use_original_classifier:
            classifier = self.get_original_classifier(original_model)
        elif classifier_factory:
            classifier = classifier_factory(classifier_in_features, num_classes)
        else:
            classifier = self.get_classifier(classifier_in_features, num_classes)
        self._classifier = classifier

    @abstractmethod
    def get_original_model(self):
        pass

    @abstractmethod
    def get_features(self, original_model):
        pass

    @abstractmethod
    def get_classifier_in_features(self, original_model):
        pass

    def get_original_model_info(self, original_model):
        return None

    def calculate_classifier_in_features(self, original_model):
        with no_grad_variable(torch.zeros(1, 3, *self.input_size)) as input_var:
            self.eval()
            try:
                output = self.features(input_var)
                if self.pool is not None:
                    output = self.pool(output)
            except RuntimeError as e:
                if self.catch_output_size_exception and 'Output size is too small' in str(e):
                    _, _, traceback = sys.exc_info()
                    message = 'Input size {input_size} is too small for this model. Try increasing the input size of images and change the value of input_size argument accordingly.'.format(input_size=self.input_size)
                    raise RuntimeError(message).with_traceback(traceback)
                else:
                    raise e
            self.train()
            return product(output.size()[1:])

    def check_args(self, **kwargs):
        pass

    def get_pool(self):
        return nn.AdaptiveAvgPool2d(1)

    def get_classifier(self, in_features, num_classes):
        return nn.Linear(in_features, self.num_classes)

    def get_original_classifier(self, original_model):
        raise NotImplementedError()

    def features(self, x):
        return self._features(x)

    def classifier(self, x):
        return self._classifier(x)

    def forward(self, x):
        x = self.features(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.flatten_features_output:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


ModelInfo = namedtuple('ModelInfo', ['input_space', 'input_size', 'input_range', 'mean', 'std'])


class PretrainedModelsWrapper(ModelWrapperBase):

    def get_original_model_info(self, original_model):
        return ModelInfo(input_space=original_model.input_space, input_size=original_model.input_size, input_range=original_model.input_range, mean=original_model.mean, std=original_model.std)

    def get_original_model(self):
        model = getattr(pretrainedmodels, self.model_name)
        if self.pretrained:
            model_kwargs = {'pretrained': 'imagenet', 'num_classes': 1000}
        else:
            model_kwargs = {'pretrained': None}
        return model(**model_kwargs)

    def get_features(self, original_model):
        return original_model.features

    def get_original_classifier(self, original_model):
        return original_model.last_linear

    def get_classifier_in_features(self, original_model):
        return original_model.last_linear.in_features


class ResNeXtWrapper(PretrainedModelsWrapper):
    model_names = ['resnext101_32x4d', 'resnext101_64x4d']


class NasNetWrapper(PretrainedModelsWrapper):
    model_names = ['nasnetalarge']

    def get_features(self, original_model):
        features = nn.Module()
        for name, module in list(original_model.named_children())[:-3]:
            features.add_module(name, module)
        return features

    def features(self, x):
        x_conv0 = self._features.conv0(x)
        x_stem_0 = self._features.cell_stem_0(x_conv0)
        x_stem_1 = self._features.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self._features.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self._features.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self._features.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self._features.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self._features.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self._features.cell_5(x_cell_4, x_cell_3)
        x_reduction_cell_0 = self._features.reduction_cell_0(x_cell_5, x_cell_4)
        x_cell_6 = self._features.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self._features.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self._features.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self._features.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self._features.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self._features.cell_11(x_cell_10, x_cell_9)
        x_reduction_cell_1 = self._features.reduction_cell_1(x_cell_11, x_cell_10)
        x_cell_12 = self._features.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self._features.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self._features.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self._features.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self._features.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self._features.cell_17(x_cell_16, x_cell_15)
        x = self._features.relu(x_cell_17)
        return x


class NasNetMobileWrapper(PretrainedModelsWrapper):
    model_names = ['nasnetamobile']

    def get_features(self, original_model):
        features = nn.Module()
        for name, module in list(original_model.named_children())[:-3]:
            features.add_module(name, module)
        return features

    def features(self, input):
        x_conv0 = self._features.conv0(input)
        x_stem_0 = self._features.cell_stem_0(x_conv0)
        x_stem_1 = self._features.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self._features.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self._features.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self._features.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self._features.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self._features.reduction_cell_0(x_cell_3, x_cell_2)
        x_cell_6 = self._features.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self._features.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self._features.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self._features.cell_9(x_cell_8, x_cell_7)
        x_reduction_cell_1 = self._features.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self._features.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self._features.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self._features.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self._features.cell_15(x_cell_14, x_cell_13)
        x = self._features.relu(x_cell_15)
        return x


class InceptionResNetV2Wrapper(PretrainedModelsWrapper):
    model_names = ['inceptionresnetv2']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-2])


class DPNWrapper(PretrainedModelsWrapper):
    model_names = ['dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']
    flatten_features_output = False

    def get_original_model(self):
        model = getattr(pretrainedmodels, self.model_name)
        if self.pretrained:
            if self.model_name in {'dpn68b', 'dpn92', 'dpn107'}:
                pretrained = 'imagenet+5k'
            else:
                pretrained = 'imagenet'
            model_kwargs = {'pretrained': pretrained, 'num_classes': 1000}
        else:
            model_kwargs = {'pretrained': None}
        return model(**model_kwargs)

    def get_classifier_in_features(self, original_model):
        return original_model.last_linear.in_channels

    def get_classifier(self, in_features, num_classes):
        return nn.Conv2d(in_features, num_classes, kernel_size=1, bias=True)

    def classifier(self, x):
        x = self._classifier(x)
        if not self.training:
            x = adaptive_avgmax_pool2d(x, pool_type='avgmax')
        return x.view(x.size(0), -1)


class InceptionV4Wrapper(PretrainedModelsWrapper):
    model_names = ['inception_v4']

    def get_original_model(self):
        if self.pretrained:
            model_kwargs = {'pretrained': 'imagenet', 'num_classes': 1000}
        else:
            model_kwargs = {'pretrained': None}
        return pretrainedmodels.inceptionv4(**model_kwargs)


class XceptionWrapper(PretrainedModelsWrapper):
    model_names = ['xception']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-1])


class SenetWrapper(PretrainedModelsWrapper):
    model_names = ['senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']

    def get_features(self, original_model):
        return nn.Sequential(original_model.layer0, original_model.layer1, original_model.layer2, original_model.layer3, original_model.layer4)


class PNasNetWrapper(PretrainedModelsWrapper):
    model_names = ['pnasnet5large']

    def get_features(self, original_model):
        features = nn.Module()
        for name, module in list(original_model.named_children())[:-3]:
            features.add_module(name, module)
        return features

    def features(self, x):
        x_conv_0 = self._features.conv_0(x)
        x_stem_0 = self._features.cell_stem_0(x_conv_0)
        x_stem_1 = self._features.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self._features.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self._features.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self._features.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self._features.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self._features.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self._features.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self._features.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self._features.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self._features.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self._features.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self._features.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self._features.cell_11(x_cell_9, x_cell_10)
        x = self._features.relu(x_cell_11)
        return x


class PolyNetWrapper(PretrainedModelsWrapper):
    model_names = ['polynet']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-3])


class TorchvisionWrapper(ModelWrapperBase):

    def get_original_model_info(self, original_model):
        return ModelInfo(input_space='RGB', input_size=[3, 224, 224], input_range=[0, 1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_original_model(self):
        model = getattr(torchvision_models, self.model_name)
        return model(pretrained=self.pretrained)

    def get_original_classifier(self, original_model):
        return original_model.classifier


class ResNetWrapper(TorchvisionWrapper):
    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

    def get_features(self, original_model):
        return nn.Sequential(*list(original_model.children())[:-2])

    def get_classifier_in_features(self, original_model):
        return original_model.fc.in_features

    def get_original_classifier(self, original_model):
        return original_model.fc


class DenseNetWrapper(TorchvisionWrapper):
    model_names = ['densenet121', 'densenet169', 'densenet201', 'densenet161']

    def get_features(self, original_model):
        return nn.Sequential(*original_model.features, nn.ReLU(inplace=True))

    def get_classifier_in_features(self, original_model):
        return original_model.classifier.in_features


class NetWithFcClassifierWrapper(TorchvisionWrapper):

    def check_args(self, model_name, pool, use_original_classifier, input_size, num_classes, pretrained, **kwargs):
        super().check_args()
        if input_size is None:
            raise Exception('You must provide input_size, e.g. make_model({model_name}, num_classes={num_classes}, pretrained={pretrained}, input_size=(224, 224)'.format(model_name=model_name, num_classes=num_classes, pretrained=pretrained))
        if use_original_classifier:
            if pool is not None and pool is not default:
                raise Exception("You can't use pool layer with the original classifier")
            if input_size != (224, 224):
                raise Exception('For the original classifier input_size value must be (224, 224)')

    def get_classifier_in_features(self, original_model):
        return self.calculate_classifier_in_features(original_model)

    def get_features(self, original_model):
        return original_model.features

    def get_pool(self):
        return None


class AlexNetWrapper(NetWithFcClassifierWrapper):
    model_names = ['alexnet']

    def get_classifier(self, in_features, num_classes):
        return nn.Sequential(nn.Linear(in_features, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))


class VGGWrapper(NetWithFcClassifierWrapper):
    model_names = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

    def get_classifier(self, in_features, num_classes):
        return nn.Sequential(nn.Linear(in_features, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))


class SqueezeNetWrapper(TorchvisionWrapper):
    model_names = ['squeezenet1_0', 'squeezenet1_1']

    def get_features(self, original_model):
        return original_model.features

    def get_pool(self):
        return None

    def get_classifier_in_features(self, original_model):
        return self.calculate_classifier_in_features(original_model)

    def get_classifier(self, in_features, num_classes):
        classifier = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        return classifier

    def forward(self, x):
        x = self.features(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


class InceptionWrapper(ModelWrapperBase):

    def get_original_model_info(self, original_model):
        return ModelInfo(input_space='RGB', input_size=[3, 299, 299], input_range=[0, 1], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_original_model(self):
        model = getattr(torchvision_models, self.model_name)
        return model(pretrained=self.pretrained)

    def get_original_classifier(self, original_model):
        return original_model.fc

    def get_classifier_in_features(self, original_model):
        return original_model.fc.in_features


class InceptionV3Wrapper(InceptionWrapper):
    model_names = ['inception_v3']

    def get_features(self, original_model):
        features = nn.Sequential(original_model.Conv2d_1a_3x3, original_model.Conv2d_2a_3x3, original_model.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2), original_model.Conv2d_3b_1x1, original_model.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2), original_model.Mixed_5b, original_model.Mixed_5c, original_model.Mixed_5d, original_model.Mixed_6a, original_model.Mixed_6b, original_model.Mixed_6c, original_model.Mixed_6d, original_model.Mixed_6e, original_model.Mixed_7a, original_model.Mixed_7b, original_model.Mixed_7c)
        return features


class GoogLeNetWrapper(InceptionWrapper):
    model_names = ['googlenet']

    def get_features(self, original_model):
        features = nn.Sequential(original_model.conv1, original_model.maxpool1, original_model.conv2, original_model.conv3, original_model.maxpool2, original_model.inception3a, original_model.inception3b, original_model.maxpool3, original_model.inception4a, original_model.inception4b, original_model.inception4c, original_model.inception4d, original_model.inception4e, original_model.maxpool4, original_model.inception5a, original_model.inception5b)
        return features


class MobileNetV2Wrapper(TorchvisionWrapper):
    model_names = ['mobilenet_v2']

    def get_features(self, original_model):
        return original_model.features

    def get_original_classifier(self, original_model):
        return original_model.classifier[-1]

    def get_classifier_in_features(self, original_model):
        return original_model.classifier[-1].in_features


class ShuffleNetV2Wrapper(TorchvisionWrapper):
    model_names = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']

    def get_features(self, original_model):
        features = nn.Sequential(original_model.conv1, original_model.maxpool, original_model.stage2, original_model.stage3, original_model.stage4, original_model.conv5)
        return features

    def get_original_classifier(self, original_model):
        return original_model.fc

    def get_classifier_in_features(self, original_model):
        return original_model.fc.in_features

