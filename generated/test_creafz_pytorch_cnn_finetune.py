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


from abc import ABCMeta


from abc import abstractmethod


from collections import namedtuple


import warnings


import torch


from torch import nn


from torch.utils import model_zoo


from torch.autograd import Variable


import torch.nn as nn


import torch.optim as optim


import types


default = object()


def product(iterable):
    return functools.reduce(operator.mul, iterable)


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
                    current_class = "<class '{module}.{qualname}'>".format(
                        module=namespace['__module__'], qualname=namespace[
                        '__qualname__'])
                    warnings.warn(
                        "{current_class} redefined model_name '{model_name}'that was already registered by {previous_class}"
                        .format(current_class=current_class, model_name=
                        model_name, previous_class=MODEL_REGISTRY[model_name]))
                MODEL_REGISTRY[model_name] = cls
        return cls


class ModelWrapperMeta(ABCMeta, ModelRegistryMeta):
    """An intermediate class that allows usage of both
    ABCMeta and ModelRegistryMeta simultaneously
    """
    pass


class ModelWrapperBase(nn.Module, metaclass=ModelWrapperMeta):
    """Base class for all wrappers. To create a new wrapper you should
    subclass it and add model names that are supported by the wrapper to
    the model_names property. Those model names will be automatically
    registered in the global MODEL_REGISTRY upon class initialization.
    """
    flatten_features_output = True

    def __init__(self, *, model_name, num_classes, pretrained, dropout_p,
        pool, classifier_factory, use_original_classifier, input_size,
        original_model_state_dict, catch_output_size_exception):
        super().__init__()
        if num_classes < 1:
            raise ValueError('num_classes should be greater or equal to 1')
        if use_original_classifier and classifier_factory:
            raise ValueError(
                "You can't use classifier_factory when use_original_classifier is set to True"
                )
        self.check_args(model_name=model_name, num_classes=num_classes,
            dropout_p=dropout_p, pretrained=pretrained, pool=pool,
            classifier_fn=classifier_factory, use_original_classifier=
            use_original_classifier, input_size=input_size)
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
            self.original_model_info = self.get_original_model_info(
                original_model)
        else:
            self.original_model_info = None
        if input_size:
            classifier_in_features = self.calculate_classifier_in_features(
                original_model)
        else:
            classifier_in_features = self.get_classifier_in_features(
                original_model)
        if use_original_classifier:
            classifier = self.get_original_classifier(original_model)
        elif classifier_factory:
            classifier = classifier_factory(classifier_in_features, num_classes
                )
        else:
            classifier = self.get_classifier(classifier_in_features,
                num_classes)
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
        with no_grad_variable(torch.zeros(1, 3, *self.input_size)
            ) as input_var:
            self.eval()
            try:
                output = self.features(input_var)
                if self.pool is not None:
                    output = self.pool(output)
            except RuntimeError as e:
                if (self.catch_output_size_exception and 
                    'Output size is too small' in str(e)):
                    _, _, traceback = sys.exc_info()
                    message = (
                        'Input size {input_size} is too small for this model. Try increasing the input size of images and change the value of input_size argument accordingly.'
                        .format(input_size=self.input_size))
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_creafz_pytorch_cnn_finetune(_paritybench_base):
    pass
