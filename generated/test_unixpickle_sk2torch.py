import sys
_module = sys.modules[__name__]
del sys
svm_vector_field = _module
setup = _module
sk2torch = _module
dummy = _module
dummy_test = _module
gradient_boosting = _module
gradient_boosting_test = _module
kernel = _module
kernel_test = _module
label_binarizer = _module
label_binarizer_test = _module
linear_model = _module
linear_model_test = _module
min_max_scaler = _module
min_max_scaler_test = _module
nn = _module
nn_test = _module
nystroem = _module
nystroem_test = _module
pca = _module
pca_test = _module
pipeline = _module
pipeline_test = _module
stacking = _module
stacking_test = _module
standard_scaler = _module
standard_scaler_test = _module
svc = _module
svc_test = _module
svr = _module
svr_test = _module
tree = _module
tree_test = _module
ttr = _module
ttr_test = _module
util = _module
wrap = _module

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


import matplotlib.pyplot as plt


import numpy as np


import torch


from sklearn.svm import SVC


from typing import List


from typing import Type


from typing import Union


import torch.jit


import torch.nn as nn


from sklearn.dummy import DummyClassifier


from sklearn.dummy import DummyRegressor


import itertools


from collections import Counter


from abc import abstractmethod


import torch.nn.functional as F


from sklearn.base import BaseEstimator


from sklearn.ensemble import GradientBoostingClassifier


from sklearn.ensemble import GradientBoostingRegressor


from typing import Any


from typing import Optional


from typing import Tuple


from sklearn.metrics.pairwise import pairwise_kernels


from sklearn.preprocessing import LabelBinarizer


from copy import deepcopy


from sklearn.linear_model import LinearRegression


from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import LogisticRegressionCV


from sklearn.linear_model import Ridge


from sklearn.linear_model import RidgeClassifier


from sklearn.linear_model import RidgeClassifierCV


from sklearn.linear_model import RidgeCV


from sklearn.linear_model import SGDClassifier


from sklearn.linear_model import SGDRegressor


from sklearn.linear_model._base import LinearModel


from sklearn.svm import LinearSVC


from sklearn.svm import LinearSVR


import warnings


from typing import Callable


from sklearn.datasets import load_breast_cancer


from sklearn.datasets import load_digits


from sklearn.exceptions import ConvergenceWarning


from sklearn.preprocessing import MinMaxScaler


from sklearn.neural_network import MLPClassifier


from sklearn.neural_network import MLPRegressor


from sklearn.kernel_approximation import Nystroem


from sklearn.decomposition import PCA


from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import StackingClassifier


from sklearn.ensemble import StackingRegressor


from sklearn.svm import NuSVC


from sklearn.svm import SVR


from sklearn.svm import NuSVR


from sklearn.tree import DecisionTreeClassifier


from sklearn.tree import DecisionTreeRegressor


from sklearn.compose import TransformedTargetRegressor


class TorchDummyClassifier(nn.Module):

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [DummyClassifier]

    @classmethod
    def wrap(cls, obj: DummyClassifier) ->Union['TorchDummyClassifierSingle', 'TorchDummyClassifierMulti']:
        assert not obj.sparse_output_, 'sparse classifiers are not supported'
        if isinstance(obj.n_classes_, list):
            return TorchDummyClassifierMulti(singles=[TorchDummyClassifierSingle(classes=torch.from_numpy(obj.classes_[i]), class_prior=torch.from_numpy(obj.class_prior_[i]), strategy=obj.strategy, constant=torch.from_numpy(np.array(obj.constant[i])) if obj.constant is not None else torch.from_numpy(obj.classes_[i])[0]) for i in range(len(obj.n_classes_))])
        else:
            return TorchDummyClassifierSingle(classes=torch.from_numpy(obj.classes_), class_prior=torch.from_numpy(obj.class_prior_), strategy=obj.strategy, constant=torch.from_numpy(np.array(obj.constant)) if obj.constant is not None else torch.from_numpy(obj.classes_)[0])


class TorchDummyRegressor(nn.Module):

    def __init__(self, strategy: str, constant: torch.Tensor):
        super().__init__()
        if strategy == 'constant':
            self.register_buffer('constant', constant)
        else:
            self.constant = nn.Parameter(constant)

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [DummyRegressor]

    @classmethod
    def wrap(cls, obj: DummyRegressor) ->'TorchDummyRegressor':
        return cls(strategy=obj.strategy, constant=torch.from_numpy(obj.constant_).view(-1))

    def forward(self, x: torch.Tensor):
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        res = self.constant[None].repeat(len(x), 1)
        if res.shape[1] == 1:
            res = res.view(-1)
        return res


class _GradientBoostingStage(nn.Module):

    def __init__(self, trees: List[BaseEstimator]):
        super().__init__()
        self.trees = nn.ModuleList([wrap(x) for x in trees])

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return torch.stack([tree(x).view(-1) for tree in self.trees], dim=-1)


class _GradientBoostingBase(nn.Module):

    def __init__(self, obj: Union[GradientBoostingClassifier, GradientBoostingRegressor]):
        super().__init__()
        self.has_init = obj.init_ != 'zero'
        self.init = wrap(obj.init_) if self.has_init else nn.Identity()
        if not self.has_init:

            def dummy_fn(x: torch.Tensor) ->torch.Tensor:
                return x
            self.init.predict = dummy_fn
            self.init.predict_proba = dummy_fn
        self.stages = nn.ModuleList([_GradientBoostingStage(x) for x in obj.estimators_.tolist()])
        self.learning_rate = obj.learning_rate
        self.loss = obj.loss
        dimension = len(self.stages[0].trees)
        param = next(self.parameters())
        self.register_buffer('zero_out', torch.zeros(dimension, dtype=param.dtype, device=param.device))

    def _raw_output(self, x: torch.Tensor) ->torch.Tensor:
        out = self._init_outputs(x)
        for stage in self.stages:
            out = out + stage.forward(x) * self.learning_rate
        if out.shape[1] == 1:
            return out.view(-1)
        return out

    @abstractmethod
    def _init_outputs(self, x: torch.Tensor) ->torch.Tensor:
        pass


class TorchGradientBoostingClassifier(_GradientBoostingBase):

    def __init__(self, obj: GradientBoostingClassifier):
        super().__init__(obj)
        assert self.loss in ['exponential', 'deviance']
        self.register_buffer('classes', torch.from_numpy(obj.classes_))

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [GradientBoostingClassifier]

    @classmethod
    def wrap(cls, obj: GradientBoostingClassifier) ->'TorchGradientBoostingClassifier':
        return cls(obj)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        if self.loss == 'exponential':
            decisions = self.decision_function(x)
            assert len(decisions.shape) == 1
            return self.classes[(decisions >= 0).long()]
        proba = self.predict_proba(x)
        return self.classes[proba.argmax(-1)]

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        decisions = self.decision_function(x)
        if self.loss == 'deviance':
            if len(decisions.shape) == 1:
                return torch.stack([F.logsigmoid(-decisions), F.logsigmoid(decisions)], dim=-1)
            else:
                return F.log_softmax(decisions, dim=-1)
        else:
            return torch.stack([F.logsigmoid(-2 * decisions), F.logsigmoid(2 * decisions)], dim=-1)

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) ->torch.Tensor:
        return self._raw_output(x)

    def _init_outputs(self, x: torch.Tensor) ->torch.Tensor:
        if not self.has_init:
            return self.zero_out[None].repeat(len(x), 1)
        eps = 1.1920929e-07
        init_probs = self.init.predict_proba(x).clamp(eps, 1 - eps)
        if self.loss == 'exponential':
            assert init_probs.shape[1] == 2
            prob_pos = init_probs[:, 1]
            return (0.5 * (prob_pos / (1 - prob_pos)).log())[:, None]
        elif init_probs.shape[1] == 2:
            prob_pos = init_probs[:, 1]
            return (prob_pos / (1 - prob_pos)).log()[:, None]
        else:
            return init_probs.log()


class TorchGradientBoostingRegressor(_GradientBoostingBase):

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [GradientBoostingRegressor]

    @classmethod
    def wrap(cls, obj: GradientBoostingRegressor) ->'TorchGradientBoostingRegressor':
        return cls(obj)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        return self._raw_output(x)

    def _init_outputs(self, x: torch.Tensor) ->torch.Tensor:
        if not self.has_init:
            return self.zero_out[None].repeat(len(x), 1)
        return self.init.predict(x).view(len(x), -1)


class Kernel(nn.Module):

    def __init__(self, metric: str, gamma: Optional[float]=None, coef0: Optional[float]=None, degree: Optional[float]=None):
        super().__init__()
        if metric not in ['linear', 'poly', 'polynomial', 'rbf', 'sigmoid']:
            raise ValueError(f'unsupported kernel metric: {metric}')
        self.metric = metric
        self.has_gamma = gamma is not None
        self.gamma = float(gamma or 0.0)
        self.has_coef0 = coef0 is not None
        self.coef0 = float(coef0 or 0.0)
        self.has_degree = degree is not None
        self.degree = float(degree or 0.0)

    @classmethod
    def wrap(self, estimator: Any) ->'Kernel':
        if not isinstance(estimator.kernel, str):
            raise ValueError(f'kernel must be str, but got {estimator.kernel}')
        return Kernel(metric=estimator.kernel, gamma=estimator.gamma if not hasattr(estimator, '_gamma') else estimator._gamma, coef0=estimator.coef0, degree=estimator.degree)

    def forward(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        if self.metric == 'linear':
            return x @ y.t()
        elif self.metric in ['poly', 'polynomial']:
            return (self._gamma(x) * x @ y.t() + self._coef0()) ** self._degree()
        elif self.metric == 'rbf':
            x_norm = (x ** 2).sum(-1)[:, None]
            y_norm = (y ** 2).sum(-1)[None]
            dots = x @ y.t()
            dists = (x_norm + y_norm - 2 * dots).clamp_min(0)
            return (-self._gamma(x) * dists).exp()
        elif self.metric == 'sigmoid':
            return (self._gamma(x) * x @ y.t() + self._coef0()).tanh()
        raise RuntimeError(f'unsupported kernel: {self.metric}')

    def _gamma(self, x: torch.Tensor) ->float:
        if self.has_gamma:
            return self.gamma
        elif self.metric in ['rbf', 'sigmoid', 'poly', 'polynomial']:
            return 1.0 / x.shape[1]
        raise RuntimeError(f'unknown default gamma for kernel: {self.metric}')

    def _coef0(self) ->float:
        if self.has_coef0:
            return self.coef0
        elif self.metric in ['sigmoid', 'poly', 'polynomial']:
            return 1.0
        raise RuntimeError(f'unknown default coef0 for kernel: {self.metric}')

    def _degree(self) ->float:
        if self.has_degree:
            return self.degree
        elif self.metric in ['poly', 'polynomial']:
            return 3.0
        raise RuntimeError(f'unknown default degree for kernel: {self.metric}')


class TorchLabelBinarizer(nn.Module):

    def __init__(self, classes: torch.Tensor, neg_label: int, pos_label: int, y_type: str):
        super().__init__()
        self.register_buffer('classes', classes)
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.y_type = y_type

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [LabelBinarizer]

    @classmethod
    def wrap(cls, obj: LabelBinarizer) ->'TorchLabelBinarizer':
        assert obj.y_type_ in ['multiclass', 'multilabel-indicator', 'binary']
        return cls(classes=torch.from_numpy(obj.classes_), pos_label=int(obj.pos_label), neg_label=int(obj.neg_label), y_type=obj.y_type_)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        if self.y_type == 'multilabel-indicator':
            pos_label = self.pos_label
            pos_switch = pos_label == 0
            if pos_switch:
                pos_label = -self.neg_label
            if pos_label != 1:
                x = torch.where(x != 0, torch.tensor(pos_label), x)
            x = x.long()
            if self.neg_label != 0:
                x = torch.where(x == 0, self.neg_label, x)
            if pos_switch:
                x = torch.where(x == pos_label, 0, x)
            return x
        elif self.y_type == 'multiclass':
            return torch.where(x[..., None] == self.classes, self.pos_label, self.neg_label).long()
        else:
            assert self.y_type == 'binary'
            return torch.where(x[..., None] == self.classes[1:], self.pos_label, self.neg_label).long()

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor, threshold: Optional[float]=None) ->torch.Tensor:
        if self.y_type == 'multiclass':
            return self.classes[x.argmax(-1)]
        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2
        outputs = self.classes[(x > threshold).long()]
        if self.y_type == 'binary':
            outputs = outputs.view(-1)
        return outputs


class TorchLinearModel(nn.Module):

    def __init__(self, model: LinearModel):
        super().__init__()
        if hasattr(model, 'densify'):
            model = deepcopy(model)
            model.densify()
        else:
            assert isinstance(model.coef_, np.ndarray), 'sparse linear model is not supported'
        weights = torch.from_numpy(model.coef_)
        biases = torch.from_numpy(np.array(model.intercept_))
        if len(weights.shape) == 1:
            weights = weights[None]
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _decision_function(self, x: torch.Tensor) ->torch.Tensor:
        outputs = x @ self.weights.t() + self.biases
        if outputs.shape[1] == 1:
            return outputs.view(-1)
        return outputs


class TorchLinearRegression(TorchLinearModel):

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [LinearRegression, Ridge, RidgeCV, SGDRegressor, LinearSVR]

    @classmethod
    def wrap(cls, obj: Union[LinearRegression, Ridge, RidgeCV, SGDRegressor, LinearSVR]) ->'TorchLinearRegression':
        return cls(obj)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Predict class labels for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        return self._decision_function(x)

    def _decision_function(self, x: torch.Tensor) ->torch.Tensor:
        outputs = x @ self.weights.t() + self.biases
        if outputs.shape[1] == 1:
            return outputs.view(-1)
        return outputs


class TorchLinearClassifier(TorchLinearModel):

    def __init__(self, model: LinearModel):
        super().__init__(model=model)
        self.register_buffer('classes', torch.from_numpy(model.classes_))

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [RidgeClassifier, RidgeClassifierCV, LinearSVC]

    @classmethod
    def wrap(cls, obj: Union[RidgeClassifier, RidgeClassifierCV, LinearSVC]) ->'TorchLinearClassifier':
        return cls(model=obj)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Predict class labels for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        scores = self.decision_function(x)
        if len(scores.shape) == 1:
            indices = (scores > 0).long()
        else:
            indices = scores.argmax(-1)
        return self.classes[indices]

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) ->torch.Tensor:
        return self._decision_function(x)


class TorchSGDClassifier(TorchLinearClassifier):

    def __init__(self, loss: str, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [SGDClassifier]

    @classmethod
    def wrap(cls, obj: SGDClassifier) ->'TorchSGDClassifier':
        return cls(loss=obj.loss, model=obj)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        if self.loss == 'log':
            logits = self.decision_function(x)
            if len(logits.shape) == 1:
                return torch.stack([F.logsigmoid(-logits), F.logsigmoid(logits)], dim=-1)
            return F.log_softmax(F.logsigmoid(logits), dim=-1)
        else:
            raise RuntimeError('probability prediction not supported for loss: ' + self.loss)


class TorchLogisticRegression(TorchLinearClassifier):

    def __init__(self, multi_class: str, **kwargs):
        super().__init__(**kwargs)
        self.multi_class = multi_class

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [LogisticRegression, LogisticRegressionCV]

    @classmethod
    def wrap(cls, obj: Union[LogisticRegression, LogisticRegressionCV]) ->'TorchLogisticRegression':
        multi_class = obj.multi_class
        if multi_class == 'auto' and (len(obj.classes_) == 2 or obj.solver == 'liblinear'):
            multi_class = 'ovr'
        return cls(multi_class=multi_class, model=obj)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        logits = self.decision_function(x)
        if self.multi_class == 'ovr':
            if len(logits.shape) == 1:
                return torch.stack([F.logsigmoid(-logits), F.logsigmoid(logits)], dim=-1)
            return F.log_softmax(F.logsigmoid(logits), dim=-1)
        else:
            if len(logits.shape) == 1:
                logits = torch.stack([-logits, logits], dim=-1)
            return F.log_softmax(logits, dim=-1)


class TorchMinMaxScaler(nn.Module):

    def __init__(self, scale: torch.Tensor, min_: torch.Tensor, feature_range: torch.Tensor, clip: bool):
        super().__init__()
        self.scale = nn.Parameter(scale)
        self.min = nn.Parameter(min_)
        self.register_buffer('feature_range', feature_range)
        self.clip = clip

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [MinMaxScaler]

    @classmethod
    def wrap(cls, obj: MinMaxScaler) ->'TorchMinMaxScaler':
        scale = torch.from_numpy(obj.scale_)
        min_ = torch.from_numpy(obj.min_)
        return TorchMinMaxScaler(scale, min_, torch.tensor(obj.feature_range), obj.clip)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        x = x * self.scale + self.min
        if self.clip:
            x = torch.minimum(torch.maximum(x, self.feature_range[0]), self.feature_range[1])
        return x

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) ->torch.Tensor:
        return (x - self.min) / self.scale


class _WrappedMLP(nn.Module):

    def __init__(self, layers: nn.Sequential, out_act: str):
        super().__init__()
        assert out_act in ['identity', 'softmax', 'logistic']
        self.layers = layers
        self.out_act = out_act

    @classmethod
    def wrap(cls, obj: Union[MLPClassifier, MLPRegressor]) ->'_WrappedMLP':
        acts = {'identity': nn.Identity, 'logistic': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU}
        if obj.activation not in acts:
            raise ValueError(f'unsupported activation: {obj.activation}')
        act = acts[obj.activation]
        modules = []
        for weights, biases in zip(obj.coefs_, obj.intercepts_):
            w = torch.from_numpy(weights)
            b = torch.from_numpy(biases)
            module = nn.Linear(weights.shape[0], weights.shape[1], dtype=w.dtype)
            with torch.no_grad():
                module.weight.copy_(w.t())
                module.bias.copy_(b)
            modules.append(module)
            modules.append(act())
        del modules[-1]
        return cls(layers=nn.Sequential(*modules), out_act=obj.out_activation_)

    def forward(self, x: torch.Tensor, include_negative: bool=False) ->torch.Tensor:
        x = self.layers(x)
        if x.shape[1] == 1 and self.out_act == 'logistic' and include_negative:
            x = torch.cat([F.logsigmoid(-x), F.logsigmoid(x)], dim=-1)
        elif self.out_act == 'logistic':
            x = F.logsigmoid(x)
        elif self.out_act == 'softmax':
            x = F.log_softmax(x, dim=-1)
        return x


class TorchMLPClassifier(nn.Module):

    def __init__(self, module: _WrappedMLP, label_binarizer: TorchLabelBinarizer):
        super().__init__()
        self.module = module
        self.label_binarizer = label_binarizer

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [MLPClassifier]

    @classmethod
    def wrap(cls, obj: MLPClassifier) ->'TorchMLPClassifier':
        return cls(module=_WrappedMLP.wrap(obj), label_binarizer=TorchLabelBinarizer.wrap(obj._label_binarizer))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        probs = self.module(x).exp()
        return self.label_binarizer.inverse_transform(probs)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict_log_proba(x).exp()

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.module(x, include_negative=True)


class TorchMLPRegressor(nn.Module):

    def __init__(self, module: _WrappedMLP):
        super().__init__()
        self.module = module

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [MLPRegressor]

    @classmethod
    def wrap(cls, obj: MLPRegressor) ->'TorchMLPRegressor':
        return cls(module=_WrappedMLP.wrap(obj))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        out = self.module(x)
        if out.shape[1] == 1:
            return out.view(-1)
        return out


class TorchNystroem(nn.Module):

    def __init__(self, kernel: Kernel, components: torch.Tensor, normalization: torch.Tensor):
        super().__init__()
        self.kernel = kernel
        self.components = nn.Parameter(components)
        self.normalization = nn.Parameter(normalization)

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [Nystroem]

    @classmethod
    def wrap(cls, obj: Nystroem) ->'TorchNystroem':
        kernel = Kernel.wrap(obj)
        return cls(kernel=kernel, components=torch.from_numpy(obj.components_), normalization=torch.from_numpy(obj.normalization_))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        return self.kernel(x, self.components) @ self.normalization.t()


class TorchPCA(nn.Module):

    def __init__(self, mean: torch.Tensor, components: torch.Tensor, scale: Optional[torch.Tensor]):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.components = nn.Parameter(components)
        if scale is not None:
            self.scale = nn.Parameter(scale)
        else:
            self.scale = 1.0

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [PCA]

    @classmethod
    def wrap(cls, obj: PCA) ->'TorchPCA':
        components = torch.from_numpy(obj.components_)
        mean = obj.mean_
        if mean is None:
            mean = torch.zeros(components.shape[1], device=components.device, dtype=components.dtype)
        else:
            mean = torch.from_numpy(mean)
        explained_variance = torch.from_numpy(obj.explained_variance_)
        return cls(mean=mean, components=components, scale=explained_variance.rsqrt() if obj.whiten else None)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        return (x - self.mean) @ self.components.t() * self.scale

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) ->torch.Tensor:
        return x / self.scale @ self.components + self.mean


def fill_unsupported(module: nn.Module, *names: str):
    """
    Fill unsupported method names on the module with a function that raises an
    exception when it is called. This can be used to appease TorchScript.
    """
    for name in names:
        if not hasattr(module, name):

            @torch.jit.export
            def unsupported_fn(self, _: torch.Tensor, unsup_method_name: str=name) ->torch.Tensor:
                raise RuntimeError(f'method {unsup_method_name} is not supported on this object')
            setattr(module, name, unsupported_fn)


class TorchPipeline(nn.Module):

    def __init__(self, stages: List[Tuple[str, nn.Module]]):
        super().__init__()
        self.transforms = nn.ModuleDict({k: v for k, v in stages[:-1]})
        for transform in self.transforms.values():
            fill_unsupported(transform, 'inverse_transform')
        self.final_name, self.final = stages[-1]
        fill_unsupported(self.final, 'decision_function', 'predict', 'predict_proba', 'predict_log_proba', 'transform', 'inverse_transform')

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [Pipeline]

    @classmethod
    def wrap(cls, obj: Pipeline) ->'TorchPipeline':
        mapping = []
        for k, v in obj.steps:
            mapping.append((k, wrap(v)))
        return cls(mapping)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call forward() on the final model.
        """
        return self.final(self._run_transforms(x))

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call predict() on the final model.
        """
        return self.final.predict(self._run_transforms(x))

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call decision_function() on the final model.
        """
        return self.final.decision_function(self._run_transforms(x))

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call predict_proba() on the final model.
        """
        return self.final.predict_proba(self._run_transforms(x))

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call predict_log_proba() on the final model.
        """
        return self.final.predict_log_proba(self._run_transforms(x))

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call transform() on the final model.
        """
        return self.final.transform(self._run_transforms(x))

    def _run_transforms(self, x: torch.Tensor) ->torch.Tensor:
        for transform in self.transforms.values():
            x = transform(x)
        return x

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) ->torch.Tensor:
        """
        Run the pipeline and call inverse_transform() on the final model.
        """
        x = self.final.inverse_transform(x)
        for transform in self.transforms.values()[::-1]:
            x = transform.inverse_transform(x)
        return x


class TorchStackingClassifier(nn.Module):

    def __init__(self, passthrough: bool, estimators: List[nn.Module], stack_methods: List[str], final_estimator: nn.Module, classes: torch.Tensor):
        super().__init__()
        self.passthrough = passthrough
        self.estimators = nn.ModuleList(estimators)
        for model in self.estimators:
            fill_unsupported(model, 'predict_proba', 'decision_function', 'predict')
        self.stack_methods = stack_methods
        self.final_estimator = final_estimator
        fill_unsupported(self.final_estimator, 'predict_proba', 'decision_function', 'predict')
        self.register_buffer('classes', classes)

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [StackingClassifier]

    @classmethod
    def wrap(cls, obj: StackingClassifier) ->'TorchStackingClassifier':
        return cls(passthrough=obj.passthrough, estimators=[wrap(x) for x in obj.estimators_], stack_methods=obj.stack_method_, final_estimator=wrap(obj.final_estimator_), classes=torch.from_numpy(obj.classes_))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        return self.classes[self.final_estimator.predict(self.transform(x))]

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.final_estimator.predict_proba(self.transform(x))

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) ->torch.Tensor:
        return self.final_estimator.decision_function(self.transform(x))

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        outputs = []
        for i, estimator in enumerate(self.estimators):
            method = self.stack_methods[i]
            if method == 'predict':
                out = estimator.predict(x)
            elif method == 'predict_proba':
                out = estimator.predict_proba(x)
                if out.shape[1] == 2:
                    out = out[:, 1:]
            else:
                assert method == 'decision_function'
                out = estimator.decision_function(x)
            if len(out.shape) == 1:
                out = out[:, None]
            outputs.append(out)
        if self.passthrough:
            outputs.append(x.view(len(x), -1))
        return torch.cat(outputs, dim=-1)


class TorchStackingRegressor(nn.Module):

    def __init__(self, passthrough: bool, estimators: List[nn.Module], final_estimator: nn.Module):
        super().__init__()
        self.passthrough = passthrough
        self.estimators = nn.ModuleList(estimators)
        self.final_estimator = final_estimator

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [StackingRegressor]

    @classmethod
    def wrap(cls, obj: StackingRegressor) ->'TorchStackingRegressor':
        return cls(passthrough=obj.passthrough, estimators=[wrap(x) for x in obj.estimators_], final_estimator=wrap(obj.final_estimator_))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        return self.final_estimator.predict(self.transform(x))

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        outputs = []
        for i, estimator in enumerate(self.estimators):
            outputs.append(estimator.predict(x).view(-1, 1))
        if self.passthrough:
            outputs.append(x.view(len(x), -1))
        return torch.cat(outputs, dim=-1)


class TorchStandardScaler(nn.Module):

    def __init__(self, mean: Optional[torch.Tensor], scale: Optional[torch.Tensor]):
        super().__init__()
        self.mean = nn.Parameter(mean if mean is not None else torch.zeros(()))
        self.scale = nn.Parameter(scale if scale is not None else torch.ones(()))

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [StandardScaler]

    @classmethod
    def wrap(cls, obj: StandardScaler) ->'TorchStandardScaler':
        return cls(mean=torch.from_numpy(obj.mean_) if obj.with_mean else None, scale=torch.from_numpy(obj.scale_) if obj.with_std else None)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.transform(x)

    @torch.jit.export
    def transform(self, x: torch.Tensor) ->torch.Tensor:
        return (x - self.mean) / self.scale

    @torch.jit.export
    def inverse_transform(self, x: torch.Tensor) ->torch.Tensor:
        return x * self.scale + self.mean


class TorchSVC(nn.Module):

    def __init__(self, kernel: Kernel, ovr: bool, break_ties: bool, n_support: List[int], support_vectors: torch.Tensor, intercept: torch.Tensor, dual_coef: torch.Tensor, classes: torch.Tensor, prob_a: Optional[torch.Tensor], prob_b: Optional[torch.Tensor]):
        super().__init__()
        self.kernel = kernel
        self.ovr = ovr
        self.break_ties = break_ties
        self.n_support = n_support
        self.support_vectors = nn.Parameter(support_vectors)
        self.intercept = nn.Parameter(intercept)
        self.dual_coef = nn.Parameter(dual_coef)
        self.register_buffer('classes', classes)
        self.supports_prob = prob_a is not None and prob_b is not None
        self.prob_a = nn.Parameter(torch.ones_like(intercept) if prob_a is None else prob_a)
        self.prob_b = nn.Parameter(torch.zeros_like(intercept) if prob_b is None else prob_b)
        self.n_classes = len(n_support)
        self.sv_offsets = []
        offset = 0
        for count in n_support:
            self.sv_offsets.append(offset)
            offset += count
        self._ovo_index_map = [(-1) for _ in range(self.n_classes ** 2)]
        k = 0
        for i in range(self.n_classes - 1):
            for j in range(i + 1, self.n_classes):
                self._ovo_index_map[i * self.n_classes + j] = k
                self._ovo_index_map[j * self.n_classes + i] = k
                k += 1

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [SVC, NuSVC]

    @classmethod
    def wrap(cls, obj: Union[SVC, NuSVC]) ->'TorchSVC':
        assert not obj._sparse, 'sparse SVC not supported'
        assert obj.decision_function_shape in ['ovo', 'ovr']
        return cls(kernel=Kernel.wrap(obj), ovr=obj.decision_function_shape == 'ovr', break_ties=obj.break_ties, n_support=obj.n_support_.tolist(), support_vectors=torch.from_numpy(obj.support_vectors_), intercept=torch.from_numpy(obj.intercept_), dual_coef=torch.from_numpy(obj.dual_coef_), classes=torch.from_numpy(obj.classes_), prob_a=torch.from_numpy(obj.probA_) if obj.probability else None, prob_b=torch.from_numpy(obj.probB_) if obj.probability else None)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        ovo, ovr = self.decision_function_ovo_ovr(x)
        if self.n_classes == 2:
            indices = (ovo.view(-1) > 0).long()
        elif self.ovr and self.break_ties:
            indices = ovr.argmax(dim=-1)
        else:
            indices = ovr.round().argmax(dim=-1)
        return self.classes[indices]

    @torch.jit.export
    def decision_function(self, x: torch.Tensor) ->torch.Tensor:
        ovo, ovr = self.decision_function_ovo_ovr(x)
        if len(self.n_support) == 2:
            return ovo.view(-1)
        elif self.ovr:
            return ovr
        else:
            return ovo

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        assert self.supports_prob, 'model must be trained with probability=True'
        if self.n_classes == 2:
            ovo, _ = self.decision_function_ovo_ovr(x)
            logit = ovo * self.prob_a - self.prob_b
            return torch.cat([F.logsigmoid(logit), F.logsigmoid(-logit)], dim=-1)
        return self.predict_proba(x).log()

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        assert self.supports_prob, 'model must be trained with probability=True'
        ovo, _ = self.decision_function_ovo_ovr(x)
        if self.n_classes == 2:
            logit = ovo * self.prob_a - self.prob_b
            return torch.cat([logit.sigmoid(), (-logit).sigmoid()], dim=-1)
        probs = (-ovo * self.prob_a - self.prob_b).sigmoid()
        min_prob = 1e-07
        probs = probs.clamp(min_prob, 1 - min_prob)
        matrix = self._prob_matrix(probs)
        inv_diag = 1 / (torch.diagonal(matrix, dim1=1, dim2=2) + 1e-12)
        guess = torch.ones(len(x), self.n_classes) / self.n_classes
        masks = torch.eye(self.n_classes)
        delta = torch.zeros_like(guess)
        for i in range(max(100, self.n_classes)):
            for coord in range(self.n_classes):
                mask = masks[coord]
                mg = (matrix @ guess[:, :, None]).view(guess.shape)
                outer = torch.einsum('ij,ij->i', guess, mg)[:, None]
                delta = outer - mg
                guess = guess + inv_diag * delta * mask
                guess = guess / guess.sum(dim=-1, keepdim=True)
            if delta.abs().max().item() < 0.005 / self.n_classes:
                break
        return guess

    def _prob_matrix(self, probs: torch.Tensor) ->torch.Tensor:
        """
        Compute the pairwise probability matrix Q from page 31 of
        https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf.
        """
        matrix_elems = []
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i == j:
                    prob_sum = torch.zeros_like(probs[:, 0])
                    for k in range(self.n_classes):
                        if k != i:
                            prob_sum = prob_sum + self._pairwise_prob(probs, k, i) ** 2
                    matrix_elems.append(prob_sum)
                else:
                    matrix_elems.append(-self._pairwise_prob(probs, i, j) * self._pairwise_prob(probs, j, i))
        return torch.stack(matrix_elems, dim=-1).view(len(probs), self.n_classes, self.n_classes)

    def _pairwise_prob(self, probs: torch.Tensor, i: int, j: int) ->torch.Tensor:
        p = probs[:, self._ovo_index_map[i * self.n_classes + j]]
        if i > j:
            return 1 - p
        return p

    def decision_function_ovo_ovr(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """Compute the one-versus-one and one-versus-rest decision functions."""
        kernel_out = self.kernel(x, self.support_vectors)
        votes = torch.zeros((len(x), self.n_classes))
        confidence_sum = [torch.zeros(len(x)) for i in range(self.n_classes)]
        ovos = []
        k = 0
        for i in range(self.n_classes - 1):
            for j in range(i + 1, self.n_classes):
                neg_confidence = self._dual_sum(kernel_out, i, j) + self._dual_sum(kernel_out, j, i) + self.intercept[k]
                k += 1
                ovos.append(neg_confidence)
                confidence_sum[i] = confidence_sum[i] + neg_confidence
                confidence_sum[j] = confidence_sum[j] - neg_confidence
                pred = neg_confidence < 0
                votes[:, i] += torch.logical_not(pred)
                votes[:, j] += pred
        ovo = torch.stack(ovos, dim=-1)
        confidences = torch.stack(confidence_sum, dim=-1)
        confidences = confidences / (3 * (confidences.abs() + 1))
        return ovo, votes + confidences

    def _dual_sum(self, kernel_out: torch.Tensor, i: int, j: int) ->torch.Tensor:
        assert j != i
        if j > i:
            j -= 1
        offset, count = self.sv_offsets[i], self.n_support[i]
        coeffs = self.dual_coef[j, offset:offset + count]
        kernel_row = kernel_out[:, offset:offset + count]
        return kernel_row @ coeffs


class TorchSVR(nn.Module):

    def __init__(self, kernel: Kernel, support_vectors: torch.Tensor, dual_coef: torch.Tensor, intercept: torch.Tensor):
        super().__init__()
        self.kernel = kernel
        self.support_vectors = nn.Parameter(support_vectors)
        self.dual_coef = nn.Parameter(dual_coef)
        self.intercept = nn.Parameter(intercept)

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [SVR, NuSVR]

    @classmethod
    def wrap(cls, obj: Union[SVR, NuSVR]) ->'TorchSVR':
        assert not obj._sparse, 'sparse SVR not supported'
        return cls(kernel=Kernel.wrap(obj), support_vectors=torch.from_numpy(obj.support_vectors_), dual_coef=torch.from_numpy(obj.dual_coef_), intercept=torch.from_numpy(obj.intercept_))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        """
        Predict regression values for the given feature vectors.

        An alias for self.predict().
        """
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        """
        Predict class labels for the given feature vectors.
        """
        kernel_out = self.kernel(x, self.support_vectors)
        return torch.einsum('jk,jk->j', self.dual_coef, kernel_out) + self.intercept


class BaseTree(nn.Module):

    def __init__(self, tree: Any):
        super().__init__()
        is_branch = np.zeros(len(tree.value), dtype=bool)
        node_conditions = np.zeros([len(tree.value)] * 2, dtype=bool)
        node_conditions_mask = node_conditions.copy()
        decision_paths = node_conditions.copy()

        def enumerate_tree(node_id: int, parent_id: Optional[int]=None):
            if parent_id is not None:
                node_conditions_mask[node_id] = node_conditions_mask[parent_id]
                node_conditions_mask[node_id][parent_id] = True
                decision_paths[node_id] = decision_paths[parent_id]
            decision_paths[node_id, node_id] = True
            left_id, right_id = tree.children_left[node_id], tree.children_right[node_id]
            if left_id != right_id:
                is_branch[node_id] = True
                node_conditions[left_id] = node_conditions[node_id]
                node_conditions[right_id] = node_conditions[node_id]
                node_conditions[right_id][node_id] = True
                enumerate_tree(left_id, node_id)
                enumerate_tree(right_id, node_id)
        enumerate_tree(0)
        max_depth = np.max(np.sum(node_conditions_mask.astype(np.int64), axis=-1))
        if max_depth < 2 ** 7:
            mat_dtype = torch.int8
        elif max_depth < 2 ** 15:
            mat_dtype = torch.int16
        else:
            mat_dtype = torch.int32
        is_leaf = np.logical_not(is_branch)
        self.register_buffer('feature', torch.from_numpy(tree.feature[is_branch]))
        self.value = nn.Parameter(torch.from_numpy(tree.value[is_leaf]))
        self.threshold = nn.Parameter(torch.from_numpy(tree.threshold[is_branch]))
        self.register_buffer('cond', torch.from_numpy(node_conditions[np.ix_(is_leaf, is_branch)]))
        self.register_buffer('cond_mask', torch.from_numpy(node_conditions_mask[np.ix_(is_leaf, is_branch)]))
        self.register_buffer('decision_paths', torch.from_numpy(decision_paths[is_leaf]))

    def _leaf_indices(self, x: torch.Tensor) ->torch.Tensor:
        comparisons = x[:, self.feature] > self.threshold
        cond_counts = comparisons @ self.cond.t()
        no_false_neg = cond_counts == self.cond.sum(1)
        no_false_pos = cond_counts == comparisons @ self.cond_mask.t()
        return (no_false_neg & no_false_pos).int().argmax(-1)

    @torch.jit.export
    def raw_values(self, x: torch.Tensor) ->torch.Tensor:
        return self.value[self._leaf_indices(x)]

    @torch.jit.export
    def decision_path(self, x: torch.Tensor):
        return self.decision_paths[self._leaf_indices(x)]


class TorchDecisionTreeRegressor(BaseTree):

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [DecisionTreeRegressor]

    @classmethod
    def wrap(cls, obj: DecisionTreeRegressor) ->'TorchDecisionTreeRegressor':
        return TorchDecisionTreeRegressor(obj.tree_)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        y = self.raw_values(x).view(len(x), -1)
        if y.shape[1] == 1:
            y = y.view(-1)
        return y


class _SingleClassOutput(nn.Module):

    def __init__(self, n_classes: int, classes: torch.Tensor):
        super().__init__()
        self.n_classes = int(n_classes)
        self.register_buffer('classes', classes)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.classes[x[:, :self.n_classes].argmax(1)]

    def predict_proba(self, x: torch.Tensor) ->torch.Tensor:
        proba = x[:, :self.n_classes]
        normalizer = proba.sum(1, keepdim=True)
        proba = proba / torch.where(normalizer == 0.0, 1.0, normalizer)
        return proba

    def predict_log_proba(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict_proba(x).log()


class TorchDecisionTreeClassifier(BaseTree):

    def __init__(self, n_outputs: int, n_classes: Union[int, List[int]], classes: Union[List[torch.Tensor], torch.Tensor], tree: Any):
        super().__init__(tree)
        self.n_outputs = int(n_outputs)
        if n_outputs == 1:
            self.outputs = nn.ModuleList([_SingleClassOutput(n_classes, classes)])
        else:
            self.outputs = nn.ModuleList([_SingleClassOutput(x, y) for x, y in zip(n_classes, classes)])

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [DecisionTreeClassifier]

    @classmethod
    def wrap(cls, obj: DecisionTreeClassifier) ->'TorchDecisionTreeClassifier':
        if obj.n_outputs_ == 1:
            classes = torch.from_numpy(obj.classes_)
        else:
            classes = [torch.from_numpy(x) for x in obj.classes_]
        return TorchDecisionTreeClassifier(obj.n_outputs_, obj.n_classes_, classes, obj.tree_)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        y = self.raw_values(x)
        outputs = []
        for i, x in enumerate(self.outputs):
            outputs.append(x(y[:, i]))
        if self.n_outputs == 1:
            return outputs[0]
        else:
            return torch.stack(outputs, dim=-1)

    @torch.jit.export
    def predict_proba(self, x: torch.Tensor) ->Union[torch.Tensor, List[torch.Tensor]]:
        y = self.raw_values(x)
        outputs = []
        for i, x in enumerate(self.outputs):
            outputs.append(x.predict_proba(y[:, i]))
        return self._collapse(outputs)

    @torch.jit.export
    def predict_log_proba(self, x: torch.Tensor) ->Union[torch.Tensor, List[torch.Tensor]]:
        y = self.raw_values(x)
        outputs = []
        for i, x in enumerate(self.outputs):
            outputs.append(x.predict_log_proba(y[:, i]))
        return self._collapse(outputs)

    def _collapse(self, x: List[torch.Tensor]) ->Union[torch.Tensor, List[torch.Tensor]]:
        if self.n_outputs == 1:
            return x[0]
        else:
            return x


class TorchTransformedTargetRegressor(nn.Module):

    def __init__(self, regressor: nn.Module, transformer: nn.Module, training_dim: int):
        super().__init__()
        self.regressor = regressor
        self.transformer = transformer
        self.training_dim = training_dim

    @classmethod
    def supported_classes(cls) ->List[Type]:
        return [TransformedTargetRegressor]

    @classmethod
    def wrap(cls, obj: TransformedTargetRegressor) ->'TorchTransformedTargetRegressor':
        assert obj.transformer_ is not None, 'identity and function transformers not supported'
        return TorchTransformedTargetRegressor(regressor=wrap(obj.regressor_), transformer=wrap(obj.transformer_), training_dim=obj._training_dim)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.predict(x)

    @torch.jit.export
    def predict(self, x: torch.Tensor) ->torch.Tensor:
        y = self.regressor(x)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        y = self.transformer.inverse_transform(y)
        if self.training_dim == 1 and len(y.shape) == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (TorchMLPRegressor,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_WrappedMLP,
     lambda: ([], {'layers': _mock_layer(), 'out_act': 'identity'}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_unixpickle_sk2torch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

