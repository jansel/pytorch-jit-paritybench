import sys
_module = sys.modules[__name__]
del sys
master = _module
setup = _module
test_base = _module
test_transforms = _module
ttach = _module
__version__ = _module
aliases = _module
base = _module
functional = _module
transforms = _module
wrappers = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch.nn.functional as F


from functools import partial


from typing import Optional


from typing import List


from typing import Union


from typing import Tuple


import torch.nn as nn


from typing import Mapping


class BaseTransform:
    identity_param = None

    def __init__(self, name: str, params: Union[list, tuple]):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError

    def apply_deaug_label(self, label, *args, **params):
        raise NotImplementedError


class Chain:

    def __init__(self, functions: List[callable]):
        self.functions = functions or []

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x


class Transformer:

    def __init__(self, image_pipeline: Chain, mask_pipeline: Chain, label_pipeline: Chain):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
        self.label_pipeline = label_pipeline

    def augment_image(self, image):
        return self.image_pipeline(image)

    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

    def deaugment_label(self, label):
        return self.label_pipeline(label)


class Compose:

    def __init__(self, transforms: List[BaseTransform]):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self) ->Transformer:
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            image_aug_chain = Chain([partial(t.apply_aug_image, **{t.pname: p}) for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = Chain([partial(t.apply_deaug_mask, **{t.pname: p}) for t, p in zip(self.deaug_transforms, deaug_params)])
            label_deaug_chain = Chain([partial(t.apply_deaug_label, **{t.pname: p}) for t, p in zip(self.deaug_transforms, deaug_params)])
            yield Transformer(image_pipeline=image_aug_chain, mask_pipeline=mask_deaug_chain, label_pipeline=label_deaug_chain)

    def __len__(self) ->int:
        return len(self.aug_transform_parameters)


class Merger:

    def __init__(self, type: str='mean', n: int=1):
        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))
        self.output = None
        self.type = type
        self.n = n

    def append(self, x):
        if self.type == 'tsharpen':
            x = x ** 0.5
        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = F.max(self.output, x)
        elif self.type == 'min':
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result


class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(self, model: nn.Module, transforms: Compose, merge_mode: str='mean', output_mask_key: Optional[str]=None):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(self, image: torch.Tensor, *args) ->Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)
        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}
        return result


class ClassificationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(self, model: nn.Module, transforms: Compose, merge_mode: str='mean', output_label_key: Optional[str]=None):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key

    def forward(self, image: torch.Tensor, *args) ->Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)
        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}
        return result

