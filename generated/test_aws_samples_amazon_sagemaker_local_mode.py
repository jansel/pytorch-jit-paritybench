import sys
_module = sys.modules[__name__]
del sys
cifar10 = _module
utils_cifar = _module
catboost_bring_your_own_container_local_training_and_serving = _module
predictor = _module
wsgi = _module
catboost_scikit_learn_script_mode_local_training_and_serving = _module
catboost_train_deploy = _module
bootstrap = _module
dask_bring_your_own_container_local_processing = _module
processing_script = _module
deep_java_library_bring_your_own_container_serving_local_mode = _module
delta_lake_bring_your_own_container_local_training_and_serving = _module
delta_sharing_bring_your_own_container_local_processing = _module
scikit_boston_housing = _module
delta_sharing_scikit_learn_local_training_and_serving = _module
inference = _module
gensim_with_word2vec_model_artifacts_local_serving = _module
train = _module
hdbscan_bring_your_own_container_local_training = _module
hebert_model = _module
huggingface_hebert_sentiment_analysis_local_serving = _module
lightgbm_bring_your_own_container_local_training_and_serving = _module
prophet_bring_your_own_container_local_training_and_serving = _module
pytorch_extend_dlc_container_ofa_local_serving = _module
pytorch_graviton_script_mode_local_model_inference = _module
pytorch_nlp_script_mode_local_model_inference = _module
pytorch_script_mode_local_model_inference = _module
utils_cifar = _module
cifar10_pytorch = _module
pytorch_script_mode_local_training_and_serving = _module
utils_cifar = _module
mnist = _module
pytorch_wandb_script_mode_local_training = _module
inference = _module
pytorch_yolov5_local_model_inference = _module
scikit_learn_bring_your_own_container_and_own_model_local_serving = _module
scikit_learn_bring_your_own_container_local_processing = _module
scikit_learn_bring_your_own_model_local_serving = _module
scikit_learn_graviton_bring_your_own_container_local_training_and_serving = _module
SKLearnProcessor_local_processing = _module
FrameworkProcessor_nltk_local_processing = _module
scikit_learn_script_mode_local_serving_multiple_models_with_one_invocation = _module
scikit_learn_script_mode_local_serving_no_model_artifact = _module
scikit_learn_california = _module
scikit_learn_script_mode_local_training_and_serving = _module
california_housing = _module
tensorflow_bring_your_own_california_housing_local_serving_without_tfs = _module
tensorflow_bring_your_own_california_housing_local_training_and_batch_transform = _module
tensorflow_bring_your_own_california_housing_local_training_and_serving = _module
california_housing_tf2 = _module
tensorflow_bring_your_own_california_housing_local_training_toolkit = _module
model_handler = _module
tensorflow_bring_your_own_california_housing_mms_local_serving = _module
tensorflow_extend_dlc_california_housing_local_training = _module
tensorflow_graviton_bring_your_own_california_housing_local_training = _module
tensorflow_graviton_bring_your_own_california_housing_local_training_toolkit = _module
tensorflow_graviton_script_mode_local_model_inference = _module
tensorflow_script_mode_california_housing_local_training_and_batch_transform = _module
tensorflow_script_mode_california_housing_local_training_and_serving = _module
mnist_tf2 = _module
tensorflow_script_mode_debug_local_training = _module
cifar10_tf2 = _module
tensorflow_script_mode_local_training_resnet50 = _module
tensorflow_script_mode_local_model_inference = _module
tensorflow_script_mode_local_model_inference_file = _module
tensorflow_script_mode_local_training_and_serving = _module
abalone = _module
xgboost_script_mode_local_training_and_serving = _module

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


import logging


import torch


import torch.distributed as dist


import torch.nn as nn


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision


import torchvision.models


import torchvision.transforms as transforms


import torch.nn.functional as F


import matplotlib.pyplot as plt


import numpy as np


import pandas as pd


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


from torchvision.datasets import MNIST


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

