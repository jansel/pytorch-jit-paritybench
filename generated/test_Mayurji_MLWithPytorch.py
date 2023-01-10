import sys
_module = sys.modules[__name__]
del sys
LinearRegression = _module
LogisticRegression = _module
DecisionTree = _module
KMeans = _module
NaiveBayes = _module
KNN = _module
svm = _module
tfidf = _module
pca = _module
Lasso_Ridge_Regression = _module
gmm = _module
NaiveBayes = _module
lda = _module
adaboost = _module
dbscan = _module
BayesianRegression = _module
PAM = _module
utility = _module
tsne = _module
ElasticNetRegression = _module
spectralClustering = _module
LDA_TopicModeling = _module
AffinityPropagation = _module
gd = _module
regularization = _module
ransac = _module
normalization = _module
mlp = _module
MLP = _module
activation = _module
optimizer = _module
loss = _module

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


from sklearn.datasets import make_blobs


from sklearn.datasets import load_breast_cancer


from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score


import scipy


import numpy as np


from sklearn.datasets import load_iris


from scipy.stats import mode


from sklearn.preprocessing import MinMaxScaler


from sklearn.utils import shuffle


import matplotlib.pyplot as plt


from sklearn.datasets import load_boston


from itertools import combinations_with_replacement


import math


from sklearn import datasets


import pandas as pd


from scipy.stats import chi2


from scipy.stats import multivariate_normal


import logging


from sklearn.datasets import load_digits


from sklearn.datasets import load_diabetes


from sklearn.datasets import make_moons


from scipy.spatial.distance import pdist


from scipy.spatial.distance import squareform


from sklearn.cluster import KMeans


from sklearn.datasets import fetch_20newsgroups


from sklearn.feature_extraction.text import CountVectorizer


from torch import nn


import random


from sklearn.datasets import make_regression


from sklearn.neighbors import KNeighborsClassifier

