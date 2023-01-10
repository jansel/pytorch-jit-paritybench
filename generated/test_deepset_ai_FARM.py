import sys
_module = sys.modules[__name__]
del sys
conf = _module
qa_formats = _module
conversion_huggingface_models = _module
conversion_huggingface_models_classification = _module
doc_classification = _module
doc_classification_cola = _module
doc_classification_crossvalidation = _module
doc_classification_custom_optimizer = _module
doc_classification_fasttext_LM = _module
doc_classification_holdout = _module
doc_classification_multilabel = _module
doc_classification_multilabel_roberta = _module
doc_classification_with_earlystopping = _module
doc_classification_word_embedding_LM = _module
doc_regression = _module
dpr_encoder = _module
embeddings_extraction = _module
embeddings_extraction_s3e_pooling = _module
evaluation = _module
lm_finetuning = _module
mtl01_tclass_tclass = _module
natural_questions = _module
ner = _module
onnx_question_answering = _module
passage_ranking = _module
question_answering = _module
question_answering_confidence = _module
question_answering_crossvalidation = _module
streaming_inference = _module
text_pair_classification = _module
train_from_scratch = _module
train_from_scratch_with_sagemaker = _module
wordembedding_inference = _module
farm = _module
_version = _module
conversion = _module
convert_tf_checkpoint_to_pytorch = _module
transformers = _module
data_handler = _module
data_silo = _module
dataloader = _module
dataset = _module
input_features = _module
inputs = _module
nq_utils = _module
processor = _module
samples = _module
utils = _module
eval = _module
metrics = _module
msmarco_passage_farm = _module
msmarco_passage_official = _module
semantic_answer_similarity_evaluation = _module
squad_evaluation = _module
experiment = _module
file_utils = _module
infer = _module
inference_rest_api = _module
modeling = _module
adaptive_model = _module
biadaptive_model = _module
language_model = _module
optimization = _module
prediction_head = _module
predictions = _module
tokenization = _module
wordembedding_utils = _module
train = _module
utils = _module
visual = _module
ascii = _module
images = _module
text = _module
run_all_experiments = _module
setup = _module
conftest = _module
convert_result_to_csv = _module
question_answering = _module
question_answering_accuracy = _module
question_answering_components = _module
create_testdata = _module
test_optimization = _module
test_conversion = _module
test_data_silo = _module
test_doc_classification_distilbert = _module
test_doc_regression = _module
test_dpr = _module
test_evaluation_metrics = _module
test_inference = _module
test_lm_finetuning = _module
test_model_versioning = _module
test_natural_questions = _module
test_ner = _module
test_ner_amp = _module
test_onnx_conversion = _module
test_prediction_head = _module
test_processor_qa = _module
test_processor_saving_loading = _module
test_question_answering = _module
test_s3e_pooling = _module
test_text_pair = _module
test_tokenization = _module

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


import numbers


from collections import defaultdict


import torch


from sklearn.metrics import matthews_corrcoef


from sklearn.metrics import f1_score


import torch.multiprocessing as mp


import copy


from functools import partial


import random


from itertools import chain


from itertools import groupby


import numpy as np


from sklearn.utils.class_weight import compute_class_weight


from torch.utils.data import Dataset


from torch.utils.data import Subset


from torch.utils.data import IterableDataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import KFold


from sklearn.model_selection import ShuffleSplit


from sklearn.model_selection import StratifiedShuffleSplit


from math import ceil


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from typing import Iterable


from torch.utils.data import ConcatDataset


from torch.utils.data import TensorDataset


import abc


import inspect


from abc import ABC


from inspect import signature


from random import randint


from numpy.random import random as random_float


from sklearn.preprocessing import StandardScaler


from functools import reduce


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn.metrics import mean_squared_error


from sklearn.metrics import r2_score


from sklearn.metrics import classification_report


from functools import wraps


import warnings


from typing import Generator


from typing import List


from typing import Union


import numpy


from torch import nn


from collections import OrderedDict


from torch.nn.parallel import DistributedDataParallel


from torch.nn import DataParallel


from typing import Tuple


from torch import optim


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


from torch.nn import NLLLoss


import torch.distributed as dist


from torch import multiprocessing as mp


from copy import deepcopy


import pandas as pd


import time


from torch.utils.data import SequentialSampler


logger = logging.getLogger(__name__)


def pick_single_fn(heads, fn_name):
    """ Iterates over heads and returns a static method called fn_name
    if and only if one head has a method of that name. If no heads have such a method, None is returned.
    If more than one head has such a method, an Exception is thrown"""
    merge_fns = []
    for h in heads:
        merge_fns.append(getattr(h, fn_name, None))
    merge_fns = [x for x in merge_fns if x is not None]
    if len(merge_fns) == 0:
        return None
    elif len(merge_fns) == 1:
        return merge_fns[0]
    else:
        raise Exception(f'More than one of the prediction heads have a {fn_name}() function')


def stack(list_of_lists):
    n_lists_final = len(list_of_lists[0])
    ret = [list() for _ in range(n_lists_final)]
    for l in list_of_lists:
        for i, x in enumerate(l):
            ret[i] += x
    return ret


class BaseAdaptiveModel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific AdaptiveModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: arguments to pass for loading the model.
        :return: instance of a model
        """
        if (Path(kwargs['load_dir']) / 'model.onnx').is_file():
            model = cls.subclasses['ONNXAdaptiveModel'].load(**kwargs)
        else:
            model = cls.subclasses['AdaptiveModel'].load(**kwargs)
        return model

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all predictions from all prediction heads
        """
        all_preds = []
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits, **kwargs):
        """
        Format predictions for inference.

        :param logits: model logits
        :type logits: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        n_heads = len(self.prediction_heads)
        if n_heads == 0:
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)
        elif n_heads == 1:
            preds_final = []
            try:
                preds = kwargs['preds']
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs['preds'] = preds_flat
            except KeyError:
                kwargs['preds'] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and 'predictions' in preds:
                preds_final.append(preds)
        else:
            preds_final = [list() for _ in range(n_heads)]
            preds = kwargs.get('preds')
            if preds is not None:
                preds_for_heads = stack(preds)
                logits_for_heads = [None] * n_heads
                del kwargs['preds']
            else:
                preds_for_heads = [None] * n_heads
                logits_for_heads = logits
            preds_final = [list() for _ in range(n_heads)]
            if not 'samples' in kwargs:
                samples = [s for b in kwargs['baskets'] for s in b.samples]
                kwargs['samples'] = samples
            for i, (head, preds_for_head, logits_for_head) in enumerate(zip(self.prediction_heads, preds_for_heads, logits_for_heads)):
                preds = head.formatted_preds(logits=logits_for_head, preds=preds_for_head, **kwargs)
                preds_final[i].append(preds)
            merge_fn = pick_single_fn(self.prediction_heads, 'merge_formatted_preds')
            if merge_fn:
                preds_final = merge_fn(preds_final)
        return preds_final

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """
        if 'nextsentence' not in tasks:
            idx = None
            for i, ph in enumerate(self.prediction_heads):
                if ph.task_name == 'nextsentence':
                    idx = i
            if idx is not None:
                logger.info('Removing the NextSentenceHead since next_sent_pred is set to False in the BertStyleLMProcessor')
                del self.prediction_heads[i]
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]['label_tensor_name']
            label_list = tasks[head.task_name]['label_list']
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]['label_list']
            head.label_list = label_list
            if 'RegressionHead' in str(type(head)):
                num_labels = 1
            else:
                num_labels = len(label_list)
            head.metric = tasks[head.task_name]['metric']

    @classmethod
    def _get_prediction_head_files(cls, load_dir, strict=True):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [(load_dir / f) for f in files if '.bin' in f and 'prediction_head' in f]
        config_files = [(load_dir / f) for f in files if 'config.json' in f and 'prediction_head' in f]
        model_files.sort()
        config_files.sort()
        if strict:
            error_str = f'There is a mismatch in number of model files ({len(model_files)}) and config files ({len(config_files)}).This might be because the Language Model Prediction Head does not currently support saving and loading'
            assert len(model_files) == len(config_files), error_str
        logger.info(f'Found files for loading {len(model_files)} prediction heads')
        return model_files, config_files


OUTPUT_DIM_NAMES = ['dim', 'hidden_size', 'd_model']


def s3e_pooling(token_embs, token_ids, token_weights, centroids, token_to_cluster, mask, svd_components=None):
    """
    Pooling of word/token embeddings as described by Wang et al in their paper
    "Efficient Sentence Embedding via Semantic Subspace Analysis"
    (https://arxiv.org/abs/2002.09620)
    Adjusted their implementation from here: https://github.com/BinWang28/Sentence-Embedding-S3E

    This method takes a fitted "s3e model" and token embeddings from a language model and returns sentence embeddings
    using the S3E Method. The model can be fitted via `fit_s3e_on_corpus()`.

    Usage: See `examples/embeddings_extraction_s3e_pooling.py`

    :param token_embs: numpy array of shape (batch_size, max_seq_len, emb_dim) containing the embeddings for each token
    :param token_ids: numpy array of shape (batch_size, max_seq_len) containing the ids for each token in the vocab
    :param token_weights: dict with key=token_id, value= weight in corpus
    :param centroids: numpy array of shape (n_cluster, emb_dim) that describes the centroids of our clusters in the embedding space
    :param token_to_cluster: numpy array of shape (vocab_size, 1) where token_to_cluster[i] = cluster_id that token with id i belongs to
    :param svd_components: Components from a truncated singular value decomposition (SVD, aka LSA) to be
                           removed from the final sentence embeddings in a postprocessing step.
                           SVD must be fit on representative sample of sentence embeddings first and can
                           then be removed from all subsequent embeddings in this function.
                           We expect the sklearn.decomposition.TruncatedSVD.fit(<your_embeddings>)._components to be passed here.
    :return: embeddings matrix of shape (batch_size, emb_dim + (n_clusters*n_clusters+1)/2)
    """
    embeddings = []
    n_clusters = centroids.shape[0]
    emb_dim = token_embs.shape[2]
    n_samples = token_embs.shape[0]
    token_ids[mask] = -1
    for sample_idx in range(n_samples):
        stage_vec = [{}]
        for tok_idx, tok_id in enumerate(token_ids[sample_idx, :]):
            if tok_id != -1:
                stage_vec[-1][tok_id] = token_embs[sample_idx, tok_idx]
        stage_vec.append({})
        for k, v in stage_vec[-2].items():
            cluster = token_to_cluster[k]
            if cluster in stage_vec[-1]:
                stage_vec[-1][cluster].append(stage_vec[-2][k] * token_weights[k])
            else:
                stage_vec[-1][cluster] = []
                stage_vec[-1][cluster].append(stage_vec[-2][k] * token_weights[k])
        for k, v in stage_vec[-1].items():
            centroid_vec = centroids[k]
            v = [(wv - centroid_vec) for wv in v]
            stage_vec[-1][k] = np.sum(v, 0)
        sentvec = []
        vec = np.zeros(emb_dim)
        for key, value in stage_vec[0].items():
            vec = vec + value * token_weights[key]
        sentvec.append(vec / len(stage_vec[0].keys()))
        matrix = np.zeros((n_clusters, emb_dim))
        for j in range(n_clusters):
            if j in stage_vec[-1]:
                matrix[j, :] = stage_vec[-1][j]
        matrix_no_mean = matrix - matrix.mean(1)[:, np.newaxis]
        cov = matrix_no_mean.dot(matrix_no_mean.T)
        iu1 = np.triu_indices(cov.shape[0])
        iu2 = np.triu_indices(cov.shape[0], 1)
        cov[iu2] = cov[iu2] * np.sqrt(2)
        vec = cov[iu1]
        vec = vec / np.linalg.norm(vec)
        sentvec.append(vec)
        sentvec = np.concatenate(sentvec)
        embeddings.append(sentvec)
    embeddings = np.vstack(embeddings)
    if svd_components is not None:
        embeddings = embeddings - embeddings.dot(svd_components.transpose()) * svd_components
    return embeddings


class LanguageModel(nn.Module):
    """
    The parent class for any kind of model that can embed language into a semantic vector space. Practically
    speaking, these models read in tokenized sentences and return vectors that capture the meaning of sentences
    or of tokens.
    """
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() or all specific LanguageModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def forward(self, input_ids, padding_mask, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_scratch(cls, model_type, vocab_size):
        if model_type.lower() == 'bert':
            model = Bert
        return model.from_scratch(vocab_size)

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, n_added_tokens=0, language_model_class=None, **kwargs):
        """
        Load a pretrained language model either by

        1. specifying its name and downloading it
        2. or pointing to the directory it is saved in.

        Available remote models:

        * bert-base-uncased
        * bert-large-uncased
        * bert-base-cased
        * bert-large-cased
        * bert-base-multilingual-uncased
        * bert-base-multilingual-cased
        * bert-base-chinese
        * bert-base-german-cased
        * roberta-base
        * roberta-large
        * xlnet-base-cased
        * xlnet-large-cased
        * xlm-roberta-base
        * xlm-roberta-large
        * albert-base-v2
        * albert-large-v2
        * distilbert-base-german-cased
        * distilbert-base-multilingual-cased
        * google/electra-small-discriminator
        * google/electra-base-discriminator
        * google/electra-large-discriminator
        * facebook/dpr-question_encoder-single-nq-base
        * facebook/dpr-ctx_encoder-single-nq-base

        See all supported model variations here: https://huggingface.co/models

        The appropriate language model class is inferred automatically from model config
        or can be manually supplied via `language_model_class`.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param language_model_class: (Optional) Name of the language model class to load (e.g. `Bert`)
        :type language_model_class: str

        """
        kwargs['revision'] = revision
        logger.info('')
        logger.info('LOADING MODEL')
        logger.info('=============')
        config_file = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(config_file):
            logger.info(f'Model found locally at {pretrained_model_name_or_path}')
            config = json.load(open(config_file))
            language_model = cls.subclasses[config['name']].load(pretrained_model_name_or_path)
        else:
            logger.info(f'Could not find {pretrained_model_name_or_path} locally.')
            logger.info(f'Looking on Transformers Model Hub (in local cache and online)...')
            if language_model_class is None:
                language_model_class = cls.get_language_model_class(pretrained_model_name_or_path)
            if language_model_class:
                language_model = cls.subclasses[language_model_class].load(pretrained_model_name_or_path, **kwargs)
            else:
                language_model = None
        if not language_model:
            raise Exception(f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. Ensure that the model class name can be inferred from the directory name when loading a Transformers' model. Here's a list of available models: https://farm.deepset.ai/api/modeling.html#farm.modeling.language_model.LanguageModel.load")
        else:
            logger.info(f'Loaded {pretrained_model_name_or_path}')
        if n_added_tokens != 0:
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(f'Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab.')
            language_model.model.resize_token_embeddings(vocab_size)
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size
        return language_model

    @staticmethod
    def get_language_model_class(model_name_or_path, **kwargs):
        model_name_or_path = str(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        model_type = config.model_type
        if model_type == 'xlm-roberta':
            language_model_class = 'XLMRoberta'
        elif model_type == 'roberta':
            if 'mlm' in model_name_or_path.lower():
                raise NotImplementedError('MLM part of codebert is currently not supported in FARM')
            language_model_class = 'Roberta'
        elif model_type == 'camembert':
            language_model_class = 'Camembert'
        elif model_type == 'albert':
            language_model_class = 'Albert'
        elif model_type == 'distilbert':
            language_model_class = 'DistilBert'
        elif model_type == 'bert':
            language_model_class = 'Bert'
        elif model_type == 'xlnet':
            language_model_class = 'XLNet'
        elif model_type == 'electra':
            language_model_class = 'Electra'
        elif model_type == 'dpr':
            if config.architectures[0] == 'DPRQuestionEncoder':
                language_model_class = 'DPRQuestionEncoder'
            elif config.architectures[0] == 'DPRContextEncoder':
                language_model_class = 'DPRContextEncoder'
            elif config.archictectures[0] == 'DPRReader':
                raise NotImplementedError('DPRReader models are currently not supported.')
        elif model_type == 'big_bird':
            language_model_class = 'BigBird'
        else:
            logger.warning('Could not infer LanguageModel class from config. Trying to infer LanguageModel class from model name.')
            language_model_class = LanguageModel._infer_language_model_class_from_string(model_name_or_path)
        return language_model_class

    @staticmethod
    def _infer_language_model_class_from_string(model_name_or_path):
        if 'xlm' in model_name_or_path.lower() and 'roberta' in model_name_or_path.lower():
            language_model_class = 'XLMRoberta'
        elif 'bigbird' in model_name_or_path.lower():
            language_model_class = 'BigBird'
        elif 'roberta' in model_name_or_path.lower():
            language_model_class = 'Roberta'
        elif 'codebert' in model_name_or_path.lower():
            if 'mlm' in model_name_or_path.lower():
                raise NotImplementedError('MLM part of codebert is currently not supported in FARM')
            else:
                language_model_class = 'Roberta'
        elif 'camembert' in model_name_or_path.lower() or 'umberto' in model_name_or_path.lower():
            language_model_class = 'Camembert'
        elif 'albert' in model_name_or_path.lower():
            language_model_class = 'Albert'
        elif 'distilbert' in model_name_or_path.lower():
            language_model_class = 'DistilBert'
        elif 'bert' in model_name_or_path.lower():
            language_model_class = 'Bert'
        elif 'xlnet' in model_name_or_path.lower():
            language_model_class = 'XLNet'
        elif 'electra' in model_name_or_path.lower():
            language_model_class = 'Electra'
        elif 'word2vec' in model_name_or_path.lower() or 'glove' in model_name_or_path.lower():
            language_model_class = 'WordEmbedding_LM'
        elif 'minilm' in model_name_or_path.lower():
            language_model_class = 'Bert'
        elif 'dpr-question_encoder' in model_name_or_path.lower():
            language_model_class = 'DPRQuestionEncoder'
        elif 'dpr-ctx_encoder' in model_name_or_path.lower():
            language_model_class = 'DPRContextEncoder'
        else:
            language_model_class = None
        return language_model_class

    def get_output_dims(self):
        config = self.model.config
        for odn in OUTPUT_DIM_NAMES:
            if odn in dir(config):
                return getattr(config, odn)
        else:
            raise Exception('Could not infer the output dimensions of the language model')

    def freeze(self, layers):
        """ To be implemented"""
        raise NotImplementedError()

    def unfreeze(self):
        """ To be implemented"""
        raise NotImplementedError()

    def save_config(self, save_dir):
        save_filename = Path(save_dir) / 'language_model_config.json'
        with open(save_filename, 'w') as file:
            setattr(self.model.config, 'name', self.__class__.__name__)
            setattr(self.model.config, 'language', self.language)
            if self.__class__.__name__ == 'DPRQuestionEncoder' or self.__class__.__name__ == 'DPRContextEncoder':
                setattr(transformers.DPRConfig, 'model_type', self.model.config.model_type)
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir, state_dict=None):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        :param state_dict: A dictionary containing a whole state of the module including names of layers. By default, the unchanged state dict of the module is used
        :type state_dict: dict
        """
        save_name = Path(save_dir) / 'language_model.bin'
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if not state_dict:
            state_dict = model_to_save.state_dict()
        torch.save(state_dict, save_name)
        self.save_config(save_dir)

    @classmethod
    def _get_or_infer_language_from_name(cls, language, name):
        if language is not None:
            return language
        else:
            return cls._infer_language_from_name(name)

    @classmethod
    def _infer_language_from_name(cls, name):
        known_languages = 'german', 'english', 'chinese', 'indian', 'french', 'polish', 'spanish', 'multilingual'
        matches = [lang for lang in known_languages if lang in name]
        if 'camembert' in name:
            language = 'french'
            logger.info(f'Automatically detected language from language model name: {language}')
        elif 'umberto' in name:
            language = 'italian'
            logger.info(f'Automatically detected language from language model name: {language}')
        elif len(matches) == 0:
            language = 'english'
        elif len(matches) > 1:
            language = matches[0]
        else:
            language = matches[0]
            logger.info(f'Automatically detected language from language model name: {language}')
        return language

    def formatted_preds(self, logits, samples, ignore_first_token=True, padding_mask=None, input_ids=None, **kwargs):
        """
        Extracting vectors from language model (e.g. for extracting sentence embeddings).
        Different pooling strategies and layers are available and will be determined from the object attributes
        `extraction_layer` and `extraction_strategy`. Both should be set via the Inferencer:
        Example:  Inferencer(extraction_strategy='cls_token', extraction_layer=-1)

        :param logits: Tuple of (sequence_output, pooled_output) from the language model.
                       Sequence_output: one vector per token, pooled_output: one vector for whole sequence
        :param samples: For each item in logits we need additional meta information to format the prediction (e.g. input text).
                        This is created by the Processor and passed in here from the Inferencer.
        :param ignore_first_token: Whether to include the first token for pooling operations (e.g. reduce_mean).
                                   Many models have here a special token like [CLS] that you don't want to include into your average of token embeddings.
        :param padding_mask: Mask for the padding tokens. Those will also not be included in the pooling operations to prevent a bias by the number of padding tokens.
        :param input_ids: ids of the tokens in the vocab
        :param kwargs: kwargs
        :return: list of dicts containing preds, e.g. [{"context": "some text", "vec": [-0.01, 0.5 ...]}]
        """
        if not hasattr(self, 'extraction_layer') or not hasattr(self, 'extraction_strategy'):
            raise ValueError("`extraction_layer` or `extraction_strategy` not specified for LM. Make sure to set both, e.g. via Inferencer(extraction_strategy='cls_token', extraction_layer=-1)`")
        sequence_output = logits[0][0]
        pooled_output = logits[0][1]
        if self.extraction_strategy == 'pooled':
            if self.extraction_layer != -1:
                raise ValueError(f'Pooled output only works for the last layer, but got extraction_layer = {self.extraction_layer}. Please set `extraction_layer=-1`.)')
            vecs = pooled_output.cpu().numpy()
        elif self.extraction_strategy == 'per_token':
            vecs = sequence_output.cpu().numpy()
        elif self.extraction_strategy == 'reduce_mean':
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == 'reduce_max':
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token)
        elif self.extraction_strategy == 'cls_token':
            vecs = sequence_output[:, 0, :].cpu().numpy()
        elif self.extraction_strategy == 's3e':
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token, input_ids=input_ids, s3e_stats=self.s3e_stats)
        else:
            raise NotImplementedError
        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred['context'] = sample.clear_text['text']
            pred['vec'] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output, padding_mask, strategy, ignore_first_token, input_ids=None, s3e_stats=None):
        token_vecs = sequence_output.cpu().numpy()
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        if ignore_first_token:
            ignore_mask_2d[:, 0] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, np.newaxis]
        if strategy == 'reduce_max':
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).max(axis=1).data
        if strategy == 'reduce_mean':
            pooled_vecs = np.ma.array(data=token_vecs, mask=ignore_mask_3d).mean(axis=1).data
        if strategy == 's3e':
            input_ids = input_ids.cpu().numpy()
            pooled_vecs = s3e_pooling(token_embs=token_vecs, token_ids=input_ids, token_weights=s3e_stats['token_weights'], centroids=s3e_stats['centroids'], token_to_cluster=s3e_stats['token_to_cluster'], svd_components=s3e_stats.get('svd_components', None), mask=padding_mask == 0)
        return pooled_vecs


class FeedForwardBlock(nn.Module):
    """ A feed forward neural network of variable depth and width. """

    def __init__(self, layer_dims, **kwargs):
        super(FeedForwardBlock, self).__init__()
        self.layer_dims = layer_dims
        n_layers = len(layer_dims) - 1
        layers_all = []
        self.output_size = layer_dims[-1]
        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits


def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False


class PredictionHead(nn.Module):
    """ Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions. """
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create(cls, prediction_head_name, layer_dims, class_weights=None):
        """
        Create subclass of Prediction Head.

        :param prediction_head_name: Classname (exact string!) of prediction head we want to create
        :type prediction_head_name: str
        :param layer_dims: describing the feed forward block structure, e.g. [768,2]
        :type layer_dims: List[Int]
        :param class_weights: The loss weighting to be assigned to certain label classes during training.
           Used to correct cases where there is a strong class imbalance.
        :type class_weights: list[Float]
        :return: Prediction Head of class prediction_head_name
        """
        return cls.subclasses[prediction_head_name](layer_dims=layer_dims, class_weights=class_weights)

    def save_config(self, save_dir, head_num=0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :type save_dir: str or Path
        :param head_num: Which head to save
        :type head_num: int
        """
        self.generate_config()
        output_config_file = Path(save_dir) / f'prediction_head_{head_num}_config.json'
        with open(output_config_file, 'w') as file:
            json.dump(self.config, file)

    def save(self, save_dir, head_num=0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :type save_dir: str or Path
        :param head_num: which head to save
        :type head_num: int
        """
        output_model_file = Path(save_dir) / f'prediction_head_{head_num}.bin'
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if is_json(value) and key[0] != '_':
                config[key] = value
            if self.task_name == 'text_similarity' and key == 'similarity_function':
                config['similarity_function'] = value
        config['name'] = self.__class__.__name__
        config.pop('config', None)
        self.config = config

    @classmethod
    def load(cls, config_file, strict=True, load_weights=True):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :type config_file: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        config = json.load(open(config_file))
        prediction_head = cls.subclasses[config['name']](**config)
        if load_weights:
            model_file = cls._get_model_file(config_file=config_file)
            logger.info('Loading prediction head from {}'.format(model_file))
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=strict)
        return prediction_head

    def logits_to_loss(self, logits, labels):
        """
        Implement this function in your special Prediction Head.
        Should combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param labels: labels, can vary in shape and type, depending on task
        :type labels: object
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits):
        """
        Implement this function in your special Prediction Head.
        Should combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def prepare_labels(self, **kwargs):
        """
        Some prediction heads need additional label conversion.
        E.g. NER needs word level labels turned into subword token level labels.

        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: labels in the right format
        :rtype: object
        """
        raise NotImplementedError()

    def resize_input(self, input_dim):
        """ This function compares the output dimensionality of the language model against the input dimensionality
        of the prediction head. If there is a mismatch, the prediction head will be resized to fit."""
        if 'feed_forward' not in dir(self):
            return
        else:
            old_dims = self.feed_forward.layer_dims
            if input_dim == old_dims[0]:
                return
            new_dims = [input_dim] + old_dims[1:]
            logger.info(f'Resizing input dimensions of {type(self).__name__} ({self.task_name}) from {old_dims} to {new_dims} to match language model')
            self.feed_forward = FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file):
        if 'config.json' in str(config_file) and 'prediction_head' in str(config_file):
            head_num = int(''.join([char for char in os.path.basename(config_file) if char.isdigit()]))
            model_file = Path(os.path.dirname(config_file)) / f'prediction_head_{head_num}.bin'
        else:
            raise ValueError(f"This doesn't seem to be a proper prediction_head config file: '{config_file}'")
        return model_file

    def _set_name(self, name):
        self.task_name = name


ID_NAMES = ['example_id', 'external_id', 'doc_id', 'id']


SAMPLE = """
      .--.        _____                       _      
    .'_\\/_'.     / ____|                     | |     
    '. /\\ .'    | (___   __ _ _ __ ___  _ __ | | ___ 
      "||"       \\___ \\ / _` | '_ ` _ \\| '_ \\| |/ _ \\ 
       || /\\     ____) | (_| | | | | | | |_) | |  __/
    /\\ ||//\\)   |_____/ \\__,_|_| |_| |_| .__/|_|\\___|
   (/\\||/                             |_|           
______\\||/___________________________________________                     
"""


class Sample(object):
    """A single training/test sample. This should contain the input and the label. Is initialized with
    the human readable clear_text. Over the course of data preprocessing, this object is populated
    with tokenized and featurized versions of the data."""

    def __init__(self, id, clear_text, tokenized=None, features=None):
        """
        :param id: The unique id of the sample
        :type id: str
        :param clear_text: A dictionary containing various human readable fields (e.g. text, label).
        :type clear_text: dict
        :param tokenized: A dictionary containing the tokenized version of clear text plus helpful meta data: offsets (start position of each token in the original text) and start_of_word (boolean if a token is the first one of a word).
        :type tokenized: dict
        :param features: A dictionary containing features in a vectorized format needed by the model to process this sample.
        :type features: dict

        """
        self.id = id
        self.clear_text = clear_text
        self.features = features
        self.tokenized = tokenized

    def __str__(self):
        if self.clear_text:
            clear_text_str = '\n \t'.join([(k + ': ' + str(v)) for k, v in self.clear_text.items()])
            if len(clear_text_str) > 10000:
                clear_text_str = clear_text_str[:10000] + f'\nTHE REST IS TOO LONG TO DISPLAY. Remaining chars :{len(clear_text_str) - 10000}'
        else:
            clear_text_str = 'None'
        if self.features:
            if isinstance(self.features, list):
                features = self.features[0]
            else:
                features = self.features
            feature_str = '\n \t'.join([(k + ': ' + str(v)) for k, v in features.items()])
        else:
            feature_str = 'None'
        if self.tokenized:
            tokenized_str = '\n \t'.join([(k + ': ' + str(v)) for k, v in self.tokenized.items()])
            if len(tokenized_str) > 10000:
                tokenized_str = tokenized_str[:10000] + f'\nTHE REST IS TOO LONG TO DISPLAY. Remaining chars: {len(tokenized_str) - 10000}'
        else:
            tokenized_str = 'None'
        s = f'\n{SAMPLE}\nID: {self.id}\nClear Text: \n \t{clear_text_str}\nTokenized: \n \t{tokenized_str}\nFeatures: \n \t{feature_str}\n_____________________________________________________'
        return s


class SampleBasket:
    """ An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(self, id_internal: str, raw: dict, id_external=None, samples=None):
        """
        :param id: A unique identifying id. Used for identification within FARM.
        :type id: str
        :param external_id: Used for identification outside of FARM. E.g. if another framework wants to pass along its own id with the results.
        :type external_id: str
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :type raw: dict
        :param samples: An optional list of Samples used to populate the basket at initialization.
        :type samples: Sample
        """
        self.id_internal = id_internal
        self.id_external = id_external
        self.raw = raw
        self.samples = samples


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        if t_name == 'regression_label_ids':
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)
        else:
            try:
                check = features[0][t_name]
                if isinstance(check, numbers.Number):
                    base = check
                elif isinstance(check, list):
                    base = list(flatten_list(check))[0]
                else:
                    base = check.ravel()[0]
                if not np.issubdtype(type(base), np.integer):
                    logger.warning(f"Problem during conversion to torch tensors:\nA non-integer value for feature '{t_name}' with a value of: '{base}' will be converted to a torch tensor of dtype long.")
            except:
                logger.warning(f"Could not determine type for feature '{t_name}'. Converting now to a tensor of default type long.")
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)
        all_tensors.append(cur_tensor)
    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


DOWNSTREAM_TASK_MAP = {'gnad': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/gnad.tar.gz', 'germeval14': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval14.tar.gz', 'germeval18': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval18.tar.gz', 'squad20': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz', 'covidqa': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz', 'conll03detrain': 'https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train', 'conll03dedev': 'https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa', 'conll03detest': 'https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb', 'conll03entrain': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train', 'conll03endev': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa', 'conll03entest': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb', 'cord_19': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cord_19.tar.gz', 'lm_finetune_nips': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz', 'toxic-comments': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/toxic-comments.tar.gz', 'cola': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cola.tar.gz', 'asnq_binary': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/asnq_binary.tar.gz', 'germeval17': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval17.tar.gz', 'natural_questions': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/natural_questions.tar.gz'}


def _get_md5checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda : f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _conll03get(dataset, directory, language):
    with open(directory / f'{dataset}.txt', 'wb') as file:
        response = get(DOWNSTREAM_TASK_MAP[f'conll03{language}{dataset}'])
        file.write(response.content)
    if f'conll03{language}{dataset}' == 'conll03detrain':
        if 'ae4be68b11dc94e0001568a9095eb391' != _get_md5checksum(str(directory / f'{dataset}.txt')):
            logger.error(f'Someone has changed the file for conll03detrain. This data was collected from an external github repository.\nPlease make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')
    elif f'conll03{language}{dataset}' == 'conll03detest':
        if 'b8514f44366feae8f317e767cf425f28' != _get_md5checksum(str(directory / f'{dataset}.txt')):
            logger.error(f'Someone has changed the file for conll03detest. This data was collected from an external github repository.\nPlease make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')
    elif f'conll03{language}{dataset}' == 'conll03entrain':
        if '11a942ce9db6cc64270372825e964d26' != _get_md5checksum(str(directory / f'{dataset}.txt')):
            logger.error(f'Someone has changed the file for conll03entrain. This data was collected from an external github repository.\nPlease make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def _download_extract_downstream_data(input_file, proxies=None):
    full_path = Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info('downloading and extracting file {} to dir {}'.format(taskname, datadir))
    if 'conll03-' in taskname:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for dataset in ['train', 'dev', 'test']:
            if 'de' in taskname:
                _conll03get(dataset, directory, 'de')
            elif 'en' in taskname:
                _conll03get(dataset, directory, 'en')
            else:
                logger.error('Cannot download {}. Unknown data source.'.format(taskname))
    elif taskname not in DOWNSTREAM_TASK_MAP:
        logger.error('Cannot download {}. Unknown data source.'.format(taskname))
    else:
        if os.name == 'nt':
            delete_tmp_file = False
        else:
            delete_tmp_file = True
        with tempfile.NamedTemporaryFile(delete=delete_tmp_file) as temp_file:
            http_get(DOWNSTREAM_TASK_MAP[taskname], temp_file, proxies=proxies)
            temp_file.flush()
            temp_file.seek(0)
            if 'germeval14' in taskname:
                if '2c9d5337d7a25b9a4bf6f5672dd091bc' != _get_md5checksum(temp_file.name):
                    logger.error(f'Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')
            elif 'germeval18' in taskname:
                if '23244fa042dcc39e844635285c455205' != _get_md5checksum(temp_file.name):
                    logger.error(f'Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')
            elif 'gnad' in taskname:
                if 'ef62fe3f59c1ad54cf0271d8532b8f22' != _get_md5checksum(temp_file.name):
                    logger.error(f'Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')
            elif 'germeval17' in taskname:
                if 'f1bf67247dcfe7c3c919b7b20b3f736e' != _get_md5checksum(temp_file.name):
                    logger.error(f'Someone has changed the file for {taskname}. Please make sure the correct file is used and update the md5sum in farm/data_handler/utils.py')
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)


def read_tsv(filename, rename_columns, quotechar='"', delimiter='\t', skiprows=None, header=0, proxies=None, max_samples=None):
    """Reads a tab separated value file. Tries to download the data if filename is not found"""
    if not os.path.exists(filename):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies=proxies)
    columns_needed = list(rename_columns.keys())
    df = pd.read_csv(filename, sep=delimiter, encoding='utf-8', quotechar=quotechar, dtype=str, skiprows=skiprows, header=header, usecols=columns_needed)
    if max_samples:
        df = df.sample(max_samples)
    df.rename(columns=rename_columns, inplace=True)
    df.fillna('', inplace=True)
    raw_dict = df.to_dict(orient='records')
    return raw_dict


EMBEDDING_VOCAB_FILES_MAP = {}


def load_from_cache(pretrained_model_name_or_path, s3_dict, **kwargs):
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    s3_file = s3_dict[pretrained_model_name_or_path]
    try:
        resolved_file = cached_path(s3_file, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download)
        if resolved_file is None:
            raise EnvironmentError
    except EnvironmentError:
        if pretrained_model_name_or_path in s3_dict:
            msg = "Couldn't reach server at '{}' to download data.".format(s3_file)
        else:
            msg = "Model name '{}' was not found in model name list. We assumed '{}' was a path, a model identifier, or url to a configuration file or a directory containing such a file but couldn't find any such file at this path or url.".format(pretrained_model_name_or_path, s3_file)
        raise EnvironmentError(msg)
    if resolved_file == s3_file:
        logger.info('loading file {}'.format(s3_file))
    else:
        logger.info('loading file {} from cache at {}'.format(s3_file, resolved_file))
    return resolved_file


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <= 96 or cp >= 123 and cp <= 126:
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def run_split_on_punc(text, never_split=None):
    """Splits punctuation on a piece of text.
    Function taken from HuggingFace: transformers.tokenization_bert.BasicTokenizer
    """
    if never_split is not None and text in never_split:
        return [text]
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return [''.join(x) for x in output]


class Tokenizer:
    """
    Simple Wrapper for Tokenizers from the transformers package. Enables loading of different Tokenizer classes with a uniform interface.
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, tokenizer_class=None, use_fast=True, **kwargs):
        """
        Enables loading of different Tokenizer classes with a uniform interface. Either infer the class from
        model config or define it manually via `tokenizer_class`.

        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (e.g. `bert-base-uncased`)
        :type pretrained_model_name_or_path: str
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
        :type tokenizer_class: str
        :param use_fast: (Optional, False by default) Indicate if FARM should try to load the fast version of the tokenizer (True) or
            use the Python one (False).
            Only DistilBERT, BERT and Electra fast tokenizers are supported.
        :type use_fast: bool
        :param kwargs:
        :return: Tokenizer
        """
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        kwargs['revision'] = revision
        if tokenizer_class is None:
            tokenizer_class = cls._infer_tokenizer_class(pretrained_model_name_or_path)
        logger.info(f"Loading tokenizer of type '{tokenizer_class}'")
        ret = None
        if 'AlbertTokenizer' in tokenizer_class:
            if use_fast:
                ret = AlbertTokenizerFast.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
            else:
                ret = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
        elif 'XLMRobertaTokenizer' in tokenizer_class:
            if use_fast:
                ret = XLMRobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'RobertaTokenizer' in tokenizer_class:
            if use_fast:
                ret = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'DistilBertTokenizer' in tokenizer_class:
            if use_fast:
                ret = DistilBertTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'BertTokenizer' in tokenizer_class:
            if use_fast:
                ret = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'XLNetTokenizer' in tokenizer_class:
            if use_fast:
                ret = XLNetTokenizerFast.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
            else:
                ret = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
        elif 'ElectraTokenizer' in tokenizer_class:
            if use_fast:
                ret = ElectraTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == 'EmbeddingTokenizer':
            if use_fast:
                logger.error('EmbeddingTokenizerFast is not supported! Using EmbeddingTokenizer instead.')
                ret = EmbeddingTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = EmbeddingTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'CamembertTokenizer' in tokenizer_class:
            if use_fast:
                ret = CamembertTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = CamembertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'DPRQuestionEncoderTokenizer' in tokenizer_class:
            if use_fast:
                ret = DPRQuestionEncoderTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = DPRQuestionEncoderTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'DPRContextEncoderTokenizer' in tokenizer_class:
            if use_fast:
                ret = DPRContextEncoderTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = DPRContextEncoderTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif 'BigBirdTokenizer' in tokenizer_class:
            if use_fast:
                ret = BigBirdTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                ret = BigBirdTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if ret is None:
            raise Exception('Unable to load tokenizer')
        else:
            return ret

    @staticmethod
    def _infer_tokenizer_class(pretrained_model_name_or_path):
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except OSError:
            try:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path + '/language_model_config.json')
            except Exception as e:
                logger.warning('No config file found. Trying to infer Tokenizer type from model name')
                tokenizer_class = Tokenizer._infer_tokenizer_class_from_string(pretrained_model_name_or_path)
                return tokenizer_class
        model_type = config.model_type
        if model_type == 'xlm-roberta':
            tokenizer_class = 'XLMRobertaTokenizer'
        elif model_type == 'roberta':
            if 'mlm' in pretrained_model_name_or_path.lower():
                raise NotImplementedError('MLM part of codebert is currently not supported in FARM')
            tokenizer_class = 'RobertaTokenizer'
        elif model_type == 'camembert':
            tokenizer_class = 'CamembertTokenizer'
        elif model_type == 'albert':
            tokenizer_class = 'AlbertTokenizer'
        elif model_type == 'distilbert':
            tokenizer_class = 'DistilBertTokenizer'
        elif model_type == 'bert':
            tokenizer_class = 'BertTokenizer'
        elif model_type == 'xlnet':
            tokenizer_class = 'XLNetTokenizer'
        elif model_type == 'electra':
            tokenizer_class = 'ElectraTokenizer'
        elif model_type == 'dpr':
            if config.architectures[0] == 'DPRQuestionEncoder':
                tokenizer_class = 'DPRQuestionEncoderTokenizer'
            elif config.architectures[0] == 'DPRContextEncoder':
                tokenizer_class = 'DPRContextEncoderTokenizer'
            elif config.architectures[0] == 'DPRReader':
                raise NotImplementedError('DPRReader models are currently not supported.')
        elif model_type == 'big_bird':
            tokenizer_class = 'BigBirdTokenizer'
        else:
            logger.warning('Could not infer Tokenizer type from config. Trying to infer Tokenizer type from model name.')
            tokenizer_class = Tokenizer._infer_tokenizer_class_from_string(pretrained_model_name_or_path)
        return tokenizer_class

    @staticmethod
    def _infer_tokenizer_class_from_string(pretrained_model_name_or_path):
        if 'albert' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'AlbertTokenizer'
        elif 'bigbird' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'BigBirdTokenizer'
        elif 'xlm-roberta' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'XLMRobertaTokenizer'
        elif 'roberta' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'RobertaTokenizer'
        elif 'codebert' in pretrained_model_name_or_path.lower():
            if 'mlm' in pretrained_model_name_or_path.lower():
                raise NotImplementedError('MLM part of codebert is currently not supported in FARM')
            else:
                tokenizer_class = 'RobertaTokenizer'
        elif 'camembert' in pretrained_model_name_or_path.lower() or 'umberto' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'CamembertTokenizer'
        elif 'distilbert' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'DistilBertTokenizer'
        elif 'bert' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'BertTokenizer'
        elif 'xlnet' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'XLNetTokenizer'
        elif 'electra' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'ElectraTokenizer'
        elif 'word2vec' in pretrained_model_name_or_path.lower() or 'glove' in pretrained_model_name_or_path.lower() or 'fasttext' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'EmbeddingTokenizer'
        elif 'minilm' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'BertTokenizer'
        elif 'dpr-question_encoder' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'DPRQuestionEncoderTokenizer'
        elif 'dpr-ctx_encoder' in pretrained_model_name_or_path.lower():
            tokenizer_class = 'DPRContextEncoderTokenizer'
        else:
            raise ValueError(f"Could not infer tokenizer_class from model config or name '{pretrained_model_name_or_path}'. Set arg `tokenizer_class` in Tokenizer.load() to one of: AlbertTokenizer, XLMRobertaTokenizer, RobertaTokenizer, DistilBertTokenizer, BertTokenizer, XLNetTokenizer, CamembertTokenizer, ElectraTokenizer, DPRQuestionEncoderTokenizer,DPRContextEncoderTokenizer.")
        return tokenizer_class


def pad(seq, max_seq_len, pad_token, pad_on_left=False):
    ret = seq
    n_required_pad = max_seq_len - len(seq)
    for _ in range(n_required_pad):
        if pad_on_left:
            ret.insert(0, pad_token)
        else:
            ret.append(pad_token)
    return ret


def sample_to_features_text(sample, tasks, max_seq_len, tokenizer):
    """
    Generates a dictionary of features for a given input sample that is to be consumed by a text classification model.

    :param sample: Sample object that contains human readable text and label fields from a single text classification data sample
    :type sample: Sample
    :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
    :type tasks: dict
    :param max_seq_len: Sequences are truncated after this many tokens
    :type max_seq_len: int
    :param tokenizer: A tokenizer object that can turn string sentences into a list of tokens
    :return: A list with one dictionary containing the keys "input_ids", "padding_mask" and "segment_ids" (also "label_ids" if not
             in inference mode). The values are lists containing those features.
    :rtype: list
    """
    if tokenizer.is_fast:
        text = sample.clear_text['text']
        inputs = tokenizer(text, return_token_type_ids=True, truncation=True, truncation_strategy='longest_first', max_length=max_seq_len, return_special_tokens_mask=True)
        if len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1) != len(sample.tokenized['tokens']):
            logger.error(f"FastTokenizer encoded sample {sample.clear_text['text']} to {len(inputs['input_ids']) - inputs['special_tokens_mask'].count(1)} tokens, which differs from number of tokens produced in tokenize_with_metadata(). \nFurther processing is likely to be wrong.")
    else:
        tokens_a = sample.tokenized['tokens']
        tokens_b = sample.tokenized.get('tokens_b', None)
        inputs = tokenizer.encode_plus(tokens_a, tokens_b, add_special_tokens=True, truncation=False, return_token_type_ids=True, is_split_into_words=False)
    input_ids, segment_ids = inputs['input_ids'], inputs['token_type_ids']
    padding_mask = [1] * len(input_ids)
    if tokenizer.__class__.__name__ == 'XLNetTokenizer':
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)
    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)
    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    feat_dict = {'input_ids': input_ids, 'padding_mask': padding_mask, 'segment_ids': segment_ids}
    for task_name, task in tasks.items():
        try:
            label_name = task['label_name']
            label_raw = sample.clear_text[label_name]
            label_list = task['label_list']
            if task['task_type'] == 'classification':
                try:
                    label_ids = [label_list.index(label_raw)]
                except ValueError as e:
                    raise ValueError(f'[Task: {task_name}] Observed label {label_raw} not in defined label_list')
            elif task['task_type'] == 'multilabel_classification':
                label_ids = [0] * len(label_list)
                for l in label_raw.split(','):
                    if l != '':
                        label_ids[label_list.index(l)] = 1
            elif task['task_type'] == 'regression':
                label_ids = [float(label_raw)]
            else:
                raise ValueError(task['task_type'])
        except KeyError:
            label_ids = None
        if label_ids is not None:
            feat_dict[task['label_tensor_name']] = label_ids
    return [feat_dict]


SPECIAL_TOKENIZER_CHARS = '^(##|Ġ|▁)'


def _words_to_tokens(words, word_offsets, tokenizer):
    """
    Tokenize "words" into subword tokens while keeping track of offsets and if a token is the start of a word.

    :param words: list of words.
    :type words: list
    :param word_offsets: Character indices where each word begins in the original text
    :type word_offsets: list
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :return: tokens, offsets, start_of_word

    """
    tokens = []
    token_offsets = []
    start_of_word = []
    idx = 0
    for w, w_off in zip(words, word_offsets):
        idx += 1
        if idx % 500000 == 0:
            logger.info(idx)
        if len(w) == 0:
            continue
        elif len(tokens) == 0:
            tokens_word = tokenizer.tokenize(w)
        elif type(tokenizer) == RobertaTokenizer:
            tokens_word = tokenizer.tokenize(w, add_prefix_space=True)
        else:
            tokens_word = tokenizer.tokenize(w)
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            orig_tok = re.sub(SPECIAL_TOKENIZER_CHARS, '', tok)
            if orig_tok == tokenizer.special_tokens_map['unk_token']:
                w_off += 1
            else:
                w_off += len(orig_tok)
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)
    return tokens, token_offsets, start_of_word


def tokenize_with_metadata(text, tokenizer):
    """
    Performing tokenization while storing some important metadata for each token:

    * offsets: (int) Character index where the token begins in the original text
    * start_of_word: (bool) If the token is the start of a word. Particularly helpful for NER and QA tasks.

    We do this by first doing whitespace tokenization and then applying the model specific tokenizer to each "word".

    .. note::  We don't assume to preserve exact whitespaces in the tokens!
               This means: tabs, new lines, multiple whitespace etc will all resolve to a single " ".
               This doesn't make a difference for BERT + XLNet but it does for RoBERTa.
               For RoBERTa it has the positive effect of a shorter sequence length, but some information about whitespace
               type is lost which might be helpful for certain NLP tasks ( e.g tab for tables).

    :param text: Text to tokenize
    :type text: str
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :return: Dictionary with "tokens", "offsets" and "start_of_word"
    :rtype: dict

    """
    text = re.sub('\\s', ' ', text)
    if tokenizer.is_fast:
        tokenized2 = tokenizer.encode_plus(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        tokens2 = tokenized2['input_ids']
        offsets2 = np.array([x[0] for x in tokenized2['offset_mapping']])
        words = np.array(tokenized2.encodings[0].words)
        words[0] = -1
        words[-1] = words[-2]
        words += 1
        start_of_word2 = [0] + list(np.ediff1d(words))
        tokenized_dict = {'tokens': tokens2, 'offsets': offsets2, 'start_of_word': start_of_word2}
    else:
        words = text.split(' ')
        word_offsets = []
        cumulated = 0
        for idx, word in enumerate(words):
            word_offsets.append(cumulated)
            cumulated += len(word) + 1
        tokens, offsets, start_of_word = _words_to_tokens(words, word_offsets, tokenizer)
        tokenized_dict = {'tokens': tokens, 'offsets': offsets, 'start_of_word': start_of_word}
    return tokenized_dict


def truncate_sequences(seq_a, seq_b, tokenizer, max_seq_len, truncation_strategy='longest_first', with_special_tokens=True, stride=0):
    """
    Reduces a single sequence or a pair of sequences to a maximum sequence length.
    The sequences can contain tokens or any other elements (offsets, masks ...).
    If `with_special_tokens` is enabled, it'll remove some additional tokens to have exactly enough space for later adding special tokens (CLS, SEP etc.)

    Supported truncation strategies:

    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under max_length starting from the longest one at each token (when there is a pair of input sequences). Overflowing tokens only contains overflow from the first sequence.
    - only_first: Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
    - only_second: Only truncate the second sequence
    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)

    :param seq_a: First sequence of tokens/offsets/...
    :type seq_a: list
    :param seq_b: Optional second sequence of tokens/offsets/...
    :type seq_b: None or list
    :param tokenizer: Tokenizer (e.g. from Tokenizer.load())
    :param max_seq_len:
    :type max_seq_len: int
    :param truncation_strategy: how the sequence(s) should be truncated down. Default: "longest_first" (see above for other options).
    :type truncation_strategy: str
    :param with_special_tokens: If true, it'll remove some additional tokens to have exactly enough space for later adding special tokens (CLS, SEP etc.)
    :type with_special_tokens: bool
    :param stride: optional stride of the window during truncation
    :type stride: int
    :return: truncated seq_a, truncated seq_b, overflowing tokens

    """
    pair = bool(seq_b is not None)
    len_a = len(seq_a)
    len_b = len(seq_b) if pair else 0
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=pair) if with_special_tokens else 0
    total_len = len_a + len_b + num_special_tokens
    overflowing_tokens = []
    if max_seq_len and total_len > max_seq_len:
        seq_a, seq_b, overflowing_tokens = tokenizer.truncate_sequences(seq_a, pair_ids=seq_b, num_tokens_to_remove=total_len - max_seq_len, truncation_strategy=truncation_strategy, stride=stride)
    return seq_a, seq_b, overflowing_tokens


def expand_labels(labels_word, initial_mask, non_initial_token):
    if not labels_word:
        return None
    labels_token = []
    word_index = 0
    for im in initial_mask:
        if im:
            labels_token.append(labels_word[word_index])
            word_index += 1
        else:
            labels_token.append(non_initial_token)
    assert len(labels_token) == len(initial_mask)
    return labels_token


def _convertIOB1_to_IOB2(tags: List[str]):
    """
    script taken from: https://gist.github.com/allanj/b9bd448dc9b70d71eb7c2b6dd33fe4ef
    IOB1:  O I I B I
    IOB2:  O B I B I
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True


def _convert_germeval14_labels(tags: List[str]):
    newtags = []
    for tag in tags:
        tag = tag.replace('part', '')
        tag = tag.replace('deriv', '')
        newtags.append(tag)
    return newtags


def read_ner_file(filename, sep='\t', proxies=None):
    """
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    """
    if 'conll03-de' in str(filename):
        if sep != ' ':
            logger.error(f'Separator {sep} for dataset German CONLL03 does not match the requirements. Setting seperator to whitespace')
            sep = ' '
    if 'germeval14' in str(filename):
        if sep != '\t':
            logger.error(f'Separator {sep} for dataset GermEval14 de does not match the requirements. Setting seperator to tab')
            sep = '\t'
    if not os.path.exists(filename):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies)
    if 'conll03-de' in str(filename):
        f = open(filename, encoding='cp1252')
    else:
        f = open(filename, encoding='utf-8')
    data = []
    sentence = []
    label = []
    for line in f:
        if line.startswith('#'):
            continue
        if len(line) == 0 or '-DOCSTART-' in line or line[0] == '\n':
            if len(sentence) > 0:
                if 'conll03' in str(filename):
                    _convertIOB1_to_IOB2(label)
                if 'germeval14' in str(filename):
                    label = _convert_germeval14_labels(label)
                data.append({'text': ' '.join(sentence), 'ner_label': label})
                sentence = []
                label = []
            continue
        splits = line.split(sep)
        if 'germeval14' in str(filename):
            sentence.append(splits[1])
            label.append(splits[-2])
        else:
            sentence.append(splits[0])
            label.append(splits[-1][:-1])
    if len(sentence) > 0:
        if label[-1] == '':
            logger.error(f"The last NER label: '{splits[-1]}'  in your dataset might have been converted incorrectly. Please insert a newline at the end of the file.")
            label[-1] = 'O'
        if 'conll03-de' in str(filename):
            _convertIOB1_to_IOB2(label)
        if 'germeval14' in str(filename):
            label = _convert_germeval14_labels(label)
        data.append({'text': ' '.join(sentence), 'ner_label': label})
    return data


def get_passage_offsets(doc_offsets, doc_stride, passage_len_t, doc_text):
    """
    Get spans (start and end offsets) for passages by applying a sliding window function.
    The sliding window moves in steps of doc_stride.
    Returns a list of dictionaries which each describe the start, end and id of a passage
    that is formed when chunking a document using a sliding window approach. """
    passage_spans = []
    passage_id = 0
    doc_len_t = len(doc_offsets)
    while True:
        passage_start_t = passage_id * doc_stride
        passage_end_t = passage_start_t + passage_len_t
        passage_start_c = doc_offsets[passage_start_t]
        if passage_end_t >= doc_len_t - 1:
            passage_end_c = len(doc_text)
        else:
            end_ch_idx = doc_offsets[passage_end_t + 1]
            raw_passage_text = doc_text[:end_ch_idx]
            passage_end_c = len(raw_passage_text.strip())
        passage_span = {'passage_start_t': passage_start_t, 'passage_end_t': passage_end_t, 'passage_start_c': passage_start_c, 'passage_end_c': passage_end_c, 'passage_id': passage_id}
        passage_spans.append(passage_span)
        passage_id += 1
        if passage_end_t >= doc_len_t:
            break
    return passage_spans


def offset_to_token_idx_vecorized(token_offsets, ch_idx):
    """ Returns the idx of the token at the given character idx"""
    if ch_idx >= np.max(token_offsets):
        idx = np.argmax(token_offsets) + 1
    else:
        idx = np.argmax(token_offsets > ch_idx) - 1
    return idx


def read_squad_file(filename, proxies=None):
    """Read a SQuAD json file"""
    if not os.path.exists(filename):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies)
    with open(filename, 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    return input_data


def _get_start_of_word_QA(word_ids):
    words = np.array(word_ids)
    start_of_word_single = [1] + list(np.ediff1d(words))
    return start_of_word_single


def tokenize_batch_question_answering(pre_baskets, tokenizer, indices):
    """
    Tokenizes text data for question answering tasks. Tokenization means splitting words into subwords, depending on the
    tokenizer's vocabulary.

    - We first tokenize all documents in batch mode. (When using FastTokenizers Rust multithreading can be enabled by TODO add how to enable rust mt)
    - Then we tokenize each question individually
    - We construct dicts with question and corresponding document text + tokens + offsets + ids

    :param pre_baskets: input dicts with QA info #todo change to input objects
    :param tokenizer: tokenizer to be used
    :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets are unique
    :return: baskets, list containing question and corresponding document information
    """
    assert len(indices) == len(pre_baskets)
    assert tokenizer.is_fast, "Processing QA data is only supported with fast tokenizers for now.\nPlease load Tokenizers with 'use_fast=True' option."
    baskets = []
    texts = [d['context'] for d in pre_baskets]
    tokenized_docs_batch = tokenizer.batch_encode_plus(texts, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False, verbose=False)
    tokenids_batch = tokenized_docs_batch['input_ids']
    offsets_batch = []
    for o in tokenized_docs_batch['offset_mapping']:
        offsets_batch.append(np.array([x[0] for x in o]))
    start_of_words_batch = []
    for e in tokenized_docs_batch.encodings:
        start_of_words_batch.append(_get_start_of_word_QA(e.words))
    for i_doc, d in enumerate(pre_baskets):
        document_text = d['context']
        for i_q, q in enumerate(d['qas']):
            question_text = q['question']
            tokenized_q = tokenizer.encode_plus(question_text, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=False)
            question_tokenids = tokenized_q['input_ids']
            question_offsets = [x[0] for x in tokenized_q['offset_mapping']]
            question_sow = _get_start_of_word_QA(tokenized_q.encodings[0].words)
            external_id = q['id']
            internal_id = f'{indices[i_doc]}-{i_q}'
            raw = {'document_text': document_text, 'document_tokens': tokenids_batch[i_doc], 'document_offsets': offsets_batch[i_doc], 'document_start_of_word': start_of_words_batch[i_doc], 'question_text': question_text, 'question_tokens': question_tokenids, 'question_offsets': question_offsets, 'question_start_of_word': question_sow, 'answers': q['answers']}
            raw['document_tokens_strings'] = tokenized_docs_batch.encodings[i_doc].tokens
            raw['question_tokens_strings'] = tokenized_q.encodings[0].tokens
            baskets.append(SampleBasket(raw=raw, id_internal=internal_id, id_external=external_id, samples=None))
    return baskets


def try_get(keys, dictionary):
    try:
        for key in keys:
            if key in dictionary:
                ret = dictionary[key]
                if type(ret) == list:
                    ret = ret[0]
                return ret
    except Exception as e:
        logger.warning(f'Cannot extract from dict {dictionary} with error: {e}')
    return None


class Processor(ABC):
    """
    Is used to generate PyTorch Datasets from input data. An implementation of this abstract class should be created
    for each new data source.
    Implement the abstract methods: file_to_dicts(), _dict_to_samples(), _sample_to_features()
    to be compatible with your data format
    """
    subclasses = {}

    def __init__(self, tokenizer, max_seq_len, train_filename, dev_filename, test_filename, dev_split, data_dir, tasks={}, proxies=None, multithreading_rust=True):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param data_dir: The directory in which the train, test and perhaps dev files can be found.
        :type data_dir: str
        :param tasks: Tasks for which the processor shall extract labels from the input data.
                      Usually this includes a single, default task, e.g. text classification.
                      In a multitask setting this includes multiple tasks, e.g. 2x text classification.
                      The task name will be used to connect with the related PredictionHead.
        :type tasks: dict
        :param proxies: proxy configuration to allow downloads of remote datasets.
                    Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :type proxies: dict
        :param multithreading_rust: Whether to allow multithreading in Rust, e.g. for FastTokenizers.
                                    Note: Enabling multithreading in Rust AND multiprocessing in python might cause
                                    deadlocks.
        :type multithreading_rust: bool
        """
        if not multithreading_rust:
            os.environ['RAYON_RS_NUM_CPUS'] = '1'
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tasks = tasks
        self.proxies = proxies
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.dev_split = dev_split
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = None
        self.baskets = []
        self._log_params()
        self.problematic_sample_ids = set()

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific Processor implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load(cls, processor_name, data_dir, tokenizer, max_seq_len, train_filename, dev_filename, test_filename, dev_split, **kwargs):
        """
        Loads the class of processor specified by processor name.

        :param processor_name: The class of processor to be loaded.
        :type processor_name: str
        :param data_dir: Directory where data files are located.
        :type data_dir: str
        :param tokenizer: A tokenizer object
        :param max_seq_len: Sequences longer than this will be truncated.
        :type max_seq_len: int
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data.
                             If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: The name of the file containing test data.
        :type test_filename: str
        :param dev_split: The proportion of the train set that will sliced.
                          Only works if dev_filename is set to None
        :type dev_split: float
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: An instance of the specified processor.
        """
        sig = signature(cls.subclasses[processor_name])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(f"Got more parameters than needed for loading {processor_name}: {unused_args}. Those won't be used!")
        processor = cls.subclasses[processor_name](data_dir=data_dir, tokenizer=tokenizer, max_seq_len=max_seq_len, train_filename=train_filename, dev_filename=dev_filename, test_filename=test_filename, dev_split=dev_split, **kwargs)
        return processor

    @classmethod
    def load_from_dir(cls, load_dir):
        """
         Infers the specific type of Processor from a config file (e.g. GNADProcessor) and loads an instance of it.

        :param load_dir: str, directory that contains a 'processor_config.json'
        :return: An instance of a Processor Subclass (e.g. GNADProcessor)
        """
        processor_config_file = Path(load_dir) / 'processor_config.json'
        config = json.load(open(processor_config_file))
        config['inference'] = True
        if 'lower_case' in config.keys():
            logger.warning("Loading tokenizer from deprecated FARM config. If you used `custom_vocab` or `never_split_chars`, this won't work anymore.")
            tokenizer = Tokenizer.load(load_dir, tokenizer_class=config['tokenizer'], do_lower_case=config['lower_case'])
        else:
            tokenizer = Tokenizer.load(load_dir, tokenizer_class=config['tokenizer'])
        del config['tokenizer']
        processor = cls.load(tokenizer=tokenizer, processor_name=config['processor'], **config)
        for task_name, task in config['tasks'].items():
            processor.add_task(name=task_name, metric=task['metric'], label_list=task['label_list'], label_column_name=task['label_column_name'], text_column_name=task.get('text_column_name', None), task_type=task['task_type'])
        if processor is None:
            raise Exception
        return processor

    @classmethod
    def convert_from_transformers(cls, tokenizer_name_or_path, task_type, max_seq_len, doc_stride, revision=None, tokenizer_class=None, tokenizer_args=None, use_fast=True, **kwargs):
        config = AutoConfig.from_pretrained(tokenizer_name_or_path, revision=revision, **kwargs)
        tokenizer_args = tokenizer_args or {}
        tokenizer = Tokenizer.load(tokenizer_name_or_path, tokenizer_class=tokenizer_class, use_fast=use_fast, revision=revision, **tokenizer_args, **kwargs)
        if task_type == 'question_answering':
            processor = SquadProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len, label_list=['start_token', 'end_token'], metric='squad', data_dir='data', doc_stride=doc_stride)
        elif task_type == 'embeddings':
            processor = InferenceProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len)
        elif task_type == 'text_classification':
            label_list = list(config.id2label[id] for id in range(len(config.id2label)))
            processor = TextClassificationProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir='data', label_list=label_list, label_column_name='label', metric='acc', quote_char='"')
        elif task_type == 'ner':
            label_list = list(config.id2label.values())
            processor = NERProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len, data_dir='data', metric='seq_f1', label_list=label_list)
        else:
            raise ValueError(f"`task_type` {task_type} is not supported yet. Valid options for arg `task_type`: 'question_answering', 'embeddings', 'text_classification', 'ner'")
        return processor

    def save(self, save_dir):
        """
        Saves the vocabulary to file and also creates a json file containing all the
        information needed to load the same processor.

        :param save_dir: Directory where the files are to be saved
        :type save_dir: str
        """
        os.makedirs(save_dir, exist_ok=True)
        config = self.generate_config()
        config['tokenizer'] = self.tokenizer.__class__.__name__
        self.tokenizer.save_pretrained(str(save_dir))
        config['processor'] = self.__class__.__name__
        output_config_file = Path(save_dir) / 'processor_config.json'
        with open(output_config_file, 'w') as file:
            json.dump(config, file)

    def generate_config(self):
        """
        Generates config file from Class and instance attributes (only for sensible config parameters).
        """
        config = {}
        for key, value in inspect.getmembers(self):
            if is_json(value) and key[0] != '_':
                if issubclass(type(value), Path):
                    value = str(value)
                config[key] = value
        return config

    def add_task(self, name, metric, label_list, label_column_name=None, label_name=None, task_type=None, text_column_name=None):
        if type(label_list) is not list:
            raise ValueError(f'Argument `label_list` must be of type list. Got: f{type(label_list)}')
        if label_name is None:
            label_name = f'{name}_label'
        label_tensor_name = label_name + '_ids'
        self.tasks[name] = {'label_list': label_list, 'metric': metric, 'label_tensor_name': label_tensor_name, 'label_name': label_name, 'label_column_name': label_column_name, 'text_column_name': text_column_name, 'task_type': task_type}

    @abc.abstractmethod
    def file_to_dicts(self, file: str) ->[dict]:
        raise NotImplementedError()

    def _dict_to_samples(cls, dictionary: dict, all_dicts=None) ->[Sample]:
        raise NotImplementedError()

    def _sample_to_features(cls, sample: Sample) ->dict:
        raise NotImplementedError()

    def _dict_to_samples_and_features(self, dictionary: dict, all_dicts=None) ->[Sample]:
        raise NotImplementedError()

    def _init_samples_in_baskets(self):
        all_dicts = [b.raw for b in self.baskets]
        for basket in self.baskets:
            try:
                basket.samples = self._dict_to_samples(dictionary=basket.raw, all_dicts=all_dicts)
                for num, sample in enumerate(basket.samples):
                    sample.id = f'{basket.id_internal}-{num}'
            except Exception as e:
                logger.error(f'Could not create sample(s) from this dict: \n {basket.raw}')
                logger.error(f'Error message: {e}')

    def _featurize_samples(self):
        curr_problematic_sample_ids = []
        for basket in self.baskets:
            for sample in basket.samples:
                try:
                    sample.features = self._sample_to_features(sample=sample)
                except Exception as e:
                    curr_problematic_sample_ids.append(sample.id)
        if curr_problematic_sample_ids:
            self.problematic_sample_ids.update(curr_problematic_sample_ids)

    @staticmethod
    def log_problematic(problematic_sample_ids):
        if problematic_sample_ids:
            n_problematic = len(problematic_sample_ids)
            problematic_id_str = ', '.join([str(i) for i in problematic_sample_ids])
            logger.error(f'Unable to convert {n_problematic} samples to features. Their ids are : {problematic_id_str}')

    def _init_and_featurize_samples_in_baskets(self):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
            try:
                basket.samples = self._dict_to_samples_and_features(dictionary=basket.raw, all_dicts=all_dicts, basket_id_internal=basket.id_internal)
                for num, sample in enumerate(basket.samples):
                    sample.id = f'{basket.id_internal}-{num}'
            except Exception as e:
                logger.error(f'Could not create sample(s) from this dict: \n {basket.raw}')
                logger.error(f'Error message: {e}')

    @staticmethod
    def _check_sample_features(basket):
        """Check if all samples in the basket has computed its features.

        Args:
            basket: the basket containing the samples

        Returns:
            True if all the samples in the basket has computed its features, False otherwise

        """
        if len(basket.samples) == 0:
            return False
        for sample in basket.samples:
            if sample.features is None:
                return False
        return True

    def _create_dataset(self):
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                self.baskets.remove(basket)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This can be used for inference mode.

        :param dicts: List of dictionaries where each contains the data of one input sample.
        :type dicts: list of dicts
        :return: a Pytorch dataset and a list of tensor names.
        """
        self.baskets = []
        for id_internal, d in enumerate(dicts):
            id_external = self._id_from_dict(d)
            if indices:
                id_internal = indices[id_internal]
            self.baskets.append(SampleBasket(raw=d, id_external=id_external, id_internal=id_internal))
        self._init_samples_in_baskets()
        self._featurize_samples()
        if indices:
            if 0 in indices:
                self._log_samples(1)
        else:
            self._log_samples(1)
        dataset, tensor_names = self._create_dataset()
        if return_baskets:
            return dataset, tensor_names, self.problematic_sample_ids, self.baskets
        else:
            return dataset, tensor_names, self.problematic_sample_ids

    def _log_samples(self, n_samples):
        logger.info('*** Show {} random examples ***'.format(n_samples))
        for i in range(n_samples):
            random_basket = random.choice(self.baskets)
            random_sample = random.choice(random_basket.samples)
            logger.info(random_sample)

    def _log_params(self):
        params = {'processor': self.__class__.__name__, 'tokenizer': self.tokenizer.__class__.__name__}
        names = ['max_seq_len', 'dev_split']
        for name in names:
            value = getattr(self, name)
            params.update({name: str(value)})
        MlLogger.log_params(params)

    @staticmethod
    def _id_from_dict(d):
        ext_id = try_get(ID_NAMES, d)
        if not ext_id and 'qas' in d:
            ext_id = try_get(ID_NAMES, d['qas'][0])
        return ext_id


def loss_per_head_sum(loss_per_head, global_step=None, batch=None):
    """
    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
    Output: aggregated loss (tensor)
    """
    return sum(loss_per_head)


class AdaptiveModel(nn.Module, BaseAdaptiveModel):
    """ PyTorch implementation containing all the modelling needed for your NLP task. Combines a language
    model and a prediction head. Allows for gradient flow back to the language model component."""

    def __init__(self, language_model, prediction_heads, embeds_dropout_prob, lm_output_types, device, loss_aggregation_fn=None):
        """
        :param language_model: Any model that turns token ids into vector representations
        :type language_model: LanguageModel
        :param prediction_heads: A list of models that take embeddings and return logits for a given task
        :type prediction_heads: list
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by the
           language model will be zeroed.
        :param embeds_dropout_prob: float
        :param lm_output_types: How to extract the embeddings from the final layer of the language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :type lm_output_types: list or str
        :param device: The device on which this model will operate. Either "cpu" or "cuda".
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
                                    Output: aggregated loss (tensor)
                                    Default is a simple sum:
                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
                                    However, you can pass more complex functions that depend on the
                                    current step (e.g. for round-robin style multitask learning) or the actual
                                    content of the batch (e.g. certain labels)
                                    Note: The loss at this stage is per sample, i.e one tensor of
                                    shape (batchsize) per prediction head.
        :type loss_aggregation_fn: function
        """
        super(AdaptiveModel, self).__init__()
        self.device = device
        self.language_model = language_model
        self.lm_output_dims = language_model.get_output_dims()
        self.prediction_heads = nn.ModuleList([ph for ph in prediction_heads])
        self.fit_heads_to_lm()
        for head in self.prediction_heads:
            if head.model_type == 'language_modelling':
                head.set_shared_weights(language_model.model.embeddings.word_embeddings.weight)
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = [lm_output_types] if isinstance(lm_output_types, str) else lm_output_types
        self.log_params()
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def fit_heads_to_lm(self):
        """This iterates over each prediction head and ensures that its input dimensionality matches the output
        dimensionality of the language model. If it doesn't, it is resized so it does fit"""
        for ph in self.prediction_heads:
            ph.resize_input(self.lm_output_dims)
            ph

    def bypass_ph(self):
        """Replaces methods in the prediction heads with dummy functions. Used for benchmarking where we want to
        isolate the lm run time from ph run time."""

        def fake_forward(x):
            """Slices lm vector outputs of shape (batch_size, max_seq_len, dims) --> (batch_size, max_seq_len, 2)"""
            return x.narrow(2, 0, 2)

        def fake_logits_to_preds(logits, **kwargs):
            batch_size = logits.shape[0]
            return [None, None] * batch_size

        def fake_formatted_preds(**kwargs):
            return None
        for ph in self.prediction_heads:
            ph.forward = fake_forward
            ph.logits_to_preds = fake_logits_to_preds
            ph.formatted_preds = fake_formatted_preds

    def save(self, save_dir):
        """
        Saves the language model and prediction heads. This will generate a config file
        and model weights for each.

        :param save_dir: path to save to
        :type save_dir: Path
        """
        os.makedirs(save_dir, exist_ok=True)
        self.language_model.save(save_dir)
        for i, ph in enumerate(self.prediction_heads):
            ph.save(save_dir, i)

    @classmethod
    def load(cls, load_dir, device, strict=True, lm_name=None, processor=None):
        """
        Loads an AdaptiveModel from a directory. The directory must contain:

        * language_model.bin
        * language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Tokens

        :param load_dir: location where adaptive model is stored
        :type load_dir: Path
        :param device: to which device we want to sent the model, either cpu or cuda
        :type device: torch.device
        :param lm_name: the name to assign to the loaded language model
        :type lm_name: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        """
        if lm_name:
            language_model = LanguageModel.load(load_dir, farm_lm_name=lm_name)
        else:
            language_model = LanguageModel.load(load_dir)
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)
        model = cls(language_model, prediction_heads, 0.1, ph_output_type, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)
        return model

    def logits_to_loss_per_head(self, logits, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: logits, can vary in shape and type, depending on task.
        :type logits: object
        :return: The per sample per prediciton head loss whose first two dimensions have length n_pred_heads, batch_size
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            assert hasattr(head, 'label_tensor_name'), f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model with the processor through either 'model.connect_heads_with_processor(processor.tasks)' or by passing the processor to the Adaptive Model?"
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits, global_step=None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param global_step: number of current training step
        :type global_step: int
        :param kwargs: placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :type kwargs: object
        :return loss: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(self, **kwargs):
        """
        Push data through the whole model and returns logits. The data will propagate through the language
        model and each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to the language model and prediction head(s).
        :return: all logits as torch.tensor or multiple tensors.
        """
        sequence_output, pooled_output = self.forward_lm(**kwargs)
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm_out in zip(self.prediction_heads, self.lm_output_types):
                if lm_out == 'per_token':
                    output = self.dropout(sequence_output)
                elif lm_out == 'per_sequence' or lm_out == 'per_sequence_continuous':
                    output = self.dropout(pooled_output)
                elif lm_out == 'per_token_squad':
                    output = self.dropout(sequence_output)
                else:
                    raise ValueError('Unknown extraction strategy from language model: {}'.format(lm_out))
                all_logits.append(head(output))
        else:
            all_logits.append((sequence_output, pooled_output))
        return all_logits

    def forward_lm(self, **kwargs):
        """
        Forward pass for the language model

        :param kwargs:
        :return:
        """
        try:
            extraction_layer = self.language_model.extraction_layer
        except:
            extraction_layer = -1
        if extraction_layer == -1:
            sequence_output, pooled_output = self.language_model(**kwargs, return_dict=False, output_all_encoded_layers=False)
        else:
            self.language_model.enable_hidden_states_output()
            sequence_output, pooled_output, all_hidden_states = self.language_model(**kwargs, return_dict=False)
            sequence_output = all_hidden_states[extraction_layer]
            pooled_output = None
            self.language_model.disable_hidden_states_output()
        return sequence_output, pooled_output

    def log_params(self):
        """
        Logs paramteres to generic logger MlLogger
        """
        params = {'lm_type': self.language_model.__class__.__name__, 'lm_name': self.language_model.name, 'prediction_heads': ','.join([head.__class__.__name__ for head in self.prediction_heads]), 'lm_output_types': ','.join(self.lm_output_types)}
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")

    def verify_vocab_size(self, vocab_size):
        """ Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()"""
        model_vocab_len = self.language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
        msg = f"Vocab size of tokenizer {vocab_size} doesn't match with model {model_vocab_len}. If you added a custom vocabulary to the tokenizer, make sure to supply 'n_added_tokens' to LanguageModel.load() and BertStyleLM.load()"
        assert vocab_size == model_vocab_len, msg
        for head in self.prediction_heads:
            if head.model_type == 'language_modelling':
                ph_decoder_len = head.decoder.weight.shape[0]
                assert vocab_size == ph_decoder_len, msg

    def get_language(self):
        return self.language_model.language

    def convert_to_transformers(self):
        """
        Convert an adaptive model to huggingface's transformers format. Returns a list containing one model for each
        prediction head.

        :return: List of huggingface transformers models.
        """
        return conv.Converter.convert_to_transformers(self)

    @classmethod
    def convert_from_transformers(cls, model_name_or_path, device, revision=None, task_type=None, processor=None, **kwargs):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in FARM (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path: local path of a saved model or name of a public one.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - deepset/bert-large-uncased-whole-word-masking-squad2

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param device: "cpu" or "cuda"
        :param task_type: One of :
                          - 'question_answering'
                          - 'text_classification'
                          - 'embeddings'
                          More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :return: AdaptiveModel
        """
        return conv.Converter.convert_from_transformers(model_name_or_path, revision=revision, device=device, task_type=task_type, processor=processor, **kwargs)

    @classmethod
    def convert_to_onnx(cls, model_name, output_path, task_type, convert_to_float16=False, quantize=False, opset_version=11):
        """
        Convert a PyTorch model from transformers hub to an ONNX Model.

        :param model_name: transformers model name
        :type model_name: str
        :param output_path: output Path to write the converted to
        :type output_path: Path
        :param task_type: Type of task for the model. Available options: "embeddings", "question_answering",
                          "text_classification", "ner".
        :param convert_to_float16: By default, the model use float32 precision. With half precision of flaot16, inference
                                should be faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs, float32
                                might be more performant.
        :type convert_to_float16: bool
        :param quantize: convert floating point number to integers
        :type quantize: bool
        :param opset_version: ONNX opset version
        :type opset_version: int
        :return:
        """
        language_model_class = LanguageModel.get_language_model_class(model_name)
        if language_model_class not in ['Bert', 'Roberta', 'XLMRoberta']:
            raise Exception("The current ONNX conversion only support 'BERT', 'RoBERTa', and 'XLMRoberta' models.")
        task_type_to_pipeline_map = {'question_answering': 'question-answering', 'embeddings': 'feature-extraction', 'ner': 'ner'}
        convert(pipeline_name=task_type_to_pipeline_map[task_type], framework='pt', model=model_name, output=output_path / 'model.onnx', opset=opset_version, use_external_format=True if language_model_class == 'XLMRoberta' else False)
        processor = Processor.convert_from_transformers(tokenizer_name_or_path=model_name, task_type=task_type, max_seq_len=256, doc_stride=128, use_fast=True)
        processor.save(output_path)
        model = AdaptiveModel.convert_from_transformers(model_name, device='cpu', task_type=task_type)
        model.save(output_path)
        os.remove(output_path / 'language_model.bin')
        onnx_model_config = {'task_type': task_type, 'onnx_opset_version': opset_version, 'language_model_class': language_model_class, 'language': model.language_model.language}
        with open(output_path / 'onnx_model_config.json', 'w') as f:
            json.dump(onnx_model_config, f)
        if convert_to_float16:
            config = AutoConfig.from_pretrained(model_name)
            optimized_model = optimizer.optimize_model(input=str(output_path / 'model.onnx'), model_type='bert', num_heads=config.num_hidden_layers, hidden_size=config.hidden_size)
            optimized_model.convert_model_float32_to_float16()
            optimized_model.save_model_to_file(str(output_path / 'model.onnx'))
        if quantize:
            quantize_model(output_path / 'model.onnx')


class ONNXWrapper(AdaptiveModel):
    """
    Wrapper Class for converting PyTorch models to ONNX.

    As of torch v1.4.0, torch.onnx.export only support passing positional arguments to the forward pass of the model.
    However, the AdaptiveModel's forward takes keyword arguments. This class circumvents the issue by converting
    positional arguments to keyword arguments.
    """

    @classmethod
    def load_from_adaptive_model(cls, adaptive_model):
        model = copy.deepcopy(adaptive_model)
        model.__class__ = ONNXWrapper
        return model

    def forward(self, *batch):
        return super().forward(input_ids=batch[0], padding_mask=batch[1], segment_ids=batch[2])


class BaseBiAdaptiveModel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific AdaptiveModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: arguments to pass for loading the model.
        :return: instance of a model
        """
        if (Path(kwargs['load_dir']) / 'model.onnx').is_file():
            model = cls.subclasses['ONNXBiAdaptiveModel'].load(**kwargs)
        else:
            model = cls.subclasses['BiAdaptiveModel'].load(**kwargs)
        return model

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all predictions from all prediction heads
        """
        all_preds = []
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits, language_model1, language_model2, **kwargs):
        """
        Format predictions to strings for inference output

        :param logits: model logits
        :type logits: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        n_heads = len(self.prediction_heads)
        if n_heads == 1:
            preds_final = []
            try:
                preds = kwargs['preds']
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs['preds'] = preds_flat
            except KeyError:
                kwargs['preds'] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and 'predictions' in preds:
                preds_final.append(preds)
        return preds_final

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]['label_tensor_name']
            label_list = tasks[head.task_name]['label_list']
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]['label_list']
            head.label_list = label_list
            num_labels = len(label_list)
            head.metric = tasks[head.task_name]['metric']

    @classmethod
    def _get_prediction_head_files(cls, load_dir, strict=True):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        config_files = [(load_dir / f) for f in files if 'config.json' in f and 'prediction_head' in f]
        config_files.sort()
        return config_files


def all_reduce(tensor, group=None):
    if group is None:
        group = dist.group.WORLD
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4
    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError('encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer
    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(256 ** SIZE_STORAGE_BYTES)
    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')
    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES:enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))
    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start:start + size].copy_(cpu_buffer[:size])
    all_reduce(buffer, group=group)
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size:(i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES:size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception('Unable to unpickle data from other workers. all_gather_list requires all workers to enter the function together, so this error usually indicates that the workers have fallen out of sync somehow. Workers can fall out of sync if one of them runs out of memory, or if there are other conditions in your training script that can cause one worker to finish an epoch while other workers are still iterating over their portions of the data.')


class TextSimilarityHead(PredictionHead):
    """
    Trains a head on predicting the similarity of two texts like in Dense Passage Retrieval.
    """

    def __init__(self, similarity_function: str='dot_product', global_loss_buffer_size: int=150000, **kwargs):
        """
        Init the TextSimilarityHead.

        :param similarity_function: Function to calculate similarity between queries and passage embeddings.
                                    Choose either "dot_product" (Default) or "cosine".
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up

        :param kwargs:
        """
        super(TextSimilarityHead, self).__init__()
        self.similarity_function = similarity_function
        self.loss_fct = NLLLoss(reduction='mean')
        self.task_name = 'text_similarity'
        self.model_type = 'text_similarity'
        self.ph_output_type = 'per_sequence'
        self.global_loss_buffer_size = global_loss_buffer_size
        self.generate_config()

    @classmethod
    def dot_product_scores(cls, query_vectors, passage_vectors):
        """
        Calculates dot product similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                        of dimension n1 x D,
                        where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                        of dimension n2 x D,
                        where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                        and D is embedding size
        :type passage_vectors: torch.Tensor

        :return dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
        """
        dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
        return dot_product

    @classmethod
    def cosine_scores(cls, query_vectors, passage_vectors):
        """
        Calculates cosine similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                          and D is embedding size
        :type passage_vectors: torch.Tensor

        :return: cosine similarity score of each query with each context/passage (dimension: n1xn2)
        """
        cosine_similarities = []
        passages_per_batch = passage_vectors.shape[0]
        for query_vector in query_vectors:
            query_vector_repeated = query_vector.repeat(passages_per_batch, 1)
            current_cosine_similarities = nn.functional.cosine_similarity(query_vector_repeated, passage_vectors, dim=1)
            cosine_similarities.append(current_cosine_similarities)
        return torch.stack(cosine_similarities)

    def get_similarity_function(self):
        """
        Returns the type of similarity function used to compare queries and passages/contexts
        """
        if 'dot_product' in self.similarity_function:
            return TextSimilarityHead.dot_product_scores
        elif 'cosine' in self.similarity_function:
            return TextSimilarityHead.cosine_scores

    def forward(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Only packs the embeddings from both language models into a tuple. No further modification.
        The similarity calculation is handled later to enable distributed training (DDP)
        while keeping the support for in-batch negatives.
        (Gather all embeddings from nodes => then do similarity scores + loss)

        :param query_vectors: Tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :type query_vectors: torch.Tensor
        :param passage_vectors: Tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is the number of queries/batch size and D is embedding size
        :type passage_vectors: torch.Tensor

        :return: (query_vectors, passage_vectors)
        """
        return query_vectors, passage_vectors

    def _embeddings_to_scores(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor):
        """
        Calculates similarity scores between all given query_vectors and passage_vectors

        :param query_vectors: Tensor of queries encoded by the query encoder model
        :param passage_vectors: Tensor of passages encoded by the passage encoder model
        :return: Tensor of log softmax similarity scores of each query with each passage (dimension: n1xn2)
        """
        sim_func = self.get_similarity_function()
        scores = sim_func(query_vectors, passage_vectors)
        if len(query_vectors.size()) > 1:
            q_num = query_vectors.size(0)
            scores = scores.view(q_num, -1)
        softmax_scores = nn.functional.log_softmax(scores, dim=1)
        return softmax_scores

    def logits_to_loss(self, logits: Tuple[torch.Tensor, torch.Tensor], label_ids, **kwargs):
        """
        Computes the loss (Default: NLLLoss) by applying a similarity function (Default: dot product) to the input
        tuple of (query_vectors, passage_vectors) and afterwards applying the loss function on similarity scores.

        :param logits: Tuple of Tensors (query_embedding, passage_embedding) as returned from forward()

        :return: negative log likelihood loss from similarity scores
        """
        try:
            rank = torch.distributed.get_rank()
        except (AssertionError, RuntimeError):
            rank = -1
        query_vectors, passage_vectors = logits
        positive_idx_per_question = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)
        if rank != -1:
            q_vector_to_send = torch.empty_like(query_vectors).cpu().copy_(query_vectors).detach_()
            p_vector_to_send = torch.empty_like(passage_vectors).cpu().copy_(passage_vectors).detach_()
            global_question_passage_vectors = all_gather_list([q_vector_to_send, p_vector_to_send, positive_idx_per_question], max_size=self.global_loss_buffer_size)
            global_query_vectors = []
            global_passage_vectors = []
            global_positive_idx_per_question = []
            total_passages = 0
            for i, item in enumerate(global_question_passage_vectors):
                q_vector, p_vectors, positive_idx = item
                if i != rank:
                    global_query_vectors.append(q_vector)
                    global_passage_vectors.append(p_vectors)
                    global_positive_idx_per_question.extend([(v + total_passages) for v in positive_idx])
                else:
                    global_query_vectors.append(query_vectors)
                    global_passage_vectors.append(passage_vectors)
                    global_positive_idx_per_question.extend([(v + total_passages) for v in positive_idx_per_question])
                total_passages += p_vectors.size(0)
            global_query_vectors = torch.cat(global_query_vectors, dim=0)
            global_passage_vectors = torch.cat(global_passage_vectors, dim=0)
            global_positive_idx_per_question = torch.LongTensor(global_positive_idx_per_question)
        else:
            global_query_vectors = query_vectors
            global_passage_vectors = passage_vectors
            global_positive_idx_per_question = positive_idx_per_question
        softmax_scores = self._embeddings_to_scores(global_query_vectors, global_passage_vectors)
        targets = global_positive_idx_per_question.squeeze(-1)
        loss = self.loss_fct(softmax_scores, targets)
        return loss

    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        """
        Returns predicted ranks(similarity) of passages/context for each query

        :param logits: tensor of log softmax similarity scores of each query with each context/passage (dimension: n1xn2)
        :type logits: torch.Tensor

        :return: predicted ranks of passages for each query
        """
        query_vectors, passage_vectors = logits
        softmax_scores = self._embeddings_to_scores(query_vectors, passage_vectors)
        _, sorted_scores = torch.sort(softmax_scores, dim=1, descending=True)
        return sorted_scores

    def prepare_labels(self, label_ids, **kwargs):
        """
        Returns a tensor with passage labels(0:hard_negative/1:positive) for each query

        :return: passage labels(0:hard_negative/1:positive) for each query
        """
        labels = torch.zeros(label_ids.size(0), label_ids.numel())
        positive_indices = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)
        for i, indx in enumerate(positive_indices):
            labels[i, indx.item()] = 1
        return labels

    def formatted_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        raise NotImplementedError('formatted_preds is not supported in TextSimilarityHead yet!')


class BiAdaptiveModel(nn.Module, BaseBiAdaptiveModel):
    """ PyTorch implementation containing all the modelling needed for your NLP task. Combines 2 language
    models for representation of 2 sequences and a prediction head. Allows for gradient flow back to the 2 language model components."""

    def __init__(self, language_model1, language_model2, prediction_heads, embeds_dropout_prob=0.1, device='cuda', lm1_output_types=['per_sequence'], lm2_output_types=['per_sequence'], loss_aggregation_fn=None):
        """
        :param language_model1: Any model that turns token ids into vector representations
        :type language_model1: LanguageModel
        :param language_model2: Any model that turns token ids into vector representations
        :type language_model2: LanguageModel
        :param prediction_heads: A list of models that take 2 sequence embeddings and return logits for a given task
        :type prediction_heads: list
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by any of the 2
           language model will be zeroed.
        :param embeds_dropout_prob: float
        :param lm1_output_types: How to extract the embeddings from the final layer of the first language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :type lm1_output_types: list or str
        :param lm2_output_types: How to extract the embeddings from the final layer of the second language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :type lm2_output_types: list or str
        :param device: The device on which this model will operate. Either "cpu" or "cuda".
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
                                    Output: aggregated loss (tensor)
                                    Default is a simple sum:
                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
                                    However, you can pass more complex functions that depend on the
                                    current step (e.g. for round-robin style multitask learning) or the actual
                                    content of the batch (e.g. certain labels)
                                    Note: The loss at this stage is per sample, i.e one tensor of
                                    shape (batchsize) per prediction head.
        :type loss_aggregation_fn: function
        """
        super(BiAdaptiveModel, self).__init__()
        self.device = device
        self.language_model1 = language_model1
        self.lm1_output_dims = language_model1.get_output_dims()
        self.language_model2 = language_model2
        self.lm2_output_dims = language_model2.get_output_dims()
        self.dropout1 = nn.Dropout(embeds_dropout_prob)
        self.dropout2 = nn.Dropout(embeds_dropout_prob)
        self.prediction_heads = nn.ModuleList([ph for ph in prediction_heads])
        self.lm1_output_types = [lm1_output_types] if isinstance(lm1_output_types, str) else lm1_output_types
        self.lm2_output_types = [lm2_output_types] if isinstance(lm2_output_types, str) else lm2_output_types
        self.log_params()
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def save(self, save_dir, lm1_name='lm1', lm2_name='lm2'):
        """
        Saves the 2 language model weights and respective config_files in directories lm1 and lm2 within save_dir.

        :param save_dir: path to save to
        :type save_dir: Path
        """
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(Path.joinpath(save_dir, Path(lm1_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm1_name)))
        if not os.path.exists(Path.joinpath(save_dir, Path(lm2_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm2_name)))
        self.language_model1.save(Path.joinpath(save_dir, Path(lm1_name)))
        self.language_model2.save(Path.joinpath(save_dir, Path(lm2_name)))
        for i, ph in enumerate(self.prediction_heads):
            logger.info('prediction_head saving')
            ph.save(save_dir, i)

    @classmethod
    def load(cls, load_dir, device, strict=False, lm1_name='lm1', lm2_name='lm2', processor=None):
        """
        Loads a BiAdaptiveModel from a directory. The directory must contain:

        * directory "lm1_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * directory "lm2_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Token
        * special_tokens_map.json

        :param load_dir: location where adaptive model is stored
        :type load_dir: Path
        :param device: to which device we want to sent the model, either cpu or cuda
        :type device: torch.device
        :param lm1_name: the name to assign to the first loaded language model(for encoding queries)
        :type lm1_name: str
        :param lm2_name: the name to assign to the second loaded language model(for encoding context/passages)
        :type lm2_name: str
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of FARM.
        :type strict: bool
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        """
        if lm1_name:
            language_model1 = LanguageModel.load(os.path.join(load_dir, lm1_name))
        else:
            language_model1 = LanguageModel.load(load_dir)
        if lm2_name:
            language_model2 = LanguageModel.load(os.path.join(load_dir, lm2_name))
        else:
            language_model2 = LanguageModel.load(load_dir)
        ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=False, load_weights=False)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)
        model = cls(language_model1, language_model2, prediction_heads, 0.1, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)
        return model

    def logits_to_loss_per_head(self, logits, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: logits, can vary in shape and type, depending on task.
        :type logits: object
        :return: The per sample per prediciton head loss whose first two dimensions have length n_pred_heads, batch_size
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            assert hasattr(head, 'label_tensor_name'), f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model with the processor through either 'model.connect_heads_with_processor(processor.tasks)' or by passing the processor to the Adaptive Model?"
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits, global_step=None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param global_step: number of current training step
        :type global_step: int
        :param kwargs: placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :type kwargs: object
        :return loss: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
        """
        all_labels = []
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(self, **kwargs):
        """
        Push data through the whole model and returns logits. The data will propagate through
        the first language model and second language model based on the tensor names and both the
        encodings through each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to both the language models and prediction head(s).
        :return: all logits as torch.tensor or multiple tensors.
        """
        pooled_output = self.forward_lm(**kwargs)
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm1_out, lm2_out in zip(self.prediction_heads, self.lm1_output_types, self.lm2_output_types):
                if pooled_output[0] is not None:
                    if lm1_out == 'per_sequence' or lm1_out == 'per_sequence_continuous':
                        output1 = self.dropout1(pooled_output[0])
                    else:
                        raise ValueError('Unknown extraction strategy from BiAdaptive language_model1: {}'.format(lm1_out))
                else:
                    output1 = None
                if pooled_output[1] is not None:
                    if lm2_out == 'per_sequence' or lm2_out == 'per_sequence_continuous':
                        output2 = self.dropout2(pooled_output[1])
                    else:
                        raise ValueError('Unknown extraction strategy from BiAdaptive language_model2: {}'.format(lm2_out))
                else:
                    output2 = None
                embedding1, embedding2 = head(output1, output2)
                all_logits.append(tuple([embedding1, embedding2]))
        else:
            all_logits.append(pooled_output)
        return all_logits

    def forward_lm(self, **kwargs):
        """
        Forward pass for the BiAdaptive model.

        :param kwargs:
        :return: 2 tensors of pooled_output from the 2 language models
        """
        pooled_output = [None, None]
        if 'query_input_ids' in kwargs.keys():
            pooled_output1, hidden_states1 = self.language_model1(**kwargs)
            pooled_output[0] = pooled_output1
        if 'passage_input_ids' in kwargs.keys():
            pooled_output2, hidden_states2 = self.language_model2(**kwargs)
            pooled_output[1] = pooled_output2
        return tuple(pooled_output)

    def log_params(self):
        """
        Logs paramteres to generic logger MlLogger
        """
        params = {'lm1_type': self.language_model1.__class__.__name__, 'lm1_name': self.language_model1.name, 'lm1_output_types': ','.join(self.lm1_output_types), 'lm2_type': self.language_model2.__class__.__name__, 'lm2_name': self.language_model2.name, 'lm2_output_types': ','.join(self.lm2_output_types), 'prediction_heads': ','.join([head.__class__.__name__ for head in self.prediction_heads])}
        try:
            MlLogger.log_params(params)
        except Exception as e:
            logger.warning(f"ML logging didn't work: {e}")

    def verify_vocab_size(self, vocab_size1, vocab_size2):
        """ Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()"""
        model1_vocab_len = self.language_model1.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
        msg = f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model1_vocab_len}. If you added a custom vocabulary to the tokenizer, make sure to supply 'n_added_tokens' to LanguageModel.load() and BertStyleLM.load()"
        assert vocab_size1 == model1_vocab_len, msg
        model2_vocab_len = self.language_model2.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
        msg = f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model2_vocab_len}. If you added a custom vocabulary to the tokenizer, make sure to supply 'n_added_tokens' to LanguageModel.load() and BertStyleLM.load()"
        assert vocab_size2 == model2_vocab_len, msg

    def get_language(self):
        return self.language_model1.language, self.language_model2.language

    def convert_to_transformers(self):
        if len(self.prediction_heads) != 1:
            raise ValueError(f'Currently conversion only works for models with a SINGLE prediction head. Your model has {len(self.prediction_heads)}')
        if self.prediction_heads[0].model_type == 'text_similarity':
            if 'dpr' in self.language_model1.model.config.model_type or self.language_model1.model.config.name == 'DPRQuestionEncoder':
                transformers_model1 = DPRQuestionEncoder(config=self.language_model1.model.config)
            else:
                transformers_model1 = AutoModel.from_config(config=self.language_model1.model.config)
            if 'dpr' in self.language_model2.model.config.model_type or self.language_model2.model.config.name == 'DPRContextEncoder':
                transformers_model2 = DPRContextEncoder(config=self.language_model2.model.config)
            else:
                transformers_model2 = AutoModel.from_config(config=self.language_model2.model.config)
            setattr(transformers_model1, transformers_model1.base_model_prefix, getattr(self.language_model1.model, self.language_model1.model.base_model_prefix))
            setattr(transformers_model2, transformers_model2.base_model_prefix, getattr(self.language_model2.model, self.language_model2.model.base_model_prefix))
            logger.warning('No prediction head weights are required for DPR')
        else:
            raise NotImplementedError(f'FARM -> Transformers conversion is not supported yet for prediction heads of type {self.prediction_heads[0].model_type}')
        pass
        return transformers_model1, transformers_model2

    @classmethod
    def convert_from_transformers(cls, model_name_or_path1, model_name_or_path2, device, task_type, processor=None, similarity_function='dot_product'):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in FARM (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path1: local path of a saved model or name of a public one for Question Encoder
                                              Exemplary public names:
                                              - facebook/dpr-question_encoder-single-nq-base
                                              - deepset/bert-large-uncased-whole-word-masking-squad2
        :param model_name_or_path2: local path of a saved model or name of a public one for Context/Passage Encoder
                                      Exemplary public names:
                                      - facebook/dpr-ctx_encoder-single-nq-base
                                      - deepset/bert-large-uncased-whole-word-masking-squad2
        :param device: "cpu" or "cuda"
        :param task_type: 'text_similarity'
                          More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :return: AdaptiveModel
        """
        lm1 = LanguageModel.load(pretrained_model_name_or_path=model_name_or_path1, language_model_class='DPRQuestionEncoder')
        lm2 = LanguageModel.load(pretrained_model_name_or_path=model_name_or_path2, language_model_class='DPRContextEncoder')
        prediction_head = TextSimilarityHead(similarity_function=similarity_function)
        if task_type == 'text_similarity':
            bi_adaptive_model = cls(language_model1=lm1, language_model2=lm2, prediction_heads=[prediction_head], embeds_dropout_prob=0.1, lm1_output_types=['per_sequence'], lm2_output_types=['per_sequence'], device=device)
        else:
            raise NotImplementedError(f"Huggingface's transformer models of type {task_type} are not supported yet for BiAdaptive Models")
        if processor:
            bi_adaptive_model.connect_heads_with_processor(processor.tasks)
        return bi_adaptive_model


class Albert(LanguageModel):
    """
    An ALBERT model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    """

    def __init__(self):
        super(Albert, self).__init__()
        self.model = None
        self.name = 'albert'

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("albert-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        albert = cls()
        if 'farm_lm_name' in kwargs:
            albert.name = kwargs['farm_lm_name']
        else:
            albert.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = AlbertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            albert.model = AlbertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            albert.language = albert.model.config.language
        else:
            albert.model = AlbertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            albert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return albert

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the Albert model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids, token_type_ids=segment_ids, attention_mask=padding_mask)
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class Roberta(LanguageModel):
    """
    A roberta model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692

    """

    def __init__(self):
        super(Roberta, self).__init__()
        self.model = None
        self.name = 'roberta'

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        roberta = cls()
        if 'farm_lm_name' in kwargs:
            roberta.name = kwargs['farm_lm_name']
        else:
            roberta.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = RobertaConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            roberta.model = RobertaModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            roberta.language = roberta.model.config.language
        else:
            roberta.model = RobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return roberta

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the Roberta model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids, token_type_ids=segment_ids, attention_mask=padding_mask)
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class XLMRoberta(LanguageModel):
    """
    A roberta model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1907.11692

    """

    def __init__(self):
        super(XLMRoberta, self).__init__()
        self.model = None
        self.name = 'xlm_roberta'

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("xlm-roberta-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        xlm_roberta = cls()
        if 'farm_lm_name' in kwargs:
            xlm_roberta.name = kwargs['farm_lm_name']
        else:
            xlm_roberta.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = XLMRobertaConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            xlm_roberta.model = XLMRobertaModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            xlm_roberta.language = xlm_roberta.model.config.language
        else:
            xlm_roberta.model = XLMRobertaModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            xlm_roberta.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return xlm_roberta

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the XLMRoberta model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids, token_type_ids=segment_ids, attention_mask=padding_mask)
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class DistilBert(LanguageModel):
    """
    A DistilBERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - DistilBert doesn’t have token_type_ids, you don’t need to indicate which
    token belongs to which segment. Just separate your segments with the separation
    token tokenizer.sep_token (or [SEP])
    - Unlike the other BERT variants, DistilBert does not output the
    pooled_output. An additional pooler is initialized.

    """

    def __init__(self):
        super(DistilBert, self).__init__()
        self.model = None
        self.name = 'distilbert'
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("distilbert-base-german-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """
        distilbert = cls()
        if 'farm_lm_name' in kwargs:
            distilbert.name = kwargs['farm_lm_name']
        else:
            distilbert.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = DistilBertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            distilbert.model = DistilBertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            distilbert.language = distilbert.model.config.language
        else:
            distilbert.model = DistilBertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            distilbert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        config = distilbert.model.config
        config.summary_last_dropout = 0
        config.summary_type = 'first'
        config.summary_activation = 'tanh'
        distilbert.pooler = SequenceSummary(config)
        distilbert.pooler.apply(distilbert.model._init_weights)
        return distilbert

    def forward(self, input_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the DistilBERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids, attention_mask=padding_mask)
        pooled_output = self.pooler(output_tuple[0])
        if self.model.config.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.config.output_hidden_states = False


class XLNet(LanguageModel):
    """
    A XLNet model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1906.08237
    """

    def __init__(self):
        super(XLNet, self).__init__()
        self.model = None
        self.name = 'xlnet'
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("xlnet-base-cased" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        xlnet = cls()
        if 'farm_lm_name' in kwargs:
            xlnet.name = kwargs['farm_lm_name']
        else:
            xlnet.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = XLNetConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            xlnet.model = XLNetModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            xlnet.language = xlnet.model.config.language
        else:
            xlnet.model = XLNetModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            xlnet.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
            config = xlnet.model.config
        config.summary_last_dropout = 0
        xlnet.pooler = SequenceSummary(config)
        xlnet.pooler.apply(xlnet.model._init_weights)
        return xlnet

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the XLNet model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """
        output_tuple = self.model(input_ids, token_type_ids=segment_ids, attention_mask=padding_mask)
        pooled_output = self.pooler(output_tuple[0])
        if self.model.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.output_hidden_states = False


class EmbeddingConfig:
    """
    Config for Word Embeddings Models.
    Necessary to work with Bert and other LM style functionality
    """

    def __init__(self, name=None, embeddings_filename=None, vocab_filename=None, vocab_size=None, hidden_size=None, language=None, **kwargs):
        """
        :param name: Name of config
        :param embeddings_filename:
        :param vocab_filename:
        :param vocab_size:
        :param hidden_size:
        :param language:
        :param kwargs:
        """
        self.name = name
        self.embeddings_filename = embeddings_filename
        self.vocab_filename = vocab_filename
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.language = language
        if len(kwargs) > 0:
            logger.info(f'Passed unused params {str(kwargs)} to the EmbeddingConfig. Might not be a problem.')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, 'model_type'):
            output['model_type'] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


class EmbeddingModel:
    """
    Embedding Model that combines
    - Embeddings
    - Config Object
    - Vocab
    Necessary to work with Bert and other LM style functionality
    """

    def __init__(self, embedding_file, config_dict, vocab_file):
        """

        :param embedding_file: filename of embeddings. Usually in txt format, with the word and associated vector on each line
        :type embedding_file: str
        :param config_dict: dictionary containing config elements
        :type config_dict: dict
        :param vocab_file: filename of vocab, each line contains a word
        :type vocab_file: str
        """
        self.config = EmbeddingConfig(**config_dict)
        self.vocab = load_vocab(vocab_file)
        temp = wordembedding_utils.load_embedding_vectors(embedding_file=embedding_file, vocab=self.vocab)
        self.embeddings = torch.from_numpy(temp).float()
        assert '[UNK]' in self.vocab, 'No [UNK] symbol in Wordembeddingmodel! Aborting'
        self.unk_idx = self.vocab['[UNK]']

    def save(self, save_dir):
        save_name = Path(save_dir) / self.config.embeddings_filename
        embeddings = self.embeddings.cpu().numpy()
        with open(save_name, 'w') as f:
            for w, vec in tqdm(zip(self.vocab, embeddings), desc='Saving embeddings', total=embeddings.shape[0]):
                f.write(w + ' ' + ' '.join([('%.6f' % v) for v in vec]) + '\n')
        f.close()
        save_name = Path(save_dir) / self.config.vocab_filename
        with open(save_name, 'w') as f:
            for w in self.vocab:
                f.write(w + '\n')
        f.close()

    def resize_token_embeddings(self, new_num_tokens=None):
        temp = {}
        temp['num_embeddings'] = len(self.vocab)
        temp = DotMap(temp)
        return temp


class WordEmbedding_LM(LanguageModel):
    """
    A Language Model based only on word embeddings
    - Inside FARM, WordEmbedding Language Models must have a fixed vocabulary
    - Each (known) word in some text input is projected to its vector representation
    - Pooling operations can be applied for representing whole text sequences

    """

    def __init__(self):
        super(WordEmbedding_LM, self).__init__()
        self.model = None
        self.name = 'WordEmbedding_LM'
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * a local path of a model trained via FARM ("some_dir/farm_model")
        * the name of a remote model on s3

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        wordembedding_LM = cls()
        if 'farm_lm_name' in kwargs:
            wordembedding_LM.name = kwargs['farm_lm_name']
        else:
            wordembedding_LM.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = json.load(open(farm_lm_config, 'r'))
            farm_lm_model = Path(pretrained_model_name_or_path) / config['embeddings_filename']
            vocab_filename = Path(pretrained_model_name_or_path) / config['vocab_filename']
            wordembedding_LM.model = EmbeddingModel(embedding_file=str(farm_lm_model), config_dict=config, vocab_file=str(vocab_filename))
            wordembedding_LM.language = config.get('language', None)
        else:
            config_dict, resolved_vocab_file, resolved_model_file = wordembedding_utils.load_model(pretrained_model_name_or_path, **kwargs)
            model = EmbeddingModel(embedding_file=resolved_model_file, config_dict=config_dict, vocab_file=resolved_vocab_file)
            wordembedding_LM.model = model
            wordembedding_LM.language = model.config.language
        wordembedding_LM.pooler = lambda x: torch.mean(x, dim=0)
        return wordembedding_LM

    def save(self, save_dir):
        """
        Save the model embeddings and its config file so that it can be loaded again.
        # TODO make embeddings trainable and save trained embeddings
        # TODO save model weights as pytorch model bin for more efficient loading and saving
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        self.model.save(save_dir=save_dir)
        self.save_config(save_dir=save_dir)

    def forward(self, input_ids, **kwargs):
        """
        Perform the forward pass of the wordembedding model.
        This is just the mapping of words to their corresponding embeddings
        """
        sequence_output = []
        pooled_output = []
        for sample in input_ids:
            sample_embeddings = []
            for index in sample:
                sample_embeddings.append(self.model.embeddings[index])
            sample_embeddings = torch.stack(sample_embeddings)
            sequence_output.append(sample_embeddings)
            pooled_output.append(self.pooler(sample_embeddings))
        sequence_output = torch.stack(sequence_output)
        pooled_output = torch.stack(pooled_output)
        m = nn.BatchNorm1d(pooled_output.shape[1])
        if pooled_output.shape[0] > 1:
            pooled_output = m(pooled_output)
        return sequence_output, pooled_output

    def trim_vocab(self, token_counts, processor, min_threshold):
        """ Remove embeddings for rare tokens in your corpus (< `min_threshold` occurrences) to reduce model size"""
        logger.info(f'Removing tokens with less than {min_threshold} occurrences from model vocab')
        new_vocab = OrderedDict()
        valid_tok_indices = []
        cnt = 0
        old_num_emb = self.model.embeddings.shape[0]
        for token, tok_idx in self.model.vocab.items():
            if token_counts.get(token, 0) >= min_threshold or token in ('[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]'):
                new_vocab[token] = cnt
                valid_tok_indices.append(tok_idx)
                cnt += 1
        self.model.vocab = new_vocab
        self.model.embeddings = self.model.embeddings[valid_tok_indices, :]
        processor.tokenizer.vocab = self.model.vocab
        processor.tokenizer.ids_to_tokens = OrderedDict()
        for k, v in processor.tokenizer.vocab.items():
            processor.tokenizer.ids_to_tokens[v] = k
        logger.info(f'Reduced vocab from {old_num_emb} to {self.model.embeddings.shape[0]}')

    def normalize_embeddings(self, zero_mean=True, pca_removal=False, pca_n_components=300, pca_n_top_components=10, use_mean_vec_for_special_tokens=True, n_special_tokens=5):
        """ Normalize word embeddings as in https://arxiv.org/pdf/1808.06305.pdf
            (e.g. used for S3E Pooling of sentence embeddings)
            
        :param zero_mean: Whether to center embeddings via subtracting mean
        :type zero_mean: bool
        :param pca_removal: Whether to remove PCA components
        :type pca_removal: bool
        :param pca_n_components: Number of PCA components to use for fitting
        :type pca_n_components: int
        :param pca_n_top_components: Number of PCA components to remove
        :type pca_n_top_components: int
        :param use_mean_vec_for_special_tokens: Whether to replace embedding of special tokens with the mean embedding
        :type use_mean_vec_for_special_tokens: bool
        :param n_special_tokens: Number of special tokens like CLS, UNK etc. (used if `use_mean_vec_for_special_tokens`). 
                                 Note: We expect the special tokens to be the first `n_special_tokens` entries of the vocab.
        :type n_special_tokens: int
        :return: None
        """
        if zero_mean:
            logger.info('Removing mean from embeddings')
            mean_vec = torch.mean(self.model.embeddings, 0)
            self.model.embeddings = self.model.embeddings - mean_vec
            if use_mean_vec_for_special_tokens:
                self.model.embeddings[:n_special_tokens, :] = mean_vec
        if pca_removal:
            from sklearn.decomposition import PCA
            logger.info('Removing projections on top PCA components from embeddings (see https://arxiv.org/pdf/1808.06305.pdf)')
            pca = PCA(n_components=pca_n_components)
            pca.fit(self.model.embeddings.cpu().numpy())
            U1 = pca.components_
            explained_variance = pca.explained_variance_
            PVN_dims = pca_n_top_components
            for emb_idx in tqdm(range(self.model.embeddings.shape[0]), desc='Removing projections'):
                for pca_idx, u in enumerate(U1[0:PVN_dims]):
                    ratio = (explained_variance[pca_idx] - explained_variance[PVN_dims]) / explained_variance[pca_idx]
                    self.model.embeddings[emb_idx] = self.model.embeddings[emb_idx] - ratio * np.dot(u.transpose(), self.model.embeddings[emb_idx]) * u


class Electra(LanguageModel):
    """
    ELECTRA is a new pre-training approach which trains two transformer models:
    the generator and the discriminator. The generator replaces tokens in a sequence,
    and is therefore trained as a masked language model. The discriminator, which is
    the model we're interested in, tries to identify which tokens were replaced by
    the generator in the sequence.

    The ELECTRA model here wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.

    NOTE:
    - Electra does not output the pooled_output. An additional pooler is initialized.

    """

    def __init__(self):
        super(Electra, self).__init__()
        self.model = None
        self.name = 'electra'
        self.pooler = None

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("google/electra-base-discriminator" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """
        electra = cls()
        if 'farm_lm_name' in kwargs:
            electra.name = kwargs['farm_lm_name']
        else:
            electra.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = ElectraConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            electra.model = ElectraModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            electra.language = electra.model.config.language
        else:
            electra.model = ElectraModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            electra.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        config = electra.model.config
        config.summary_last_dropout = 0
        config.summary_type = 'first'
        config.summary_activation = 'gelu'
        config.summary_use_proj = False
        electra.pooler = SequenceSummary(config)
        electra.pooler.apply(electra.model._init_weights)
        return electra

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the ELECTRA model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids, token_type_ids=segment_ids, attention_mask=padding_mask)
        pooled_output = self.pooler(output_tuple[0])
        if self.model.config.output_hidden_states == True:
            sequence_output, all_hidden_states = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output
        else:
            sequence_output = output_tuple[0]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.config.output_hidden_states = False


class Camembert(Roberta):
    """
    A Camembert model that wraps the HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    """

    def __init__(self):
        super(Camembert, self).__init__()
        self.model = None
        self.name = 'camembert'

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a language model either by supplying

        * the name of a remote model on s3 ("camembert-base" ...)
        * or a local path of a model trained via transformers ("some_dir/huggingface_model")
        * or a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: name or path of a model
        :param language: (Optional) Name of language the model was trained for (e.g. "german").
                         If not supplied, FARM will try to infer it from the model name.
        :return: Language Model

        """
        camembert = cls()
        if 'farm_lm_name' in kwargs:
            camembert.name = kwargs['farm_lm_name']
        else:
            camembert.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            config = CamembertConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            camembert.model = CamembertModel.from_pretrained(farm_lm_model, config=config, **kwargs)
            camembert.language = camembert.model.config.language
        else:
            camembert.model = CamembertModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            camembert.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return camembert


class DPRQuestionEncoder(LanguageModel):
    """
    A DPRQuestionEncoder model that wraps HuggingFace's implementation
    """

    def __init__(self):
        super(DPRQuestionEncoder, self).__init__()
        self.model = None
        self.name = 'dpr_question_encoder'

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("facebook/dpr-question_encoder-single-nq-base" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder
        :type pretrained_model_name_or_path: str
        """
        dpr_question_encoder = cls()
        if 'farm_lm_name' in kwargs:
            dpr_question_encoder.name = kwargs['farm_lm_name']
        else:
            dpr_question_encoder.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            original_model_config = AutoConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            if original_model_config.model_type == 'dpr':
                dpr_config = transformers.DPRConfig.from_pretrained(farm_lm_config)
                dpr_question_encoder.model = transformers.DPRQuestionEncoder.from_pretrained(farm_lm_model, config=dpr_config, **kwargs)
            else:
                if original_model_config.model_type != 'bert':
                    logger.warning(f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders.Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_question_encoder.model = transformers.DPRQuestionEncoder(config=transformers.DPRConfig(**original_config_dict))
                language_model_class = cls.get_language_model_class(farm_lm_config)
                dpr_question_encoder.model.base_model.bert_model = cls.subclasses[language_model_class].load(str(pretrained_model_name_or_path)).model
            dpr_question_encoder.language = dpr_question_encoder.model.config.language
        else:
            original_model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if original_model_config.model_type == 'dpr':
                dpr_question_encoder.model = transformers.DPRQuestionEncoder.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            else:
                if original_model_config.model_type != 'bert':
                    logger.warning(f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders.Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_question_encoder.model = transformers.DPRQuestionEncoder(config=transformers.DPRConfig(**original_config_dict))
                dpr_question_encoder.model.base_model.bert_model = AutoModel.from_pretrained(str(pretrained_model_name_or_path), **original_config_dict)
            dpr_question_encoder.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return dpr_question_encoder

    def save(self, save_dir, state_dict=None):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        :param state_dict: A dictionary containing a whole state of the module including names of layers. By default, the unchanged state dict of the module is used
        :type state_dict: Optional[dict]
        """
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if self.model.config.model_type != 'dpr' and model_to_save.base_model_prefix.startswith('question_'):
            state_dict = model_to_save.state_dict()
            keys = state_dict.keys()
            for key in list(keys):
                new_key = key
                if key.startswith('question_encoder.bert_model.model.'):
                    new_key = key.split('_encoder.bert_model.model.', 1)[1]
                elif key.startswith('question_encoder.bert_model.'):
                    new_key = key.split('_encoder.bert_model.', 1)[1]
                state_dict[new_key] = state_dict.pop(key)
        super(DPRQuestionEncoder, self).save(save_dir=save_dir, state_dict=state_dict)

    def forward(self, query_input_ids, query_segment_ids, query_attention_mask, **kwargs):
        """
        Perform the forward pass of the DPRQuestionEncoder model.

        :param query_input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type query_input_ids: torch.Tensor
        :param query_segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type query_segment_ids: torch.Tensor
        :param query_attention_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :type query_attention_mask: torch.Tensor
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids=query_input_ids, token_type_ids=query_segment_ids, attention_mask=query_attention_mask, return_dict=True)
        if self.model.question_encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None

    def enable_hidden_states_output(self):
        self.model.question_encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.question_encoder.config.output_hidden_states = False


class DPRContextEncoder(LanguageModel):
    """
    A DPRContextEncoder model that wraps HuggingFace's implementation
    """

    def __init__(self):
        super(DPRContextEncoder, self).__init__()
        self.model = None
        self.name = 'dpr_context_encoder'

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("facebook/dpr-ctx_encoder-single-nq-base" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRContextEncoder
        :type pretrained_model_name_or_path: str
        """
        dpr_context_encoder = cls()
        if 'farm_lm_name' in kwargs:
            dpr_context_encoder.name = kwargs['farm_lm_name']
        else:
            dpr_context_encoder.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            original_model_config = AutoConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            if original_model_config.model_type == 'dpr':
                dpr_config = transformers.DPRConfig.from_pretrained(farm_lm_config)
                dpr_context_encoder.model = transformers.DPRContextEncoder.from_pretrained(farm_lm_model, config=dpr_config, **kwargs)
            else:
                if original_model_config.model_type != 'bert':
                    logger.warning(f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders.Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_context_encoder.model = transformers.DPRContextEncoder(config=transformers.DPRConfig(**original_config_dict))
                language_model_class = cls.get_language_model_class(farm_lm_config)
                dpr_context_encoder.model.base_model.bert_model = cls.subclasses[language_model_class].load(str(pretrained_model_name_or_path)).model
            dpr_context_encoder.language = dpr_context_encoder.model.config.language
        else:
            original_model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if original_model_config.model_type == 'dpr':
                dpr_context_encoder.model = transformers.DPRContextEncoder.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            else:
                if original_model_config.model_type != 'bert':
                    logger.warning(f"Using a model of type '{original_model_config.model_type}' which might be incompatible with DPR encoders.Bert based encoders are supported that need input_ids,token_type_ids,attention_mask as input tensors.")
                original_config_dict = vars(original_model_config)
                original_config_dict.update(kwargs)
                dpr_context_encoder.model = transformers.DPRContextEncoder(config=transformers.DPRConfig(**original_config_dict))
                dpr_context_encoder.model.base_model.bert_model = AutoModel.from_pretrained(str(pretrained_model_name_or_path), **original_config_dict)
            dpr_context_encoder.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return dpr_context_encoder

    def save(self, save_dir, state_dict=None):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        :param state_dict: A dictionary containing a whole state of the module including names of layers. By default, the unchanged state dict of the module is used
        :type state_dict: Optional[dict]
        """
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if self.model.config.model_type != 'dpr' and model_to_save.base_model_prefix.startswith('ctx_'):
            state_dict = model_to_save.state_dict()
            keys = state_dict.keys()
            for key in list(keys):
                new_key = key
                if key.startswith('ctx_encoder.bert_model.model.'):
                    new_key = key.split('_encoder.bert_model.model.', 1)[1]
                elif key.startswith('ctx_encoder.bert_model.'):
                    new_key = key.split('_encoder.bert_model.', 1)[1]
                state_dict[new_key] = state_dict.pop(key)
        super(DPRContextEncoder, self).save(save_dir=save_dir, state_dict=state_dict)

    def forward(self, passage_input_ids, passage_segment_ids, passage_attention_mask, **kwargs):
        """
        Perform the forward pass of the DPRContextEncoder model.

        :param passage_input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len]
        :type passage_input_ids: torch.Tensor
        :param passage_segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len]
        :type passage_segment_ids: torch.Tensor
        :param passage_attention_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size,  number_of_hard_negative_passages, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        max_seq_len = passage_input_ids.shape[-1]
        passage_input_ids = passage_input_ids.view(-1, max_seq_len)
        passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)
        passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)
        output_tuple = self.model(input_ids=passage_input_ids, token_type_ids=passage_segment_ids, attention_mask=passage_attention_mask, return_dict=True)
        if self.model.ctx_encoder.config.output_hidden_states == True:
            pooled_output, all_hidden_states = output_tuple.pooler_output, output_tuple.hidden_states
            return pooled_output, all_hidden_states
        else:
            pooled_output = output_tuple.pooler_output
            return pooled_output, None

    def enable_hidden_states_output(self):
        self.model.ctx_encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.ctx_encoder.config.output_hidden_states = False


class BigBird(LanguageModel):
    """
    A BERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805

    """

    def __init__(self):
        super(BigBird, self).__init__()
        self.model = None
        self.name = 'big_bird'

    @classmethod
    def from_scratch(cls, vocab_size, name='big_bird', language='en'):
        big_bird = cls()
        big_bird.name = name
        big_bird.language = language
        config = BigBirdConfig(vocab_size=vocab_size)
        big_bird.model = BigBirdModel(config)
        return big_bird

    @classmethod
    def load(cls, pretrained_model_name_or_path, language=None, **kwargs):
        """
        Load a pretrained model by supplying

        * the name of a remote model on s3 ("bert-base-cased" ...)
        * OR a local path of a model trained via transformers ("some_dir/huggingface_model")
        * OR a local path of a model trained via FARM ("some_dir/farm_model")

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str

        """
        big_bird = cls()
        if 'farm_lm_name' in kwargs:
            big_bird.name = kwargs['farm_lm_name']
        else:
            big_bird.name = pretrained_model_name_or_path
        farm_lm_config = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(farm_lm_config):
            big_bird_config = BigBirdConfig.from_pretrained(farm_lm_config)
            farm_lm_model = Path(pretrained_model_name_or_path) / 'language_model.bin'
            big_bird.model = BigBirdModel.from_pretrained(farm_lm_model, config=big_bird_config, **kwargs)
            big_bird.language = big_bird.model.config.language
        else:
            big_bird.model = BigBirdModel.from_pretrained(str(pretrained_model_name_or_path), **kwargs)
            big_bird.language = cls._get_or_infer_language_from_name(language, pretrained_model_name_or_path)
        return big_bird

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
        """
        Perform the forward pass of the BERT model.

        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.

        """
        output_tuple = self.model(input_ids, token_type_ids=segment_ids, attention_mask=padding_mask)
        if self.model.encoder.config.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.config.output_hidden_states = False


class WrappedDataParallel(DataParallel):
    """
    A way of adapting attributes of underlying class to parallel mode. See: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see: https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class WrappedDDP(DistributedDataParallel):
    """
    A way of adapting attributes of underlying class to distributed mode. Same as in WrappedDataParallel above.
    Even when using distributed on a single machine with multiple GPUs, apex can speed up training significantly.
    Distributed code must be launched with "python -m torch.distributed.launch --nproc_per_node=1 run_script.py"
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class RegressionHead(PredictionHead):

    def __init__(self, layer_dims=[768, 1], task_name='regression', **kwargs):
        super(RegressionHead, self).__init__()
        self.layer_dims = layer_dims
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = 2
        self.ph_output_type = 'per_sequence_continuous'
        self.model_type = 'regression'
        self.loss_fct = MSELoss(reduction='none')
        self.task_name = task_name
        self.generate_config()

    def forward(self, x):
        logits = self.feed_forward(x)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        return self.loss_fct(logits, label_ids.float())

    def logits_to_preds(self, logits, **kwargs):
        preds = logits.cpu().numpy()
        preds = [(x * self.label_list[1] + self.label_list[0]) for x in preds]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        label_ids = [(x * self.label_list[1] + self.label_list[0]) for x in label_ids]
        return label_ids

    def formatted_preds(self, logits, samples, **kwargs):
        preds = self.logits_to_preds(logits)
        contexts = [sample.clear_text['text'] for sample in samples]
        res = {'task': 'regression', 'task_name': self.task_name, 'predictions': []}
        for pred, context in zip(preds, contexts):
            res['predictions'].append({'context': f'{context}', 'pred': pred[0]})
        return res


class TextClassificationHead(PredictionHead):

    def __init__(self, layer_dims=None, num_labels=None, class_weights=None, loss_ignore_index=-100, loss_reduction='none', task_name='text_classification', **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_ignore_index:
        :param loss_reduction:
        :param task_name:
        :param kwargs:
        """
        super(TextClassificationHead, self).__init__()
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning('`layer_dims` will be deprecated in future releases')
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError('Please supply `num_labels` to define output dim of prediction head')
        self.num_labels = self.layer_dims[-1]
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.info(f'Prediction head initialized with size {self.layer_dims}')
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = 'per_sequence'
        self.model_type = 'text_classification'
        self.task_name = task_name
        if type(class_weights) is np.ndarray and class_weights.ndim != 1:
            raise ValueError('When you pass `class_weights` as `np.ndarray` it must have 1 dimension! You provided {} dimensions.'.format(class_weights.ndim))
        self.class_weights = class_weights
        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None
        self.loss_fct = CrossEntropyLoss(weight=balanced_weights, reduction=loss_reduction, ignore_index=loss_ignore_index)
        if 'label_list' in kwargs:
            self.label_list = kwargs['label_list']
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, **kwargs):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public name:
                                              - deepset/bert-base-german-cased-hatespeech-GermEval18Coarse

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            head = super(TextClassificationHead, cls).load(pretrained_model_name_or_path)
        else:
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, revision=revision, **kwargs)
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.id2label)])
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            head.label_list = list(full_model.config.id2label.values())
            del full_model
        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids
        return self.loss_fct(logits, label_ids.view(-1))

    def logits_to_probs(self, logits, return_class_probs, **kwargs):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        if return_class_probs:
            probs = probs
        else:
            probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        logits = logits.cpu().numpy()
        pred_ids = logits.argmax(1)
        preds = [self.label_list[int(x)] for x in pred_ids]
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        try:
            labels = [self.label_list[int(x)] for x in label_ids]
        except TypeError:
            labels = [self.label_list[int(x[0])] for x in label_ids]
        return labels

    def formatted_preds(self, logits=None, preds=None, samples=None, return_class_probs=False, **kwargs):
        """ Like QuestionAnsweringHead.formatted_preds(), this fn can operate on either logits or preds. This
        is needed since at inference, the order of operations is very different depending on whether we are performing
        aggregation or not (compare Inferencer._get_predictions() vs Inferencer._get_predictions_and_aggregate())"""
        assert logits is not None or preds is not None
        if logits is not None:
            preds = self.logits_to_preds(logits)
            probs = self.logits_to_probs(logits, return_class_probs)
        else:
            probs = [None] * len(preds)
        try:
            contexts = [sample.clear_text['text'] for sample in samples]
        except KeyError:
            contexts = [(sample.clear_text['question_text'] + ' | ' + sample.clear_text['passage_text']) for sample in samples]
        contexts_b = [sample.clear_text['text_b'] for sample in samples if 'text_b' in sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ['|'.join([a, b]) for a, b in zip(contexts, contexts_b)]
        res = {'task': 'text_classification', 'task_name': self.task_name, 'predictions': []}
        for pred, prob, context in zip(preds, probs, contexts):
            if not return_class_probs:
                pred_dict = {'start': None, 'end': None, 'context': f'{context}', 'label': f'{pred}', 'probability': prob}
            else:
                pred_dict = {'start': None, 'end': None, 'context': f'{context}', 'label': 'class_probabilities', 'probability': prob}
            res['predictions'].append(pred_dict)
        return res


class MultiLabelTextClassificationHead(PredictionHead):

    def __init__(self, layer_dims=None, num_labels=None, class_weights=None, loss_reduction='none', task_name='text_classification', pred_threshold=0.5, **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param class_weights:
        :param loss_reduction:
        :param task_name:
        :param pred_threshold:
        :param kwargs:
        """
        super(MultiLabelTextClassificationHead, self).__init__()
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning('`layer_dims` will be deprecated in future releases')
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError('Please supply `num_labels` to define output dim of prediction head')
        self.num_labels = self.layer_dims[-1]
        logger.info(f'Prediction head initialized with size {self.layer_dims}')
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.ph_output_type = 'per_sequence'
        self.model_type = 'multilabel_text_classification'
        self.task_name = task_name
        self.class_weights = class_weights
        self.pred_threshold = pred_threshold
        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            self.balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            self.balanced_weights = None
        self.loss_fct = BCEWithLogitsLoss(pos_weight=self.balanced_weights, reduction=loss_reduction)
        self.generate_config()

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        loss = self.loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1, self.num_labels))
        per_sample_loss = loss.mean(1)
        return per_sample_loss

    def logits_to_probs(self, logits, **kwargs):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits)
        probs = probs.cpu().numpy()
        return probs

    def logits_to_preds(self, logits, **kwargs):
        probs = self.logits_to_probs(logits)
        pred_ids = [np.where(row > self.pred_threshold)[0] for row in probs]
        preds = []
        for row in pred_ids:
            preds.append([self.label_list[int(x)] for x in row])
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        label_ids = [np.where(row == 1)[0] for row in label_ids]
        labels = []
        for row in label_ids:
            labels.append([self.label_list[int(x)] for x in row])
        return labels

    def formatted_preds(self, logits, samples, **kwargs):
        preds = self.logits_to_preds(logits)
        probs = self.logits_to_probs(logits)
        contexts = [sample.clear_text['text'] for sample in samples]
        res = {'task': 'text_classification', 'task_name': self.task_name, 'predictions': []}
        for pred, prob, context in zip(preds, probs, contexts):
            res['predictions'].append({'start': None, 'end': None, 'context': f'{context}', 'label': f'{pred}', 'probability': prob})
        return res


def convert_iob_to_simple_tags(preds, spans, probs):
    contains_named_entity = len([x for x in preds if 'B-' in x]) != 0
    simple_tags = []
    merged_spans = []
    tag_probs = []
    open_tag = False
    for pred, span, prob in zip(preds, spans, probs):
        if not ('B-' in pred or 'I-' in pred):
            if open_tag:
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                tag_probs.append(prob)
                open_tag = False
            continue
        elif 'B-' in pred:
            if open_tag:
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                tag_probs.append(prob)
            cur_tag = pred.replace('B-', '')
            cur_span = span
            open_tag = True
        elif 'I-' in pred:
            this_tag = pred.replace('I-', '')
            if open_tag and this_tag == cur_tag:
                cur_span = cur_span[0], span[1]
            elif open_tag:
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                tag_probs.append(prob)
                open_tag = False
    if open_tag:
        merged_spans.append(cur_span)
        simple_tags.append(cur_tag)
        tag_probs.append(prob)
        open_tag = False
    if contains_named_entity and len(simple_tags) == 0:
        raise Exception('Predicted Named Entities lost when converting from IOB to simple tags. Please check the formatof the training data adheres to either adheres to IOB2 format or is converted when read_ner_file() is called.')
    return simple_tags, merged_spans, tag_probs


class TokenClassificationHead(PredictionHead):

    def __init__(self, layer_dims=None, num_labels=None, task_name='ner', **kwargs):
        """
        :param layer_dims: The size of the layers in the feed forward component. The feed forward will have as many layers as there are ints in this list. This param will be deprecated in future
        :type layer_dims: list
        :param num_labels: The numbers of labels. Use to set the size of the final layer in the feed forward component. It is recommended to only set num_labels or layer_dims, not both.
        :type num_labels: int
        :param task_name:
        :param kwargs:
        """
        super(TokenClassificationHead, self).__init__()
        if layer_dims:
            self.layer_dims = layer_dims
            logger.warning('`layer_dims` will be deprecated in future releases')
        elif num_labels:
            self.layer_dims = [768, num_labels]
        else:
            raise ValueError('Please supply `num_labels` to define output dim of prediction head')
        self.num_labels = self.layer_dims[-1]
        logger.info(f'Prediction head initialized with size {self.layer_dims}')
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.loss_fct = CrossEntropyLoss(reduction='none')
        self.ph_output_type = 'per_token'
        self.model_type = 'token_classification'
        self.task_name = task_name
        if 'label_list' in kwargs:
            self.label_list = kwargs['label_list']
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, **kwargs):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased-finetuned-conll03-english)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased-finetuned-conll03-english

                                              See https://huggingface.co/models for full list

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            head = super(TokenClassificationHead, cls).load(pretrained_model_name_or_path)
        else:
            full_model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path, revision=revision, **kwargs)
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.label2id)])
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            head.label_list = list(full_model.config.id2label.values())
            head.generate_config()
            del full_model
        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, initial_mask, padding_mask=None, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        active_loss = padding_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = self.loss_fct(active_logits, active_labels)
        return loss

    def logits_to_preds(self, logits, initial_mask, **kwargs):
        preds_word_all = []
        preds_tokens = torch.argmax(logits, dim=2)
        preds_token = preds_tokens.detach().cpu().numpy()
        initial_mask = initial_mask.detach().cpu().numpy()
        for idx, im in enumerate(initial_mask):
            preds_t = preds_token[idx]
            preds_word_id = self.initial_token_only(preds_t, initial_mask=im)
            preds_word = [self.label_list[pwi] for pwi in preds_word_id]
            preds_word_all.append(preds_word)
        return preds_word_all

    def logits_to_probs(self, logits, initial_mask, return_class_probs, **kwargs):
        softmax = torch.nn.Softmax(dim=2)
        token_probs = softmax(logits)
        if return_class_probs:
            token_probs = token_probs
        else:
            token_probs = torch.max(token_probs, dim=2)[0]
        token_probs = token_probs.cpu().numpy()
        all_probs = []
        initial_mask = initial_mask.detach().cpu().numpy()
        for idx, im in enumerate(initial_mask):
            probs_t = token_probs[idx]
            probs_words = self.initial_token_only(probs_t, initial_mask=im)
            all_probs.append(probs_words)
        return all_probs

    def prepare_labels(self, initial_mask, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        labels_all = []
        label_ids = label_ids.cpu().numpy()
        for label_ids_one_sample, initial_mask_one_sample in zip(label_ids, initial_mask):
            label_ids = self.initial_token_only(label_ids_one_sample, initial_mask_one_sample)
            labels = [self.label_list[l] for l in label_ids]
            labels_all.append(labels)
        return labels_all

    @staticmethod
    def initial_token_only(seq, initial_mask):
        ret = []
        for init, s in zip(initial_mask, seq):
            if init:
                ret.append(s)
        return ret

    def formatted_preds(self, logits, initial_mask, samples, return_class_probs=False, **kwargs):
        preds = self.logits_to_preds(logits, initial_mask)
        probs = self.logits_to_probs(logits, initial_mask, return_class_probs)
        spans = [s.tokenized['word_spans'] for s in samples]
        res = {'task': 'ner', 'task_name': self.task_name, 'predictions': []}
        for preds_seq, probs_seq, sample, spans_seq in zip(preds, probs, samples, spans):
            if any(tag.startswith(('I-', 'B-')) for tag in preds_seq):
                tags, spans_seq, tag_probs = convert_iob_to_simple_tags(preds_seq, spans_seq, probs_seq)
            elif any(tag == 'O' for tag in preds_seq):
                logger.warning("Your token labels include 'O' which will be ignored when returing inference results!")
                o_indices = [i for i, tag in enumerate(preds_seq) if tag == 'O']
                tags = [tag for i, tag in enumerate(preds_seq) if i not in o_indices]
                spans_seq = [span for i, span in enumerate(spans_seq) if i not in o_indices]
                tag_probs = [probability for i, probability in enumerate(probs_seq) if i not in o_indices]
            else:
                logger.warning("""The token labels you are using are not IOB2 formatted. 
                                In that case farm will format prediction results for every input token.
                                Use 'O' as a label for tokens that are outside of any entity to ignore them.""")
                tags, tag_probs = preds_seq, probs_seq
            seq_res = []
            for tag, tag_prob, span in zip(tags, tag_probs, spans_seq):
                context = sample.clear_text['text'][span[0]:span[1]]
                seq_res.append({'start': span[0], 'end': span[1], 'context': f'{context}', 'label': f'{tag}', 'probability': tag_prob})
            res['predictions'].append(seq_res)
        return res


class BertLMHead(PredictionHead):

    def __init__(self, hidden_size, vocab_size, hidden_act='gelu', task_name='lm', **kwargs):
        super(BertLMHead, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.num_labels = vocab_size
        self.layer_dims = [hidden_size, vocab_size]
        self.ph_output_type = 'per_token'
        self.model_type = 'language_modelling'
        self.task_name = task_name
        self.generate_config()
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.transform_act_fn = ACT2FN[self.hidden_act]
        self.LayerNorm = BertLayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, n_added_tokens=0, **kwargs):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            if n_added_tokens != 0:
                raise NotImplementedError('Custom vocab not yet supported for model loading from FARM files')
            head = super(BertLMHead, cls).load(pretrained_model_name_or_path)
        else:
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path, revision=revision, **kwargs)
            vocab_size = bert_with_lm.config.vocab_size + n_added_tokens
            head = cls(hidden_size=bert_with_lm.config.hidden_size, vocab_size=vocab_size, hidden_act=bert_with_lm.config.hidden_act)
            head.dense.load_state_dict(bert_with_lm.cls.predictions.transform.dense.state_dict())
            head.LayerNorm.load_state_dict(bert_with_lm.cls.predictions.transform.LayerNorm.state_dict())
            if n_added_tokens == 0:
                bias_params = bert_with_lm.cls.predictions.bias
            else:
                bias_params = torch.nn.Parameter(torch.cat([bert_with_lm.cls.predictions.bias, torch.zeros(n_added_tokens)]))
            head.bias.data.copy_(bias_params)
            del bert_with_lm
            del bias_params
        return head

    def set_shared_weights(self, shared_embedding_weights):
        self.decoder.weight = shared_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        lm_logits = self.decoder(hidden_states) + self.bias
        return lm_logits

    def logits_to_loss(self, logits, **kwargs):
        lm_label_ids = kwargs.get(self.label_tensor_name)
        batch_size = lm_label_ids.shape[0]
        masked_lm_loss = self.loss_fct(logits.view(-1, self.num_labels), lm_label_ids.view(-1))
        per_sample_loss = masked_lm_loss.view(-1, batch_size).mean(dim=0)
        return per_sample_loss

    def logits_to_preds(self, logits, **kwargs):
        lm_label_ids = kwargs.get(self.label_tensor_name).cpu().numpy()
        lm_preds_ids = logits.argmax(2).cpu().numpy()
        lm_preds_ids[lm_label_ids == -1] = -1
        lm_preds_ids = lm_preds_ids.tolist()
        preds = []
        for pred_ids_for_sequence in lm_preds_ids:
            preds.append([self.label_list[int(x)] for x in pred_ids_for_sequence if int(x) != -1])
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy().tolist()
        labels = []
        for ids_for_sequence in label_ids:
            labels.append([self.label_list[int(x)] for x in ids_for_sequence if int(x) != -1])
        return labels


class NextSentenceHead(TextClassificationHead):
    """
    Almost identical to a TextClassificationHead. Only difference: we can load the weights from
     a pretrained language model that was saved in the Transformers style (all in one model).
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g.bert-base-cased)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - bert-base-cased

                                              See https://huggingface.co/models for full list

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            head = super(NextSentenceHead, cls).load(pretrained_model_name_or_path)
        else:
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path, **kwargs)
            head = cls(layer_dims=[bert_with_lm.config.hidden_size, 2], loss_ignore_index=-1, task_name='nextsentence')
            head.feed_forward.feed_forward[0].load_state_dict(bert_with_lm.cls.seq_relationship.state_dict())
            del bert_with_lm
        return head


class QACandidate:
    """
    A single QA candidate answer.
    """

    def __init__(self, answer_type: str, score: str, offset_answer_start: int, offset_answer_end: int, offset_unit: str, aggregation_level: str, probability: float=None, n_passages_in_doc: int=None, passage_id: str=None, confidence: float=None):
        """
        :param answer_type: The category that this answer falls into e.g. "no_answer", "yes", "no" or "span"
        :param score: The score representing the model's confidence of this answer
        :param offset_answer_start: The index of the start of the answer span (whether it is char or tok is stated in self.offset_unit)
        :param offset_answer_end: The index of the start of the answer span (whether it is char or tok is stated in self.offset_unit)
        :param offset_unit: States whether the offsets refer to character or token indices
        :param aggregation_level: States whether this candidate and its indices are on a passage level (pre aggregation) or on a document level (post aggregation)
        :param probability: The probability the model assigns to the answer
        :param n_passages_in_doc: Number of passages that make up the document
        :param passage_id: The id of the passage which contains this candidate answer
        :param confidence: The (calibrated) confidence score representing the model's predicted accuracy of the index of the start of the answer span
        """
        self.answer_type = answer_type
        self.score = score
        self.probability = probability
        self.answer = None
        self.offset_answer_start = offset_answer_start
        self.offset_answer_end = offset_answer_end
        self.answer_support = None
        self.offset_answer_support_start = None
        self.offset_answer_support_end = None
        self.context_window = None
        self.offset_context_window_start = None
        self.offset_context_window_end = None
        self.offset_unit = offset_unit
        self.aggregation_level = aggregation_level
        self.n_passages_in_doc = n_passages_in_doc
        self.passage_id = passage_id
        self.confidence = confidence
        self.meta = None

    def set_context_window(self, context_window_size, clear_text):
        window_str, start_ch, end_ch = self._create_context_window(context_window_size, clear_text)
        self.context_window = window_str
        self.offset_context_window_start = start_ch
        self.offset_context_window_end = end_ch

    def set_answer_string(self, token_offsets, document_text):
        pred_str, self.offset_answer_start, self.offset_answer_end = self._span_to_string(token_offsets, document_text)
        self.offset_unit = 'char'
        self._add_answer(pred_str)

    def _add_answer(self, string):
        """ Set the answer string. This method will check that the answer given is valid given the start
        and end indices that are stored in the object. """
        if string == '':
            self.answer = 'no_answer'
            if self.offset_answer_start != 0 or self.offset_answer_end != 0:
                logger.error(f'Both start and end offsets should be 0: \n{self.offset_answer_start}, {self.offset_answer_end} with a no_answer. ')
        else:
            self.answer = string
            if self.offset_answer_end - self.offset_answer_start <= 0:
                logger.error(f'End offset comes before start offset: \n({self.offset_answer_start}, {self.offset_answer_end}) with a span answer. ')
            elif self.offset_answer_end <= 0:
                logger.error(f'Invalid end offset: \n({self.offset_answer_start}, {self.offset_answer_end}) with a span answer. ')

    def _create_context_window(self, context_window_size, clear_text):
        """
        Extract from the clear_text a window that contains the answer and (usually) some amount of text on either
        side of the answer. Useful for cases where the answer and its surrounding context needs to be
        displayed in a UI. If the self.context_window_size is smaller than the extracted answer, it will be
        enlarged so that it can contain the answer

        :param context_window_size: The size of the context window to be generated. Note that the window size may be increased if the answer is longer.
        :param clear_text: The text from which the answer is extracted
        :return:
        """
        if self.offset_answer_start == 0 and self.offset_answer_end == 0:
            return '', 0, 0
        else:
            len_ans = self.offset_answer_end - self.offset_answer_start
            context_window_size = max(context_window_size, len_ans + 1)
            len_text = len(clear_text)
            midpoint = int(len_ans / 2) + self.offset_answer_start
            half_window = int(context_window_size / 2)
            window_start_ch = midpoint - half_window
            window_end_ch = midpoint + half_window
            overhang_start = max(0, -window_start_ch)
            overhang_end = max(0, window_end_ch - len_text)
            window_start_ch -= overhang_end
            window_start_ch = max(0, window_start_ch)
            window_end_ch += overhang_start
            window_end_ch = min(len_text, window_end_ch)
        window_str = clear_text[window_start_ch:window_end_ch]
        return window_str, window_start_ch, window_end_ch

    def _span_to_string(self, token_offsets: List[int], clear_text: str):
        """
        Generates a string answer span using self.offset_answer_start and self.offset_answer_end. If the candidate
        is a no answer, an empty string is returned

        :param token_offsets: A list of ints which give the start character index of the corresponding token
        :param clear_text: The text from which the answer span is to be extracted
        :return: The string answer span, followed by the start and end character indices
        """
        if self.offset_unit != 'token':
            logger.error(f'QACandidate needs to have self.offset_unit=token before calling _span_to_string() (id = {self.id})')
        start_t = self.offset_answer_start
        end_t = self.offset_answer_end
        if start_t == -1 and end_t == -1:
            return '', 0, 0
        n_tokens = len(token_offsets)
        end_t += 1
        end_t = min(end_t, n_tokens)
        start_ch = int(token_offsets[start_t])
        if end_t == n_tokens:
            end_ch = len(clear_text)
        else:
            end_ch = token_offsets[end_t]
        final_text = clear_text[start_ch:end_ch]
        if len(final_text.strip()) > 0:
            final_text = final_text.strip()
        else:
            return '', 0, 0
        end_ch = int(start_ch + len(final_text))
        return final_text, start_ch, end_ch

    def add_cls(self, predicted_class: str):
        """
        Adjust the final QA prediction depending on the prediction of the classification head (e.g. for binary answers in NQ)
        Currently designed so that the QA head's prediction will always be preferred over the Classification head

        :param predicted_class: The predicted class e.g. "yes", "no", "no_answer", "span"
        """
        if predicted_class in ['yes', 'no'] and self.answer != 'no_answer':
            self.answer_support = self.answer
            self.answer = predicted_class
            self.answer_type = predicted_class
            self.offset_answer_support_start = self.offset_answer_start
            self.offset_answer_support_end = self.offset_answer_end

    def to_doc_level(self, start, end):
        """ Populate the start and end indices with document level indices. Changes aggregation level to 'document'"""
        self.offset_answer_start = start
        self.offset_answer_end = end
        self.aggregation_level = 'document'

    def to_list(self):
        return [self.answer, self.offset_answer_start, self.offset_answer_end, self.score, self.passage_id]


class QuestionAnsweringHead(PredictionHead):
    """
    A question answering head predicts the start and end of the answer on token level.

    In addition, it gives a score for the prediction so that multiple answers can be ranked.
    There are three different kinds of scores available:
    1) (standard) score: the sum of the logits of the start and end index. This score is unbounded because the logits are unbounded.
    It is the default for ranking answers.
    2) confidence score: also based on the logits of the start and end index but scales them to the interval 0 to 1 and incorporates no_answer.
    It can be used for ranking by setting use_confidence_scores_for_ranking to True
    3) calibrated confidence score: same as 2) but divides the logits by a learned temperature_for_confidence parameter
    so that the confidence scores are closer to the model's achieved accuracy. It can be used for ranking by setting
    use_confidence_scores_for_ranking to True and temperature_for_confidence!=1.0. See examples/question_answering_confidence.py for more details.
    """

    def __init__(self, layer_dims=[768, 2], task_name='question_answering', no_ans_boost=0.0, context_window_size=100, n_best=5, n_best_per_sample=None, duplicate_filtering=-1, temperature_for_confidence=1.0, use_confidence_scores_for_ranking=False, **kwargs):
        """
        :param layer_dims: dimensions of Feed Forward block, e.g. [768,2], for adjusting to BERT embedding. Output should be always 2
        :type layer_dims: List[Int]
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
                             The higher the value, the more likely a "no answer possible given the input text" is returned by the model
        :type no_ans_boost: float
        :param context_window_size: The size, in characters, of the window around the answer span that is used when displaying the context around the answer.
        :type context_window_size: int
        :param n_best: The number of positive answer spans for each document.
        :type n_best: int
        :param n_best_per_sample: num candidate answer spans to consider from each passage. Each passage also returns "no answer" info.
                                  This is decoupled from n_best on document level, since predictions on passage level are very similar.
                                  It should have a low value
        :type n_best_per_sample: int
        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.
                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
        :type duplicate_filtering: int
        :param temperature_for_confidence: The divisor that is used to scale logits to calibrate confidence scores
        :type temperature_for_confidence: float
        :param use_confidence_scores_for_ranking: Whether to sort answers by confidence score (normalized between 0 and 1) or by standard score (unbounded)(default).
        :type use_confidence_scores_for_ranking: bool
        """
        super(QuestionAnsweringHead, self).__init__()
        if len(kwargs) > 0:
            logger.warning(f'Some unused parameters are passed to the QuestionAnsweringHead. Might not be a problem. Params: {json.dumps(kwargs)}')
        self.layer_dims = layer_dims
        assert self.layer_dims[-1] == 2
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.info(f'Prediction head initialized with size {self.layer_dims}')
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = 'per_token_squad'
        self.model_type = 'span_classification'
        self.task_name = task_name
        self.no_ans_boost = no_ans_boost
        self.context_window_size = context_window_size
        self.n_best = n_best
        if n_best_per_sample:
            self.n_best_per_sample = n_best_per_sample
        else:
            self.n_best_per_sample = n_best
        self.duplicate_filtering = duplicate_filtering
        self.generate_config()
        self.temperature_for_confidence = nn.Parameter(torch.ones(1) * temperature_for_confidence)
        self.use_confidence_scores_for_ranking = use_confidence_scores_for_ranking

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, **kwargs):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - bert-large-uncased-whole-word-masking-finetuned-squad

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            super(QuestionAnsweringHead, cls).load(pretrained_model_name_or_path)
        else:
            full_qa_model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, revision=revision, **kwargs)
            head = cls(layer_dims=[full_qa_model.config.hidden_size, 2], task_name='question_answering')
            head.feed_forward.feed_forward[0].load_state_dict(full_qa_model.qa_outputs.state_dict())
            del full_qa_model
        return head

    def forward(self, X):
        """
        One forward pass through the prediction head model, starting with language model output on token level

        """
        logits = self.feed_forward(X)
        return self.temperature_scale(logits)

    def logits_to_loss(self, logits, labels, **kwargs):
        """
        Combine predictions and labels to a per sample loss.
        """
        start_position = labels[:, 0, 0]
        end_position = labels[:, 0, 1]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)
        loss_fct = CrossEntropyLoss(reduction='none')
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        per_sample_loss = (start_loss + end_loss) / 2
        return per_sample_loss

    def temperature_scale(self, logits):
        return torch.div(logits, self.temperature_for_confidence)

    def calibrate_conf(self, logits, label_all):
        """
        Learning a temperature parameter to apply temperature scaling to calibrate confidence scores
        """
        logits = torch.cat(logits, dim=0)
        start_position = [(label[0][0] if label[0][0] >= 0 else 0) for label in label_all]
        end_position = [(label[0][1] if label[0][1] >= 0 else 0) for label in label_all]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_position = torch.tensor(start_position)
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        end_position = torch.tensor(end_position)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)
        ignored_index = start_logits.size(1) - 1
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)
        nll_criterion = CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature_for_confidence], lr=0.01, max_iter=50)

        def eval_start_end_logits():
            loss = nll_criterion(self.temperature_scale(start_logits), start_position) + nll_criterion(self.temperature_scale(end_logits), end_position)
            loss.backward()
            return loss
        optimizer.step(eval_start_end_logits)

    def logits_to_preds(self, logits, span_mask, start_of_word, seq_2_start_t, max_answer_length=1000, **kwargs):
        """
        Get the predicted index of start and end token of the answer. Note that the output is at token level
        and not word level. Note also that these logits correspond to the tokens of a sample
        (i.e. special tokens, question tokens, passage_tokens)
        """
        all_top_n = []
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        batch_size = start_logits.size()[0]
        max_seq_len = start_logits.shape[1]
        start_matrix = start_logits.unsqueeze(2).expand(-1, -1, max_seq_len)
        end_matrix = end_logits.unsqueeze(1).expand(-1, max_seq_len, -1)
        start_end_matrix = start_matrix + end_matrix
        indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=start_end_matrix.device)
        start_end_matrix[:, indices[0][:], indices[1][:]] = -888
        indices_long_span = torch.triu_indices(max_seq_len, max_seq_len, offset=max_answer_length, device=start_end_matrix.device)
        start_end_matrix[:, indices_long_span[0][:], indices_long_span[1][:]] = -777
        start_end_matrix[:, 0, 1:] = -666
        span_mask_start = span_mask.unsqueeze(2).expand(-1, -1, max_seq_len)
        span_mask_end = span_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        span_mask_2d = span_mask_start + span_mask_end
        invalid_indices = torch.nonzero(span_mask_2d != 2, as_tuple=True)
        start_end_matrix[invalid_indices[0][:], invalid_indices[1][:], invalid_indices[2][:]] = -999
        flat_scores = start_end_matrix.view(batch_size, -1)
        flat_sorted_indices_2d = flat_scores.sort(descending=True)[1]
        flat_sorted_indices = flat_sorted_indices_2d.unsqueeze(2)
        start_indices = flat_sorted_indices // max_seq_len
        end_indices = flat_sorted_indices % max_seq_len
        sorted_candidates = torch.cat((start_indices, end_indices), dim=2)
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(sorted_candidates[sample_idx], start_end_matrix[sample_idx], sample_idx, start_matrix=start_matrix[sample_idx], end_matrix=end_matrix[sample_idx])
            all_top_n.append(sample_top_n)
        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix, sample_idx, start_matrix=None, end_matrix=None):
        """ Returns top candidate answers as a list of Span objects. Operates on a matrix of summed start and end logits.
        This matrix corresponds to a single sample (includes special tokens, question tokens, passage tokens).
        This method always returns a list of len n_best + 1 (it is comprised of the n_best positive answers along with the one no_answer)"""
        top_candidates = []
        n_candidates = sorted_candidates.shape[0]
        start_idx_candidates = set()
        end_idx_candidates = set()
        start_matrix_softmax_start = torch.softmax(start_matrix[:, 0], dim=-1)
        end_matrix_softmax_end = torch.softmax(end_matrix[0, :], dim=-1)
        for candidate_idx in range(n_candidates):
            if len(top_candidates) == self.n_best_per_sample:
                break
            else:
                start_idx = sorted_candidates[candidate_idx, 0].item()
                end_idx = sorted_candidates[candidate_idx, 1].item()
                if start_idx == 0 and end_idx == 0:
                    continue
                if self.duplicate_filtering > -1 and (start_idx in start_idx_candidates or end_idx in end_idx_candidates):
                    continue
                score = start_end_matrix[start_idx, end_idx].item()
                confidence = (start_matrix_softmax_start[start_idx].item() + end_matrix_softmax_end[end_idx].item()) / 2
                top_candidates.append(QACandidate(offset_answer_start=start_idx, offset_answer_end=end_idx, score=score, answer_type='span', offset_unit='token', aggregation_level='passage', passage_id=sample_idx, confidence=confidence))
                if self.duplicate_filtering > -1:
                    for i in range(0, self.duplicate_filtering + 1):
                        start_idx_candidates.add(start_idx + i)
                        start_idx_candidates.add(start_idx - i)
                        end_idx_candidates.add(end_idx + i)
                        end_idx_candidates.add(end_idx - i)
        no_answer_score = start_end_matrix[0, 0].item()
        no_answer_confidence = (start_matrix_softmax_start[0].item() + end_matrix_softmax_end[0].item()) / 2
        top_candidates.append(QACandidate(offset_answer_start=0, offset_answer_end=0, score=no_answer_score, answer_type='no_answer', offset_unit='token', aggregation_level='passage', passage_id=None, confidence=no_answer_confidence))
        return top_candidates

    def formatted_preds(self, logits=None, preds=None, baskets=None, **kwargs):
        """ Takes a list of passage level predictions, each corresponding to one sample, and converts them into document level
        predictions. Leverages information in the SampleBaskets. Assumes that we are being passed predictions from
        ALL samples in the one SampleBasket i.e. all passages of a document. Logits should be None, because we have
        already converted the logits to predictions before calling formatted_preds.
        (see Inferencer._get_predictions_and_aggregate()).
        """
        if logits or preds is None:
            logger.error('QuestionAnsweringHead.formatted_preds() expects preds as input and logits to be None                             but was passed something different')
        samples = [s for b in baskets for s in b.samples]
        ids = [s.id for s in samples]
        passage_start_t = [s.features[0]['passage_start_t'] for s in samples]
        seq_2_start_t = [s.features[0]['seq_2_start_t'] for s in samples]
        preds_d = self.aggregate_preds(preds, passage_start_t, ids, seq_2_start_t)
        top_preds, no_ans_gaps = zip(*preds_d)
        doc_preds = self.to_qa_preds(top_preds, no_ans_gaps, baskets)
        return doc_preds

    def to_qa_preds(self, top_preds, no_ans_gaps, baskets):
        """ Groups Span objects together in a QAPred object  """
        ret = []
        for pred_d, no_ans_gap, basket in zip(top_preds, no_ans_gaps, baskets):
            token_offsets = basket.raw['document_offsets']
            pred_id = basket.id_external if basket.id_external else basket.id_internal
            question_names = ['question_text', 'qas', 'questions']
            doc_names = ['document_text', 'context', 'text']
            document_text = try_get(doc_names, basket.raw)
            question = self.get_question(question_names, basket.raw)
            ground_truth = self.get_ground_truth(basket)
            curr_doc_pred = QAPred(id=pred_id, prediction=pred_d, context=document_text, question=question, token_offsets=token_offsets, context_window_size=self.context_window_size, aggregation_level='document', ground_truth_answer=ground_truth, no_answer_gap=no_ans_gap)
            ret.append(curr_doc_pred)
        return ret

    @staticmethod
    def get_ground_truth(basket):
        if 'answers' in basket.raw:
            return basket.raw['answers']
        elif 'annotations' in basket.raw:
            return basket.raw['annotations']
        else:
            return None

    @staticmethod
    def get_question(question_names, raw_dict):
        qa_name = None
        if 'qas' in raw_dict:
            qa_name = 'qas'
        elif 'question' in raw_dict:
            qa_name = 'question'
        if qa_name:
            if type(raw_dict[qa_name][0]) == dict:
                return raw_dict[qa_name][0]['question']
        return try_get(question_names, raw_dict)

    def has_no_answer_idxs(self, sample_top_n):
        for start, end, score in sample_top_n:
            if start == 0 and end == 0:
                return True
        return False

    def aggregate_preds(self, preds, passage_start_t, ids, seq_2_start_t=None, labels=None):
        """ Aggregate passage level predictions to create document level predictions.
        This method assumes that all passages of each document are contained in preds
        i.e. that there are no incomplete documents. The output of this step
        are prediction spans. No answer is represented by a (-1, -1) span on the document level """
        n_samples = len(preds)
        all_basket_preds = {}
        all_basket_labels = {}
        for sample_idx in range(n_samples):
            basket_id = ids[sample_idx]
            basket_id = basket_id.split('-')[:-1]
            basket_id = '-'.join(basket_id)
            curr_passage_start_t = passage_start_t[sample_idx]
            if seq_2_start_t:
                cur_seq_2_start_t = seq_2_start_t[sample_idx]
                curr_passage_start_t -= cur_seq_2_start_t
            pred_d = self.pred_to_doc_idxs(preds[sample_idx], curr_passage_start_t)
            if labels:
                label_d = self.label_to_doc_idxs(labels[sample_idx], curr_passage_start_t)
            if basket_id not in all_basket_preds:
                all_basket_preds[basket_id] = []
                all_basket_labels[basket_id] = []
            all_basket_preds[basket_id].append(pred_d)
            if labels:
                all_basket_labels[basket_id].append(label_d)
        all_basket_preds = {k: self.reduce_preds(v) for k, v in all_basket_preds.items()}
        if labels:
            all_basket_labels = {k: self.reduce_labels(v) for k, v in all_basket_labels.items()}
        keys = [k for k in all_basket_preds]
        aggregated_preds = [all_basket_preds[k] for k in keys]
        if labels:
            labels = [all_basket_labels[k] for k in keys]
            return aggregated_preds, labels
        else:
            return aggregated_preds

    @staticmethod
    def reduce_labels(labels):
        """ Removes repeat answers. Represents a no answer label as (-1,-1)"""
        positive_answers = [(start, end) for x in labels for start, end in x if not (start == -1 and end == -1)]
        if not positive_answers:
            return [(-1, -1)]
        else:
            return list(set(positive_answers))

    def reduce_preds(self, preds):
        """ This function contains the logic for choosing the best answers from each passage. In the end, it
        returns the n_best predictions on the document level. """
        passage_no_answer = []
        passage_best_score = []
        passage_best_confidence = []
        no_answer_scores = []
        no_answer_confidences = []
        n_samples = len(preds)
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred.score
            best_pred_confidence = best_pred.confidence
            no_answer_score, no_answer_confidence = self.get_no_answer_score_and_confidence(sample_preds)
            no_answer_score += self.no_ans_boost
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            no_answer_confidences.append(no_answer_confidence)
            passage_best_score.append(best_pred_score)
            passage_best_confidence.append(best_pred_confidence)
        pos_answers_flat = []
        for sample_idx, passage_preds in enumerate(preds):
            for qa_candidate in passage_preds:
                if not (qa_candidate.offset_answer_start == -1 and qa_candidate.offset_answer_end == -1):
                    pos_answers_flat.append(QACandidate(offset_answer_start=qa_candidate.offset_answer_start, offset_answer_end=qa_candidate.offset_answer_end, score=qa_candidate.score, answer_type=qa_candidate.answer_type, offset_unit='token', aggregation_level='document', passage_id=str(sample_idx), n_passages_in_doc=n_samples, confidence=qa_candidate.confidence))
        pos_answer_dedup = self.deduplicate(pos_answers_flat)
        no_ans_gap = -min([(nas - pbs) for nas, pbs in zip(no_answer_scores, passage_best_score)])
        no_ans_gap_confidence = -min([(nas - pbs) for nas, pbs in zip(no_answer_confidences, passage_best_confidence)])
        best_overall_positive_score = max(x.score for x in pos_answer_dedup)
        best_overall_positive_confidence = max(x.confidence for x in pos_answer_dedup)
        no_answer_pred = QACandidate(offset_answer_start=-1, offset_answer_end=-1, score=best_overall_positive_score - no_ans_gap, answer_type='no_answer', offset_unit='token', aggregation_level='document', passage_id=None, n_passages_in_doc=n_samples, confidence=best_overall_positive_confidence - no_ans_gap_confidence)
        n_preds = [no_answer_pred] + pos_answer_dedup
        n_preds_sorted = sorted(n_preds, key=lambda x: x.confidence if self.use_confidence_scores_for_ranking else x.score, reverse=True)
        n_preds_reduced = n_preds_sorted[:self.n_best]
        return n_preds_reduced, no_ans_gap

    @staticmethod
    def deduplicate(flat_pos_answers):
        seen = {}
        for qa_answer in flat_pos_answers:
            if (qa_answer.offset_answer_start, qa_answer.offset_answer_end) not in seen:
                seen[qa_answer.offset_answer_start, qa_answer.offset_answer_end] = qa_answer
            else:
                seen_score = seen[qa_answer.offset_answer_start, qa_answer.offset_answer_end].score
                if qa_answer.score > seen_score:
                    seen[qa_answer.offset_answer_start, qa_answer.offset_answer_end] = qa_answer
        return list(seen.values())

    @staticmethod
    def get_no_answer_score_and_confidence(preds):
        for qa_answer in preds:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            score = qa_answer.score
            confidence = qa_answer.confidence
            if start == -1 and end == -1:
                return score, confidence
        raise Exception

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t):
        """ Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) qa_answer but will instead be represented by (-1, -1)"""
        new_pred = []
        for qa_answer in pred:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                if start < 0:
                    logger.error('Start token index < 0 (document level)')
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                if end < 0:
                    logger.error('End token index < 0 (document level)')
            qa_answer.to_doc_level(start, end)
            new_pred.append(qa_answer)
        return new_pred

    @staticmethod
    def label_to_doc_idxs(label, passage_start_t):
        """ Converts the passage level labels to document level labels. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) span but will instead be represented by (-1, -1)"""
        new_label = []
        for start, end in label:
            if start > 0 or end > 0:
                new_label.append((start + passage_start_t, end + passage_start_t))
            if start == 0 and end == 0:
                new_label.append((-1, -1))
        return new_label

    def prepare_labels(self, labels, start_of_word, **kwargs):
        return labels

    @staticmethod
    def merge_formatted_preds(preds_all):
        """ Merges results from the two prediction heads used for NQ style QA. Takes the prediction from QA head and
        assigns it the appropriate classification label. This mapping is achieved through passage_id.
        preds_all should contain [QuestionAnsweringHead.formatted_preds(), TextClassificationHead()]. The first item
        of this list should be of len=n_documents while the second item should be of len=n_passages"""
        ret = []

        def chunk(iterable, lengths):
            if sum(lengths) != len(iterable):
                logger.error('Sum of the lengths does not match the length of the iterable')
            cumsum = list(np.cumsum(lengths))
            cumsum = [0] + cumsum
            spans = [(cumsum[i], cumsum[i + 1]) for i in range(len(cumsum) - 1)]
            ret = [iterable[start:end] for start, end in spans]
            return ret
        cls_preds = preds_all[1][0]['predictions']
        qa_preds = preds_all[0][0]
        samples_per_doc = [doc_pred.n_passages for doc_pred in preds_all[0][0]]
        cls_preds_grouped = chunk(cls_preds, samples_per_doc)
        for qa_pred, cls_preds in zip(qa_preds, cls_preds_grouped):
            qa_candidates = qa_pred.prediction
            qa_candidates_new = []
            for qa_candidate in qa_candidates:
                passage_id = qa_candidate.passage_id
                if passage_id is not None:
                    cls_pred = cls_preds[int(passage_id)]['label']
                else:
                    cls_pred = 'no_answer'
                qa_candidate.add_cls(cls_pred)
                qa_candidates_new.append(qa_candidate)
            qa_pred.prediction = qa_candidates_new
            ret.append(qa_pred)
        return ret


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForwardBlock,
     lambda: ([], {'layer_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_deepset_ai_FARM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

