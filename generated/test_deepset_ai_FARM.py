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
doc_classification_multilabel = _module
doc_classification_multilabel_roberta = _module
doc_classification_with_earlystopping = _module
doc_classification_word_embedding_LM = _module
doc_regression = _module
embeddings_extraction = _module
embeddings_extraction_s3e_pooling = _module
evaluation = _module
lm_finetuning = _module
natural_questions = _module
ner = _module
onnx_question_answering = _module
passage_ranking = _module
question_answering = _module
question_answering_crossvalidation = _module
streaming_inference = _module
text_pair_classification = _module
train_from_scratch = _module
train_from_scratch_with_sagemaker = _module
wordembedding_inference = _module
farm = _module
conversion = _module
convert_tf_checkpoint_to_pytorch = _module
BertOnnxModel = _module
OnnxModel = _module
onnx_optimization = _module
bert_model_optimization = _module
convert_tf_checkpoint_to_pytorch = _module
data_handler = _module
data_silo = _module
dataloader = _module
dataset = _module
input_features = _module
processor = _module
samples = _module
utils = _module
eval = _module
metrics = _module
msmarco_passage_farm = _module
msmarco_passage_official = _module
squad_evaluation = _module
experiment = _module
file_utils = _module
infer = _module
inference_rest_api = _module
modeling = _module
adaptive_model = _module
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
create_testdata = _module
test_conversion = _module
test_doc_classification = _module
test_doc_classification_distilbert = _module
test_doc_classification_roberta = _module
test_doc_regression = _module
test_inference = _module
test_lm_finetuning = _module
test_natural_questions = _module
test_ner = _module
test_ner_amp = _module
test_processor_saving_loading = _module
test_question_answering = _module
test_s3e_pooling = _module
test_tokenization = _module

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


import logging


import torch


import torch.multiprocessing as mp


import copy


from functools import partial


import random


from itertools import groupby


import numpy as np


from sklearn.utils.class_weight import compute_class_weight


from torch.utils.data import ConcatDataset


from torch.utils.data import Dataset


from torch.utils.data import Subset


from torch.utils.data import IterableDataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import KFold


from math import ceil


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from torch.utils.data import TensorDataset


import numbers


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn.metrics import matthews_corrcoef


from sklearn.metrics import recall_score


from sklearn.metrics import precision_score


from sklearn.metrics import f1_score


from sklearn.metrics import mean_squared_error


from sklearn.metrics import r2_score


from sklearn.metrics import classification_report


from functools import wraps


import numpy


from torch import nn


from collections import OrderedDict


import inspect


from torch.nn.parallel import DistributedDataParallel


from torch.nn import DataParallel


import itertools


from scipy.special import expit


from scipy.special import softmax


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


from torch import multiprocessing as mp


from copy import deepcopy


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
                preds_p = kwargs['preds_p']
                temp = [y[0] for y in preds_p]
                preds_p_flat = [item for sublist in temp for item in sublist]
                kwargs['preds_p'] = preds_p_flat
            except KeyError:
                kwargs['preds_p'] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and 'predictions' in preds:
                preds_final.append(preds)
        else:
            preds_final = [list() for _ in range(n_heads)]
            preds = kwargs['preds_p']
            preds_for_heads = stack(preds)
            logits_for_heads = [None] * n_heads
            samples = [s for b in kwargs['baskets'] for s in b.samples]
            kwargs['samples'] = samples
            del kwargs['preds_p']
            for i, (head, preds_p_for_head, logits_for_head) in enumerate(zip(self.prediction_heads, preds_for_heads, logits_for_heads)):
                preds = head.formatted_preds(logits=logits_for_head, preds_p=preds_p_for_head, **kwargs)
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


class NamedDataLoader(DataLoader):
    """
    A modified version of the PyTorch DataLoader that returns a dictionary where the key is
    the name of the tensor and the value is the tensor itself.
    """

    def __init__(self, dataset, batch_size, sampler=None, tensor_names=None, num_workers=0, pin_memory=False):
        """
        :param dataset: The dataset that will be wrapped by this NamedDataLoader
        :type dataset: Dataset
        :param sampler: The sampler used by the NamedDataLoader to choose which samples to include in the batch
        :type sampler: Sampler
        :param batch_size: The size of the batch to be returned by the NamedDataLoader
        :type batch_size: int
        :param tensor_names: The names of the tensor, in the order that the dataset returns them in.
        :type tensor_names: list
        :param num_workers: number of workers to use for the DataLoader
        :type num_workers: int
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
        :type pin_memory: bool
        """

        def collate_fn(batch):
            """
            A custom collate function that formats the batch as a dictionary where the key is
            the name of the tensor and the value is the tensor itself
            """
            if type(dataset).__name__ == '_StreamingDataSet':
                _tensor_names = dataset.tensor_names
            else:
                _tensor_names = tensor_names
            if type(batch[0]) == list:
                batch = batch[0]
            assert len(batch[0]) == len(_tensor_names), 'Dataset contains {} tensors while there are {} tensor names supplied: {}'.format(len(batch[0]), len(_tensor_names), _tensor_names)
            lists_temp = [[] for _ in range(len(_tensor_names))]
            ret = dict(zip(_tensor_names, lists_temp))
            for example in batch:
                for name, tensor in zip(_tensor_names, example):
                    ret[name].append(tensor)
            for key in ret:
                ret[key] = torch.stack(ret[key])
            return ret
        super(NamedDataLoader, self).__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)

    def __len__(self):
        if type(self.dataset).__name__ == '_StreamingDataSet':
            num_samples = len(self.dataset)
            num_batches = ceil(num_samples / self.dataset.batch_size)
            return num_batches
        else:
            return super().__len__()


TRACTOR_SMALL = """ 
              ______
               |o  |   !
   __          |:`_|---'-.
  |__|______.-/ _ \\-----.|       
 (o)(o)------'\\ _ /     ( )      
 """


def calc_chunksize(num_dicts, min_chunksize=4, max_chunksize=2000, max_processes=128):
    num_cpus = min(mp.cpu_count() - 1 or 1, max_processes)
    dicts_per_cpu = np.ceil(num_dicts / num_cpus)
    multiprocessing_chunk_size = int(np.clip(np.ceil(dicts_per_cpu / 5), a_min=min_chunksize, a_max=max_chunksize))
    if num_dicts != 1:
        while num_dicts % multiprocessing_chunk_size == 1:
            multiprocessing_chunk_size -= -1
    dict_batches_to_process = int(num_dicts / multiprocessing_chunk_size)
    num_processes = min(num_cpus, dict_batches_to_process) or 1
    return multiprocessing_chunk_size, num_processes


def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(json.dumps(payload_dict, sort_keys=True).encode('utf-8')).hexdigest()
    return checksum


def grouper(iterable, n, worker_id=0, total_workers=1):
    """
    Split an iterable into a list of n-sized chunks. Each element in the chunk is a tuple of (index_num, element).

    Example:

    >>> list(grouper('ABCDEFG', 3))
    [[(0, 'A'), (1, 'B'), (2, 'C')], [(3, 'D'), (4, 'E'), (5, 'F')], [(6, 'G')]]



    Use with the StreamingDataSilo

    When StreamingDataSilo is used with multiple PyTorch DataLoader workers, the generator
    yielding dicts(that gets converted to datasets) is replicated across the workers.

    To avoid duplicates, we split the dicts across workers by creating a new generator for
    each worker using this method.

    Input --> [dictA, dictB, dictC, dictD, dictE, ...] with total worker=3 and n=2

    Output for worker 1: [(dictA, dictB), (dictG, dictH), ...]
    Output for worker 2: [(dictC, dictD), (dictI, dictJ), ...]
    Output for worker 3: [(dictE, dictF), (dictK, dictL), ...]

    This method also adds an index number to every dict yielded similar to the grouper().

    :param iterable: a generator object that yields dicts
    :type iterable: generator
    :param n: the dicts are grouped in n-sized chunks that gets converted to datasets
    :type n: int
    :param worker_id: the worker_id for the PyTorch DataLoader
    :type worker_id: int
    :param total_workers: total number of workers for the PyTorch DataLoader
    :type total_workers: int
    """

    def get_iter_start_pos(gen):
        start_pos = worker_id * n
        for i in gen:
            if start_pos:
                start_pos -= 1
                continue
            yield i

    def filter_elements_per_worker(gen):
        x = n
        y = (total_workers - 1) * n
        for i in gen:
            if x:
                yield i
                x -= 1
            elif y != 1:
                y -= 1
                continue
            else:
                x = n
                y = (total_workers - 1) * n
    iterable = iter(enumerate(iterable))
    iterable = get_iter_start_pos(iterable)
    if total_workers > 1:
        iterable = filter_elements_per_worker(iterable)
    return iter(lambda : list(islice(iterable, n)), [])


WORKER_F = ' 0 \n/w\\\n/ \\\n'


WORKER_M = " 0 \n/|\\\n/'\\\n"


WORKER_X = " 0 \n/w\\\n/'\\\n"


def log_ascii_workers(n, logger):
    m_worker_lines = WORKER_M.split('\n')
    f_worker_lines = WORKER_F.split('\n')
    x_worker_lines = WORKER_X.split('\n')
    all_worker_lines = []
    for i in range(n):
        rand = np.random.randint(low=0, high=3)
        if rand % 3 == 0:
            all_worker_lines.append(f_worker_lines)
        elif rand % 3 == 1:
            all_worker_lines.append(m_worker_lines)
        else:
            all_worker_lines.append(x_worker_lines)
    zipped = zip(*all_worker_lines)
    for z in zipped:
        logger.info('  '.join(z))


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
        for tok_idx, tok_id in enumerate(token_ids[(sample_idx), :]):
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
                matrix[(j), :] = stage_vec[-1][j]
        matrix_no_mean = matrix - matrix.mean(1)[:, (np.newaxis)]
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
    def load(cls, pretrained_model_name_or_path, n_added_tokens=0, language_model_class=None, **kwargs):
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

        See all supported model variations here: https://huggingface.co/models

        The appropriate language model class is inferred automatically from `pretrained_model_name_or_path`
        or can be manually supplied via `language_model_class`.

        :param pretrained_model_name_or_path: The path of the saved pretrained model or its name.
        :type pretrained_model_name_or_path: str
        :param language_model_class: (Optional) Name of the language model class to load (e.g. `Bert`)
        :type language_model_class: str

        """
        config_file = Path(pretrained_model_name_or_path) / 'language_model_config.json'
        if os.path.exists(config_file):
            config = json.load(open(config_file))
            language_model = cls.subclasses[config['name']].load(pretrained_model_name_or_path)
        else:
            if language_model_class is None:
                pretrained_model_name_or_path = str(pretrained_model_name_or_path)
                if 'xlm' in pretrained_model_name_or_path and 'roberta' in pretrained_model_name_or_path:
                    language_model_class = 'XLMRoberta'
                elif 'roberta' in pretrained_model_name_or_path:
                    language_model_class = 'Roberta'
                elif 'albert' in pretrained_model_name_or_path:
                    language_model_class = 'Albert'
                elif 'distilbert' in pretrained_model_name_or_path:
                    language_model_class = 'DistilBert'
                elif 'bert' in pretrained_model_name_or_path:
                    language_model_class = 'Bert'
                elif 'xlnet' in pretrained_model_name_or_path:
                    language_model_class = 'XLNet'
                elif 'electra' in pretrained_model_name_or_path:
                    language_model_class = 'Electra'
                elif 'word2vec' in pretrained_model_name_or_path.lower() or 'glove' in pretrained_model_name_or_path.lower():
                    language_model_class = 'WordEmbedding_LM'
            if language_model_class:
                language_model = cls.subclasses[language_model_class].load(pretrained_model_name_or_path, **kwargs)
            else:
                language_model = None
        if not language_model:
            raise Exception(f"Model not found for {pretrained_model_name_or_path}. Either supply the local path for a saved model or one of bert/roberta/xlnet/albert/distilbert models that can be downloaded from remote. Ensure that the model class name can be inferred from the directory name when loading a Transformers' model. Here's a list of available models: https://farm.deepset.ai/api/modeling.html#farm.modeling.language_model.LanguageModel.load")
        if n_added_tokens != 0:
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            vocab_size = model_emb_size + n_added_tokens
            logger.info(f'Resizing embedding layer of LM from {model_emb_size} to {vocab_size} to cope with custom vocab.')
            language_model.model.resize_token_embeddings(vocab_size)
            model_emb_size = language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            assert vocab_size == model_emb_size
        return language_model

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
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.

        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        save_name = Path(save_dir) / 'language_model.bin'
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), save_name)
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
        if len(matches) == 0:
            language = 'english'
            logger.warning("Could not automatically detect from language model name what language it is. \n\t We guess it's an *ENGLISH* model ... \n\t If not: Init the language model by supplying the 'language' param.")
        elif len(matches) > 1:
            raise ValueError(f"Could not automatically detect from language model name what language it is.\n\t Found multiple matches: {matches}\n\t Please init the language model by manually supplying the 'language' as a parameter.\n")
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
            vecs = sequence_output[:, (0), :].cpu().numpy()
        elif self.extraction_strategy == 's3e':
            vecs = self._pool_tokens(sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token, input_ids=input_ids, s3e_stats=self.s3e_stats)
        else:
            raise NotImplementedError
        preds = []
        for vec, sample in zip(vecs, samples):
            pred = {}
            pred['context'] = sample.tokenized['tokens']
            pred['vec'] = vec
            preds.append(pred)
        return preds

    def _pool_tokens(self, sequence_output, padding_mask, strategy, ignore_first_token, input_ids=None, s3e_stats=None):
        token_vecs = sequence_output.cpu().numpy()
        padding_mask = padding_mask.cpu().numpy()
        ignore_mask_2d = padding_mask == 0
        if ignore_first_token:
            ignore_mask_2d[:, (0)] = True
        ignore_mask_3d = np.zeros(token_vecs.shape, dtype=bool)
        ignore_mask_3d[:, :, :] = ignore_mask_2d[:, :, (np.newaxis)]
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
            if is_json(value) and key[0] != '_':
                config[key] = value
        config['name'] = self.__class__.__name__
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


def span_to_string(start_t, end_t, token_offsets, clear_text):
    if start_t == -1 and end_t == -1:
        return '', 0, 0
    n_tokens = len(token_offsets)
    end_t += 1
    end_t = min(end_t, n_tokens)
    start_ch = token_offsets[start_t]
    if end_t == n_tokens:
        end_ch = len(clear_text)
    else:
        end_ch = token_offsets[end_t]
    return clear_text[start_ch:end_ch].strip(), start_ch, end_ch


class DocumentPred:
    """ Contains a collection of Span predictions for one document. Used in Question Answering. Also contains all
    attributes needed to generate the appropriate output json"""

    def __init__(self, id, document_text, question, preds, no_ans_gap, token_offsets, context_window_size, question_id=None):
        self.id = id
        self.preds = preds
        self.n_samples = preds[0].n_samples
        self.document_text = document_text
        self.question = question
        self.question_id = question_id
        self.no_ans_gap = no_ans_gap
        self.token_offsets = token_offsets
        self.context_window_size = context_window_size

    def __str__(self):
        preds_str = '\n'.join([f'{p}' for p in self.preds])
        ret = f'id: {self.id}\ndocument: {self.document_text}\npreds:\n{preds_str}'
        return ret

    def __repr__(self):
        return str(self)

    def to_json(self):
        answers = self.answers_to_json()
        ret = {'task': 'qa', 'predictions': [{'question': self.question, 'question_id': self.question_id, 'ground_truth': None, 'answers': answers, 'no_ans_gap': self.no_ans_gap}]}
        return ret

    def answers_to_json(self):
        ret = []
        for span in self.preds:
            string = span.pred_str
            start_t = span.start
            end_t = span.end
            score = span.score
            classification = span.classification
            _, ans_start_ch, ans_end_ch = span_to_string(start_t, end_t, self.token_offsets, self.document_text)
            context_string, context_start_ch, context_end_ch = self.create_context(ans_start_ch, ans_end_ch, self.document_text)
            curr = {'score': score, 'probability': -1, 'answer': string, 'offset_answer_start': ans_start_ch, 'offset_answer_end': ans_end_ch, 'context': context_string, 'classification': classification, 'offset_context_start': context_start_ch, 'offset_context_end': context_end_ch, 'document_id': self.id}
            ret.append(curr)
        return ret

    def create_context(self, ans_start_ch, ans_end_ch, clear_text):
        if ans_start_ch == 0 and ans_end_ch == 0:
            return '', 0, 0
        else:
            len_text = len(clear_text)
            midpoint = int((ans_end_ch - ans_start_ch) / 2) + ans_start_ch
            half_window = int(self.context_window_size / 2)
            context_start_ch = midpoint - half_window
            context_end_ch = midpoint + half_window
            overhang_start = max(0, -context_start_ch)
            overhang_end = max(0, context_end_ch - len_text)
            context_start_ch -= overhang_end
            context_start_ch = max(0, context_start_ch)
            context_end_ch += overhang_start
            context_end_ch = min(len_text, context_end_ch)
        context_string = clear_text[context_start_ch:context_end_ch]
        return context_string, context_start_ch, context_end_ch

    def to_squad_eval(self):
        preds = [x.to_list() for x in self.preds]
        ret = {'id': self.id, 'preds': preds}
        return ret


class Span:

    def __init__(self, start, end, score=None, sample_idx=None, n_samples=None, classification=None, unit=None, pred_str=None, id=None, level=None):
        self.start = start
        self.end = end
        self.score = score
        self.unit = unit
        self.sample_idx = sample_idx
        self.classification = classification
        self.n_samples = n_samples
        self.pred_str = pred_str
        self.id = id
        self.level = level

    def to_list(self):
        return [self.pred_str, self.start, self.end, self.score, self.sample_idx]

    def __str__(self):
        if self.pred_str is None:
            pred_str = 'is_impossible'
        else:
            pred_str = self.pred_str
        ret = f'answer: {pred_str}\nscore: {self.score}'
        return ret

    def __repr__(self):
        return str(self)


class QuestionAnsweringHead(PredictionHead):
    """
    A question answering head predicts the start and end of the answer on token level.
    """

    def __init__(self, layer_dims=[768, 2], task_name='question_answering', no_ans_boost=0.0, context_window_size=100, n_best=5, n_best_per_sample=1, **kwargs):
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
        """
        super(QuestionAnsweringHead, self).__init__()
        if kwargs is not None:
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
        self.n_best_per_sample = n_best_per_sample
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path):
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

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            super(QuestionAnsweringHead, cls).load(pretrained_model_name_or_path)
        else:
            full_qa_model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)
            head = cls(layer_dims=[full_qa_model.config.hidden_size, 2], loss_ignore_index=-1, task_name='question_answering')
            head.feed_forward.feed_forward[0].load_state_dict(full_qa_model.qa_outputs.state_dict())
            del full_qa_model
        return head

    def forward(self, X):
        """
        One forward pass through the prediction head model, starting with language model output on token level

        """
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, labels, **kwargs):
        """
        Combine predictions and labels to a per sample loss.
        """
        start_position = labels[:, (0), (0)]
        end_position = labels[:, (0), (1)]
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

    def logits_to_preds(self, logits, padding_mask, start_of_word, seq_2_start_t, max_answer_length=1000, **kwargs):
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
        n_non_padding = torch.sum(padding_mask, dim=1)
        start_matrix = start_logits.unsqueeze(2).expand(-1, -1, max_seq_len)
        end_matrix = end_logits.unsqueeze(1).expand(-1, max_seq_len, -1)
        start_end_matrix = start_matrix + end_matrix
        indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=start_end_matrix.device)
        start_end_matrix[:, (indices[0][:]), (indices[1][:])] = -999
        start_end_matrix[:, (0), 1:] = -999
        flat_scores = start_end_matrix.view(batch_size, -1)
        flat_sorted_indices_2d = flat_scores.sort(descending=True)[1]
        flat_sorted_indices = flat_sorted_indices_2d.unsqueeze(2)
        start_indices = flat_sorted_indices // max_seq_len
        end_indices = flat_sorted_indices % max_seq_len
        sorted_candidates = torch.cat((start_indices, end_indices), dim=2)
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(sorted_candidates[sample_idx], start_end_matrix[sample_idx], n_non_padding[sample_idx].item(), max_answer_length, seq_2_start_t[sample_idx].item())
            all_top_n.append(sample_top_n)
        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix, n_non_padding, max_answer_length, seq_2_start_t):
        """ Returns top candidate answers as a list of Span objects. Operates on a matrix of summed start and end logits.
        This matrix corresponds to a single sample (includes special tokens, question tokens, passage tokens).
        This method always returns a list of len n_best + 1 (it is comprised of the n_best positive answers along with the one no_answer)"""
        top_candidates = []
        n_candidates = sorted_candidates.shape[0]
        for candidate_idx in range(n_candidates):
            if len(top_candidates) == self.n_best_per_sample:
                break
            else:
                start_idx = sorted_candidates[candidate_idx, 0].item()
                end_idx = sorted_candidates[candidate_idx, 1].item()
                if start_idx == 0 and end_idx == 0:
                    continue
                if self.valid_answer_idxs(start_idx, end_idx, n_non_padding, max_answer_length, seq_2_start_t):
                    score = start_end_matrix[start_idx, end_idx].item()
                    top_candidates.append(Span(start_idx, end_idx, score, unit='token', level='passage'))
        no_answer_score = start_end_matrix[0, 0].item()
        top_candidates.append(Span(0, 0, no_answer_score, unit='token', pred_str='', level='passage'))
        return top_candidates

    @staticmethod
    def valid_answer_idxs(start_idx, end_idx, n_non_padding, max_answer_length, seq_2_start_t):
        """ Returns True if the supplied index span is a valid prediction. The indices being provided
        should be on sample/passage level (special tokens + question_tokens + passag_tokens)
        and not document level"""
        if start_idx < seq_2_start_t and start_idx != 0:
            return False
        if end_idx < seq_2_start_t and end_idx != 0:
            return False
        if start_idx >= n_non_padding - 1:
            return False
        if end_idx >= n_non_padding - 1:
            return False
        length = end_idx - start_idx + 1
        if length > max_answer_length:
            return False
        return True

    def formatted_preds(self, logits=None, preds_p=None, baskets=None, **kwargs):
        """ Takes a list of predictions, each corresponding to one sample, and converts them into document level
        predictions. Leverages information in the SampleBaskets. Assumes that we are being passed predictions from
        ALL samples in the one SampleBasket i.e. all passages of a document. Logits should be None, because we have
        already converted the logits to predictions before calling formatted_preds
        (see Inferencer._get_predictions_and_aggregate()).
        """
        assert logits is None, 'Logits are not None, something is passed wrongly into formatted_preds() in infer.py'
        assert preds_p is not None, 'No preds_p passed to formatted_preds()'
        samples = [s for b in baskets for s in b.samples]
        ids = [s.id.split('-') for s in samples]
        passage_start_t = [s.features[0]['passage_start_t'] for s in samples]
        seq_2_start_t = [s.features[0]['seq_2_start_t'] for s in samples]
        preds_d = self.aggregate_preds(preds_p, passage_start_t, ids, seq_2_start_t)
        assert len(preds_d) == len(baskets)
        top_preds, no_ans_gaps = zip(*preds_d)
        doc_preds = self.to_doc_preds(top_preds, no_ans_gaps, baskets)
        return doc_preds

    def to_doc_preds(self, top_preds, no_ans_gaps, baskets):
        """ Groups Span objects together in a DocumentPred object  """
        ret = []
        for pred_d, no_ans_gap, basket in zip(top_preds, no_ans_gaps, baskets):
            try:
                token_offsets = basket.samples[0].tokenized['document_offsets']
            except KeyError:
                token_offsets = basket.raw['document_offsets']
            try:
                document_text = basket.raw['context']
            except KeyError:
                try:
                    document_text = basket.raw['text']
                except KeyError:
                    document_text = basket.raw['document_text']
            try:
                question = basket.raw['questions'][0]
            except KeyError:
                try:
                    question = basket.raw['qas'][0]
                except KeyError:
                    question = basket.raw['question_text']
            try:
                question_id = basket.raw['squad_id']
            except KeyError:
                question_id = None
            basket_id = basket.id
            full_preds = []
            for span, basket in zip(pred_d, baskets):
                pred_str, _, _ = span_to_string(span.start, span.end, token_offsets, document_text)
                span.pred_str = pred_str
                full_preds.append(span)
            curr_doc_pred = DocumentPred(id=basket_id, preds=full_preds, document_text=document_text, question=question, no_ans_gap=no_ans_gap, token_offsets=token_offsets, context_window_size=self.context_window_size, question_id=question_id)
            ret.append(curr_doc_pred)
        return ret

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
            id_1, id_2, _ = ids[sample_idx]
            basket_id = f'{id_1}-{id_2}'
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
        no_answer_scores = []
        n_samples = len(preds)
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred.score
            no_answer_score = self.get_no_answer_score(sample_preds) + self.no_ans_boost
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            passage_best_score.append(best_pred_score)
        pos_answers_flat = [Span(span.start, span.end, span.score, sample_idx, n_samples, unit='token', level='passage') for sample_idx, passage_preds in enumerate(preds) for span in passage_preds if not (span.start == -1 and span.end == -1)]
        pos_answer_dedup = self.deduplicate(pos_answers_flat)
        no_ans_gap = -min([(nas - pbs) for nas, pbs in zip(no_answer_scores, passage_best_score)])
        best_overall_positive_score = max(x.score for x in pos_answer_dedup)
        no_answer_pred = Span(-1, -1, best_overall_positive_score - no_ans_gap, None, n_samples, unit='token')
        n_preds = [no_answer_pred] + pos_answer_dedup
        n_preds_sorted = sorted(n_preds, key=lambda x: x.score, reverse=True)
        n_preds_reduced = n_preds_sorted[:self.n_best]
        return n_preds_reduced, no_ans_gap

    @staticmethod
    def deduplicate(flat_pos_answers):
        seen = {}
        for span in flat_pos_answers:
            if (span.start, span.end) not in seen:
                seen[span.start, span.end] = span
            else:
                seen_score = seen[span.start, span.end].score
                if span.score > seen_score:
                    seen[span.start, span.end] = span
        return [span for span in seen.values()]

    @staticmethod
    def get_no_answer_score(preds):
        for span in preds:
            start = span.start
            end = span.end
            score = span.score
            if start == -1 and end == -1:
                return score
        raise Exception

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t):
        """ Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) span but will instead be represented by (-1, -1)"""
        new_pred = []
        for span in pred:
            start = span.start
            end = span.end
            score = span.score
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                assert start >= 0
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                assert start >= 0
            new_pred.append(Span(start, end, score, level='doc'))
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
        assigns it the appropriate classification label. This mapping is achieved through sample_idx.
        preds_all should contain [QuestionAnsweringHead.formatted_preds(), TextClassificationHead()]. The first item
        of this list should be of len=n_documents while the second item should be of len=n_passages"""
        ret = []

        def chunk(iterable, lengths):
            assert sum(lengths) == len(iterable)
            cumsum = list(np.cumsum(lengths))
            cumsum = [0] + cumsum
            spans = [(cumsum[i], cumsum[i + 1]) for i in range(len(cumsum) - 1)]
            ret = [iterable[start:end] for start, end in spans]
            return ret
        cls_preds = preds_all[1][0]['predictions']
        qa_preds = preds_all[0][0]
        samples_per_doc = [doc_pred.n_samples for doc_pred in preds_all[0][0]]
        cls_preds_grouped = chunk(cls_preds, samples_per_doc)
        for qa_doc_pred, cls_preds in zip(qa_preds, cls_preds_grouped):
            pred_spans = qa_doc_pred.preds
            pred_spans_new = []
            for pred_span in pred_spans:
                sample_idx = pred_span.sample_idx
                if sample_idx is not None:
                    cls_pred = cls_preds[sample_idx]['label']
                else:
                    cls_pred = None
                pred_span.classification = cls_pred
                pred_spans_new.append(pred_span)
            qa_doc_pred.preds = pred_spans_new
            ret.append(qa_doc_pred)
        return ret


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

    def __init__(self, id: str, raw: dict, external_id=None, samples=None):
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
        self.id = id
        self.external_id = external_id
        self.raw = raw
        self.samples = samples


class Tokenizer:
    """
    Simple Wrapper for Tokenizers from the transformers package. Enables loading of different Tokenizer classes with a uniform interface.
    """

    @classmethod
    def load(cls, pretrained_model_name_or_path, tokenizer_class=None, **kwargs):
        """
        Enables loading of different Tokenizer classes with a uniform interface. Either infer the class from
        `pretrained_model_name_or_path` or define it manually via `tokenizer_class`.

        :param pretrained_model_name_or_path:  The path of the saved pretrained model or its name (e.g. `bert-base-uncased`)
        :type pretrained_model_name_or_path: str
        :param tokenizer_class: (Optional) Name of the tokenizer class to load (e.g. `BertTokenizer`)
        :type tokenizer_class: str
        :param kwargs:
        :return: Tokenizer
        """
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if tokenizer_class is None:
            if 'albert' in pretrained_model_name_or_path.lower():
                tokenizer_class = 'AlbertTokenizer'
            elif 'xlm-roberta' in pretrained_model_name_or_path.lower():
                tokenizer_class = 'XLMRobertaTokenizer'
            elif 'roberta' in pretrained_model_name_or_path.lower():
                tokenizer_class = 'RobertaTokenizer'
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
            else:
                raise ValueError(f"Could not infer tokenizer_class from name '{pretrained_model_name_or_path}'. Set arg `tokenizer_class` in Tokenizer.load() to one of: AlbertTokenizer, XLMRobertaTokenizer, RobertaTokenizer, DistilBertTokenizer, BertTokenizer, or XLNetTokenizer.")
            logger.info(f"Loading tokenizer of type '{tokenizer_class}'")
        if tokenizer_class == 'AlbertTokenizer':
            ret = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
        elif tokenizer_class == 'XLMRobertaTokenizer':
            ret = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == 'RobertaTokenizer':
            ret = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == 'DistilBertTokenizer':
            ret = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == 'BertTokenizer':
            ret = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == 'XLNetTokenizer':
            ret = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path, keep_accents=True, **kwargs)
        elif tokenizer_class == 'ElectraTokenizer':
            ret = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif tokenizer_class == 'EmbeddingTokenizer':
            ret = EmbeddingTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if ret is None:
            raise Exception('Unable to load tokenizer')
        else:
            return ret


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
        try:
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.long)
        except ValueError:
            cur_tensor = torch.tensor([sample[t_name] for sample in features], dtype=torch.float32)
        all_tensors.append(cur_tensor)
    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


def convert_qa_input_dict(infer_dict):
    """ Input dictionaries in QA can either have ["context", "qas"] (internal format) as keys or
    ["text", "questions"] (api format). This function converts the latter into the former"""
    try:
        if 'context' in infer_dict and 'qas' in infer_dict:
            return infer_dict
        questions = infer_dict['questions']
        text = infer_dict['text']
        document_id = infer_dict.get('document_id', None)
        qas = [{'question': q, 'id': None, 'answers': [], 'is_impossible': False} for i, q in enumerate(questions)]
        converted = {'qas': qas, 'context': text, 'document_id': document_id}
        return converted
    except KeyError:
        raise Exception('Input does not have the expected format')


def chunk_into_passages(doc_offsets, doc_stride, passage_len_t, doc_text):
    """ Returns a list of dictionaries which each describe the start, end and id of a passage
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


def offset_to_token_idx(token_offsets, ch_idx):
    """ Returns the idx of the token at the given character idx"""
    n_tokens = len(token_offsets)
    for i in range(n_tokens):
        if i + 1 == n_tokens or token_offsets[i] <= ch_idx < token_offsets[i + 1]:
            return i


def process_answers(answers, doc_offsets, passage_start_c, passage_start_t):
    """TODO Write Comment"""
    answers_clear = []
    answers_tokenized = []
    for answer in answers:
        answer_text = answer['text']
        answer_len_c = len(answer_text)
        answer_start_c = answer['offset']
        answer_end_c = answer_start_c + answer_len_c - 1
        answer_start_t = offset_to_token_idx(doc_offsets, answer_start_c)
        answer_end_t = offset_to_token_idx(doc_offsets, answer_end_c)
        answer_start_c -= passage_start_c
        answer_end_c -= passage_start_c
        answer_start_t -= passage_start_t
        answer_end_t -= passage_start_t
        curr_answer_clear = {'text': answer_text, 'start_c': answer_start_c, 'end_c': answer_end_c}
        curr_answer_tokenized = {'start_t': answer_start_t, 'end_t': answer_end_t, 'answer_type': answer['answer_type']}
        answers_clear.append(curr_answer_clear)
        answers_tokenized.append(curr_answer_tokenized)
    return answers_clear, answers_tokenized


def create_samples_qa(dictionary, max_query_len, max_seq_len, doc_stride, n_special_tokens):
    """
    This method will split question-document pairs from the SampleBasket into question-passage pairs which will
    each form one sample. The "t" and "c" in variables stand for token and character respectively.
    """
    question_tokens = dictionary['question_tokens'][:max_query_len]
    question_len_t = len(question_tokens)
    question_offsets = dictionary['question_offsets']
    doc_tokens = dictionary['document_tokens']
    doc_offsets = dictionary['document_offsets']
    doc_text = dictionary['document_text']
    doc_start_of_word = dictionary['document_start_of_word']
    samples = []
    passage_len_t = max_seq_len - question_len_t - n_special_tokens
    passage_spans = chunk_into_passages(doc_offsets, doc_stride, passage_len_t, doc_text)
    for passage_span in passage_spans:
        passage_start_t = passage_span['passage_start_t']
        passage_end_t = passage_span['passage_end_t']
        passage_start_c = passage_span['passage_start_c']
        passage_end_c = passage_span['passage_end_c']
        passage_id = passage_span['passage_id']
        passage_offsets = doc_offsets[passage_start_t:passage_end_t]
        passage_start_of_word = doc_start_of_word[passage_start_t:passage_end_t]
        passage_offsets = [(x - passage_offsets[0]) for x in passage_offsets]
        passage_tokens = doc_tokens[passage_start_t:passage_end_t]
        passage_text = dictionary['document_text'][passage_start_c:passage_end_c]
        answers_clear, answers_tokenized = process_answers(dictionary['answers'], doc_offsets, passage_start_c, passage_start_t)
        clear_text = {'passage_text': passage_text, 'question_text': dictionary['question_text'], 'passage_id': passage_id, 'answers': answers_clear}
        tokenized = {'passage_start_t': passage_start_t, 'passage_tokens': passage_tokens, 'passage_offsets': passage_offsets, 'passage_start_of_word': passage_start_of_word, 'question_tokens': question_tokens, 'question_offsets': question_offsets, 'question_start_of_word': dictionary['question_start_of_word'][:max_query_len], 'answers': answers_tokenized, 'document_offsets': doc_offsets}
        samples.append(Sample(id=passage_id, clear_text=clear_text, tokenized=tokenized))
    return samples


DOWNSTREAM_TASK_MAP = {'gnad': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/gnad.tar.gz', 'germeval14': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval14.tar.gz', 'germeval18': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval18.tar.gz', 'squad20': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz', 'conll03detrain': 'https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train', 'conll03dedev': 'https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa', 'conll03detest': 'https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb', 'conll03entrain': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train', 'conll03endev': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa', 'conll03entest': 'https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb', 'cord_19': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cord_19.tar.gz', 'lm_finetune_nips': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/lm_finetune_nips.tar.gz', 'toxic-comments': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/toxic-comments.tar.gz', 'cola': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/cola.tar.gz', 'asnq_binary': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/asnq_binary.tar.gz', 'germeval17': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/germeval17.tar.gz', 'natural_questions': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/natural_questions.tar.gz'}


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


def read_squad_file(filename, proxies=None):
    """Read a SQuAD json file"""
    if not os.path.exists(filename):
        logger.info(f" Couldn't find {filename} locally. Trying to download ...")
        _download_extract_downstream_data(filename, proxies)
    with open(filename, 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    return input_data


def combine_vecs(question_vec, passage_vec, tokenizer, spec_tok_val=-1):
    """ Combine a question_vec and passage_vec in a style that is appropriate to the model. Will add slots in
    the returned vector for special tokens like [CLS] where the value is determine by spec_tok_val."""
    vec = tokenizer.build_inputs_with_special_tokens(token_ids_0=question_vec, token_ids_1=passage_vec)
    spec_toks_mask = tokenizer.get_special_tokens_mask(token_ids_0=question_vec, token_ids_1=passage_vec)
    combined = [(v if not special_token else spec_tok_val) for v, special_token in zip(vec, spec_toks_mask)]
    return combined


def convert_id(id_string):
    """
    Splits a string id into parts. If it is an id generated in the SQuAD pipeline it simple splits the id by the dashes
    and converts the parts to ints. If it is generated by the non-SQuAD pipeline, it splits the id by the dashes and
    converts references to "train" or "infer" into ints.
    :param id_string:
    :return:
    """
    ret = []
    datasets = ['train', 'infer']
    id_list = id_string.split('-')
    for x in id_list:
        if x in datasets:
            ret.append(datasets.index(x))
        else:
            ret.append(int(x))
    return ret


def answer_in_passage(start_idx, end_idx, passage_len):
    if passage_len > start_idx > 0 and passage_len > end_idx > 0:
        return True
    return False


def generate_labels(answers, passage_len_t, question_len_t, tokenizer, max_answers, answer_type_list=None):
    """
    Creates QA label for each answer in answers. The labels are the index of the start and end token
    relative to the passage. They are contained in an array of size (max_answers, 2).
    -1 used to fill array since there the number of answers is often less than max_answers.
    The index values take in to consideration the question tokens, and also special tokens such as [CLS].
    When the answer is not fully contained in the passage, or the question
    is impossible to answer, the start_idx and end_idx are 0 i.e. start and end are on the very first token
    (in most models, this is the [CLS] token). Note that in our implementation NQ has 4 labels
    ["is_impossible", "yes", "no", "span"] and this is what answer_type_list should look like"""
    label_idxs = np.full((max_answers, 2), fill_value=-1)
    answer_types = np.full(max_answers, fill_value=-1)
    if len(answers) == 0:
        label_idxs[(0), :] = 0
        answer_types[:] = 0
        return label_idxs, answer_types
    for i, answer in enumerate(answers):
        answer_type = answer['answer_type']
        start_idx = answer['start_t']
        end_idx = answer['end_t']
        start_vec_question = [0] * question_len_t
        end_vec_question = [0] * question_len_t
        start_vec_passage = [0] * passage_len_t
        end_vec_passage = [0] * passage_len_t
        if answer_in_passage(start_idx, end_idx, passage_len_t):
            start_vec_passage[start_idx] = 1
            end_vec_passage[end_idx] = 1
        start_vec = combine_vecs(start_vec_question, start_vec_passage, tokenizer, spec_tok_val=0)
        end_vec = combine_vecs(end_vec_question, end_vec_passage, tokenizer, spec_tok_val=0)
        start_label_present = 1 in start_vec
        end_label_present = 1 in end_vec
        if start_label_present is False and end_label_present is False:
            start_vec[0] = 1
            end_vec[0] = 1
            answer_type = 'is_impossible'
        elif start_label_present is False or end_label_present is False:
            raise Exception('The label vectors are lacking either a start or end label')
        assert sum(start_vec) == 1
        assert sum(end_vec) == 1
        start_idx = start_vec.index(1)
        end_idx = end_vec.index(1)
        label_idxs[i, 0] = start_idx
        label_idxs[i, 1] = end_idx
        if answer_type_list:
            answer_types[i] = answer_type_list.index(answer_type)
    assert np.max(label_idxs) > -1
    return label_idxs, answer_types


def get_roberta_seq_2_start(input_ids):
    first_backslash_s = input_ids.index(2)
    second_backslash_s = input_ids.index(2, first_backslash_s + 1)
    return second_backslash_s + 1


def sample_to_features_qa(sample, tokenizer, max_seq_len, answer_type_list=None, max_answers=6):
    """ Prepares data for processing by the model. Supports cases where there are
    multiple answers for the one question/document pair. max_answers is by default set to 6 since
    that is the most number of answers in the squad2.0 dev set."""
    question_tokens = sample.tokenized['question_tokens']
    question_start_of_word = sample.tokenized['question_start_of_word']
    question_len_t = len(question_tokens)
    passage_start_t = sample.tokenized['passage_start_t']
    passage_tokens = sample.tokenized['passage_tokens']
    passage_start_of_word = sample.tokenized['passage_start_of_word']
    passage_len_t = len(passage_tokens)
    answers = sample.tokenized['answers']
    sample_id = convert_id(sample.id)
    labels, answer_types = generate_labels(answers, passage_len_t, question_len_t, tokenizer, answer_type_list=answer_type_list, max_answers=max_answers)
    start_of_word = combine_vecs(question_start_of_word, passage_start_of_word, tokenizer, spec_tok_val=0)
    encoded = tokenizer.encode_plus(text=sample.tokenized['question_tokens'], text_pair=sample.tokenized['passage_tokens'], add_special_tokens=True, max_length=None, truncation_strategy='only_second', return_token_type_ids=True, return_tensors=None)
    input_ids = encoded['input_ids']
    segment_ids = encoded['token_type_ids']
    if tokenizer.__class__.__name__ in ['RobertaTokenizer', 'XLMRobertaTokenizer']:
        seq_2_start_t = get_roberta_seq_2_start(input_ids)
    else:
        seq_2_start_t = segment_ids.index(1)
    padding_mask = [1] * len(input_ids)
    pad_idx = tokenizer.pad_token_id
    padding = [pad_idx] * (max_seq_len - len(input_ids))
    zero_padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    padding_mask += zero_padding
    segment_ids += zero_padding
    start_of_word += zero_padding
    if tokenizer.__class__.__name__ in ['XLMRobertaTokenizer', 'RobertaTokenizer']:
        segment_ids = np.zeros_like(segment_ids)
    feature_dict = {'input_ids': input_ids, 'padding_mask': padding_mask, 'segment_ids': segment_ids, 'answer_type_ids': answer_types, 'id': sample_id, 'passage_start_t': passage_start_t, 'start_of_word': start_of_word, 'labels': labels, 'seq_2_start_t': seq_2_start_t}
    return [feature_dict]


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
        else:
            try:
                tokens_word = tokenizer.tokenize(w, add_prefix_space=True)
            except TypeError:
                tokens_word = tokenizer.tokenize(w)
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            orig_tok = re.sub(SPECIAL_TOKENIZER_CHARS, '', tok)
            w_off += len(orig_tok)
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)
    assert len(tokens) == len(token_offsets) == len(start_of_word)
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
    words = text.split(' ')
    word_offsets = []
    cumulated = 0
    for idx, word in enumerate(words):
        word_offsets.append(cumulated)
        cumulated += len(word) + 1
    tokens, offsets, start_of_word = _words_to_tokens(words, word_offsets, tokenizer)
    tokenized = {'tokens': tokens, 'offsets': offsets, 'start_of_word': start_of_word}
    return tokenized


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
        self.class_weights = class_weights
        if class_weights is not None:
            logger.info(f"Using class weights for task '{self.task_name}': {self.class_weights}")
            balanced_weights = nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        else:
            balanced_weights = None
        self.loss_fct = CrossEntropyLoss(weight=balanced_weights, reduction=loss_reduction, ignore_index=loss_ignore_index)
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path):
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

        """
        if os.path.exists(pretrained_model_name_or_path) and 'config.json' in pretrained_model_name_or_path and 'prediction_head' in pretrained_model_name_or_path:
            head = super(TextClassificationHead, cls).load(pretrained_model_name_or_path)
        else:
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.id2label)])
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            del full_model
        return head

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

    def logits_to_loss(self, logits, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.narrow(1, 0, 1)
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

    def formatted_preds(self, logits=None, preds_p=None, samples=None, return_class_probs=False, **kwargs):
        """ Like QuestionAnsweringHead.formatted_preds(), this fn can operate on either logits or preds_p. This
        is needed since at inference, the order of operations is very different depending on whether we are performing
        aggregation or not (compare Inferencer._get_predictions() vs Inferencer._get_predictions_and_aggregate())

        TODO: Preds_p should be renamed to preds"""
        assert logits is not None or preds_p is not None
        if logits is not None:
            preds_p = self.logits_to_preds(logits)
            probs = self.logits_to_probs(logits, return_class_probs)
        else:
            probs = [None] * len(preds_p)
        try:
            contexts = [sample.clear_text['text'] for sample in samples]
        except KeyError:
            contexts = [(sample.clear_text['question_text'] + ' | ' + sample.clear_text['passage_text']) for sample in samples]
        contexts_b = [sample.clear_text['text_b'] for sample in samples if 'text_b' in sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ['|'.join([a, b]) for a, b in zip(contexts, contexts_b)]
        assert len(preds_p) == len(probs) == len(contexts)
        res = {'task': 'text_classification', 'predictions': []}
        for pred, prob, context in zip(preds_p, probs, contexts):
            if not return_class_probs:
                pred_dict = {'start': None, 'end': None, 'context': f'{context}', 'label': f'{pred}', 'probability': prob}
            else:
                pred_dict = {'start': None, 'end': None, 'context': f'{context}', 'label': 'class_probabilities', 'probability': prob}
            res['predictions'].append(pred_dict)
        return res


def convert_iob_to_simple_tags(preds, spans):
    contains_named_entity = len([x for x in preds if 'B-' in x]) != 0
    simple_tags = []
    merged_spans = []
    open_tag = False
    for pred, span in zip(preds, spans):
        if not ('B-' in pred or 'I-' in pred):
            if open_tag:
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                open_tag = False
            continue
        elif 'B-' in pred:
            if open_tag:
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
            cur_tag = pred.replace('B-', '')
            cur_span = span
            open_tag = True
        elif 'I-' in pred:
            this_tag = pred.replace('I-', '')
            if open_tag and this_tag == cur_tag:
                cur_span['end'] = span['end']
            elif open_tag:
                merged_spans.append(cur_span)
                simple_tags.append(cur_tag)
                open_tag = False
    if open_tag:
        merged_spans.append(cur_span)
        simple_tags.append(cur_tag)
        open_tag = False
    if contains_named_entity and len(simple_tags) == 0:
        raise Exception('Predicted Named Entities lost when converting from IOB to simple tags. Please check the formatof the training data adheres to either adheres to IOB2 format or is converted when read_ner_file() is called.')
    return simple_tags, merged_spans


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
        self.generate_config()

    @classmethod
    def load(cls, pretrained_model_name_or_path):
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
            full_model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)
            head = cls(layer_dims=[full_model.config.hidden_size, len(full_model.config.label2id)])
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
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
        spans = []
        for sample, sample_preds in zip(samples, preds):
            word_spans = []
            span = None
            for token, offset, start_of_word in zip(sample.tokenized['tokens'], sample.tokenized['offsets'], sample.tokenized['start_of_word']):
                if start_of_word:
                    if span is not None:
                        word_spans.append(span)
                    span = {'start': offset, 'end': offset + len(token)}
                else:
                    span['end'] = offset + len(token.replace('##', ''))
            word_spans.append(span)
            spans.append(word_spans)
        assert len(preds) == len(probs) == len(spans)
        res = {'task': 'ner', 'predictions': []}
        for preds_seq, probs_seq, sample, spans_seq in zip(preds, probs, samples, spans):
            tags, spans_seq = convert_iob_to_simple_tags(preds_seq, spans_seq)
            seq_res = []
            for tag, prob, span in zip(tags, probs_seq, spans_seq):
                context = sample.clear_text['text'][span['start']:span['end']]
                seq_res.append({'start': span['start'], 'end': span['end'], 'context': f'{context}', 'label': f'{tag}', 'probability': prob})
            res['predictions'].extend(seq_res)
        return res


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
        assert len(self.lm_output_types) == len(self.prediction_heads)
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
            sequence_output, pooled_output = self.language_model(**kwargs, output_all_encoded_layers=False)
        else:
            self.language_model.enable_hidden_states_output()
            sequence_output, pooled_output, all_hidden_states = self.language_model(**kwargs)
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
        if len(self.prediction_heads) != 1:
            raise ValueError(f'Currently conversion only works for models with a SINGLE prediction head. Your model has {len(self.prediction_heads)}')
        elif len(self.prediction_heads[0].layer_dims) != 2:
            raise ValueError(f"""Currently conversion only works for PredictionHeads that are a single layer Feed Forward NN with dimensions [LM_output_dim, number_classes].
            Your PredictionHead has {str(self.prediction_heads[0].layer_dims)} dimensions.""")
        if self.prediction_heads[0].model_type == 'span_classification':
            transformers_model = AutoModelForQuestionAnswering.from_config(self.language_model.model.config)
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            transformers_model.qa_outputs.load_state_dict(self.prediction_heads[0].feed_forward.feed_forward[0].state_dict())
        elif self.prediction_heads[0].model_type == 'language_modelling':
            transformers_model = AutoModelWithLMHead.from_config(self.language_model.model.config)
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            ph_state_dict = self.prediction_heads[0].state_dict()
            ph_state_dict['transform.dense.weight'] = ph_state_dict.pop('dense.weight')
            ph_state_dict['transform.dense.bias'] = ph_state_dict.pop('dense.bias')
            ph_state_dict['transform.LayerNorm.weight'] = ph_state_dict.pop('LayerNorm.weight')
            ph_state_dict['transform.LayerNorm.bias'] = ph_state_dict.pop('LayerNorm.bias')
            transformers_model.cls.predictions.load_state_dict(ph_state_dict)
            logger.warning('Currently only the Masked Language Modeling component of the prediction head is converted, not the Next Sentence Prediction or Sentence Order Prediction components')
        elif self.prediction_heads[0].model_type == 'text_classification':
            if self.language_model.model.base_model_prefix == 'roberta':
                logger.error('Conversion for Text Classification with Roberta or XLMRoberta not possible at the moment.')
                raise NotImplementedError
            self.language_model.model.config.id2label = {id: label for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.label2id = {label: id for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.finetuning_task = 'text_classification'
            self.language_model.model.config.language = self.language_model.language
            self.language_model.model.config.num_labels = self.prediction_heads[0].num_labels
            transformers_model = AutoModelForSequenceClassification.from_config(self.language_model.model.config)
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            transformers_model.classifier.load_state_dict(self.prediction_heads[0].feed_forward.feed_forward[0].state_dict())
        elif self.prediction_heads[0].model_type == 'token_classification':
            self.language_model.model.config.id2label = {id: label for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.label2id = {label: id for id, label in enumerate(self.prediction_heads[0].label_list)}
            self.language_model.model.config.finetuning_task = 'token_classification'
            self.language_model.model.config.language = self.language_model.language
            self.language_model.model.config.num_labels = self.prediction_heads[0].num_labels
            transformers_model = AutoModelForTokenClassification.from_config(self.language_model.model.config)
            setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
            transformers_model.classifier.load_state_dict(self.prediction_heads[0].feed_forward.feed_forward[0].state_dict())
        else:
            raise NotImplementedError(f'FARM -> Transformers conversion is not supported yet for prediction heads of type {self.prediction_heads[0].model_type}')
        pass
        return transformers_model

    @classmethod
    def convert_from_transformers(cls, model_name_or_path, device, task_type, processor=None):
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
        lm = LanguageModel.load(model_name_or_path)
        if task_type == 'question_answering':
            ph = QuestionAnsweringHead.load(model_name_or_path)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1, lm_output_types='per_token', device=device)
        elif task_type == 'text_classification':
            if 'roberta' in model_name_or_path:
                logger.error('Conversion for Text Classification with Roberta or XLMRoberta not possible at the moment.')
                raise NotImplementedError
            ph = TextClassificationHead.load(model_name_or_path)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1, lm_output_types='per_sequence', device=device)
        elif task_type == 'ner':
            ph = TokenClassificationHead.load(model_name_or_path)
            adaptive_model = cls(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1, lm_output_types='per_token', device=device)
        elif task_type == 'embeddings':
            adaptive_model = cls(language_model=lm, prediction_heads=[], embeds_dropout_prob=0.1, lm_output_types=['per_token', 'per_sequence'], device=device)
        else:
            raise NotImplementedError(f"Huggingface's transformer models of type {task_type} are not supported yet")
        if processor:
            adaptive_model.connect_heads_with_processor(processor.tasks)
        return adaptive_model

    def convert_to_onnx(self, output_path, opset_version=11, optimize_for=None):
        """
        Convert a PyTorch AdaptiveModel to ONNX.

        The conversion is trace-based by performing a forward pass on the model with a input batch.

        :param output_path: model dir to write the model and config files
        :type output_path: Path
        :param opset_version: ONNX opset version
        :type opset_version: int
        :param optimize_for: optimize the exported model for a target device. Available options
                             are "gpu_tensor_core" (GPUs with tensor core like V100 or T4),
                             "gpu_without_tensor_core" (most other GPUs), and "cpu".
        :type optimize_for: str
        :return:
        """
        if type(self.prediction_heads[0]) is not QuestionAnsweringHead:
            raise NotImplementedError
        tokenizer = Tokenizer.load(pretrained_model_name_or_path='deepset/bert-base-cased-squad2')
        label_list = ['start_token', 'end_token']
        metric = 'squad'
        max_seq_len = 384
        batch_size = 1
        processor = SquadProcessor(tokenizer=tokenizer, max_seq_len=max_seq_len, label_list=label_list, metric=metric, train_filename='stub-file', dev_filename=None, test_filename=None, data_dir='stub-dir')
        data_silo = DataSilo(processor=processor, batch_size=1, distributed=False, automatic_loading=False)
        sample_dict = [{'context': 'The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia.', 'qas': [{'question': 'In what country is Normandy located?', 'id': '56ddde6b9a695914005b9628', 'answers': [{'text': 'France', 'answer_start': 159}], 'is_impossible': False}]}]
        data_silo._load_data(train_dicts=sample_dict)
        data_loader = data_silo.get_data_loader('train')
        data = next(iter(data_loader))
        data = list(data.values())
        inputs = {'input_ids': data[0].reshape(batch_size, max_seq_len), 'padding_mask': data[1].reshape(batch_size, max_seq_len), 'segment_ids': data[2].reshape(batch_size, max_seq_len)}
        model = ONNXWrapper.load_from_adaptive_model(self)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with torch.no_grad():
            symbolic_names = {(0): 'batch_size', (1): 'max_seq_len'}
            torch.onnx.export(model, args=tuple(inputs.values()), f=output_path / 'model.onnx'.format(opset_version), opset_version=opset_version, do_constant_folding=True, input_names=['input_ids', 'padding_mask', 'segment_ids'], output_names=['logits'], dynamic_axes={'input_ids': symbolic_names, 'padding_mask': symbolic_names, 'segment_ids': symbolic_names, 'logits': symbolic_names})
        if optimize_for:
            optimize_args = Namespace(disable_attention=False, disable_bias_gelu=False, disable_embed_layer_norm=False, opt_level=99, disable_skip_layer_norm=False, disable_bias_skip_layer_norm=False, hidden_size=768, verbose=False, input='onnx-export/model.onnx', model_type='bert', num_heads=12, output='onnx-export/model.onnx')
            if optimize_for == 'gpu_tensor_core':
                optimize_args.float16 = True
                optimize_args.input_int32 = True
            elif optimize_for == 'gpu_without_tensor_core':
                optimize_args.float16 = False
                optimize_args.input_int32 = True
            elif optimize_for == 'cpu':
                logger.info('')
                optimize_args.float16 = False
                optimize_args.input_int32 = False
            else:
                raise NotImplementedError(f"ONNXRuntime model optimization is not available for {optimize_for}. Choose one of 'gpu_tensor_core'(V100 or T4), 'gpu_without_tensor_core' or 'cpu'.")
            optimize_onnx_model(optimize_args)
        else:
            logger.info("Exporting unoptimized ONNX model. To enable optimization, supply 'optimize_for' parameter with the target device.'")
        for i, ph in enumerate(self.prediction_heads):
            ph.save_config(output_path, i)
        processor.save(output_path)
        onnx_model_config = {'onnx_opset_version': opset_version, 'language': self.get_language()}
        with open(output_path / 'model_config.json', 'w') as f:
            json.dump(onnx_model_config, f)
        logger.info(f'Model exported at path {output_path}')


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
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


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
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


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
        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = output_tuple[0], output_tuple[1], output_tuple[2]
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False


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
            config = AlbertConfig.from_pretrained(farm_lm_config)
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
        self.model.embeddings = self.model.embeddings[(valid_tok_indices), :]
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
        assert len(preds) == len(contexts)
        res = {'task': 'regression', 'predictions': []}
        for pred, context in zip(preds, contexts):
            res['predictions'].append({'context': f'{context}', 'pred': pred[0]})
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
        assert len(preds) == len(probs) == len(contexts)
        res = {'task': 'text_classification', 'predictions': []}
        for pred, prob, context in zip(preds, probs, contexts):
            res['predictions'].append({'start': None, 'end': None, 'context': f'{context}', 'label': f'{pred}', 'probability': prob})
        return res


class BertLMHead(PredictionHead):

    def __init__(self, hidden_size, vocab_size, hidden_act='gelu', task_name='lm', **kwargs):
        super(BertLMHead, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.num_labels = vocab_size
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
    def load(cls, pretrained_model_name_or_path, n_added_tokens=0):
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
            if n_added_tokens != 0:
                raise NotImplementedError('Custom vocab not yet supported for model loading from FARM files')
            head = super(BertLMHead, cls).load(pretrained_model_name_or_path)
        else:
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)
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
        assert lm_preds_ids.shape == lm_label_ids.shape
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
    def load(cls, pretrained_model_name_or_path):
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
            bert_with_lm = BertForPreTraining.from_pretrained(pretrained_model_name_or_path)
            head = cls(layer_dims=[bert_with_lm.config.hidden_size, 2], loss_ignore_index=-1, task_name='nextsentence')
            head.feed_forward.feed_forward[0].load_state_dict(bert_with_lm.cls.seq_relationship.state_dict())
            del bert_with_lm
        return head


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForwardBlock,
     lambda: ([], {'layer_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WrappedDataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_deepset_ai_FARM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

