import sys
_module = sys.modules[__name__]
del sys
change_api_category_id = _module
generate_openapi_specs = _module
release_docs = _module
conftest = _module
conf = _module
pydoc = _module
renderers = _module
convert_ipynb = _module
headers = _module
haystack = _module
document_stores = _module
base = _module
deepsetcloud = _module
elasticsearch = _module
es_converter = _module
faiss = _module
filter_utils = _module
graphdb = _module
memory = _module
memory_knowledgegraph = _module
milvus = _module
opensearch = _module
pinecone = _module
search_engine = _module
sql = _module
utils = _module
weaviate = _module
environment = _module
errors = _module
modeling = _module
data_handler = _module
data_silo = _module
dataloader = _module
dataset = _module
input_features = _module
inputs = _module
processor = _module
samples = _module
evaluation = _module
eval = _module
metrics = _module
squad = _module
infer = _module
model = _module
adaptive_model = _module
biadaptive_model = _module
feature_extraction = _module
language_model = _module
multimodal = _module
base = _module
sentence_transformers = _module
optimization = _module
prediction_head = _module
predictions = _module
triadaptive_model = _module
training = _module
base = _module
dpr = _module
question_answering = _module
utils = _module
visual = _module
nodes = _module
_json_schema = _module
answer_generator = _module
openai = _module
transformers = _module
audio = _module
_text_to_speech = _module
answer_to_speech = _module
document_to_speech = _module
connector = _module
crawler = _module
document_classifier = _module
transformers = _module
evaluator = _module
extractor = _module
entity = _module
file_classifier = _module
file_type = _module
file_converter = _module
azure = _module
docx = _module
image = _module
markdown = _module
parsr = _module
pdf = _module
tika = _module
txt = _module
label_generator = _module
pseudo_label_generator = _module
other = _module
docs2answers = _module
document_merger = _module
join = _module
join_answers = _module
join_docs = _module
route_documents = _module
preprocessor = _module
prompt = _module
prompt_node = _module
query_classifier = _module
sklearn = _module
transformers = _module
question_generator = _module
question_generator = _module
ranker = _module
sentence_transformers = _module
reader = _module
farm = _module
table = _module
transformers = _module
retriever = _module
_embedding_encoder = _module
_losses = _module
dense = _module
embedder = _module
retriever = _module
sparse = _module
text2sparql = _module
summarizer = _module
transformers = _module
translator = _module
transformers = _module
pipelines = _module
config = _module
ray = _module
standard_pipelines = _module
schema = _module
telemetry = _module
augment_squad = _module
cleaning = _module
context_matching = _module
doc_store = _module
docker = _module
early_stopping = _module
experiment_tracking = _module
export_utils = _module
import_utils = _module
labels = _module
preprocessing = _module
reflection = _module
squad_data = _module
squad_to_dpr = _module
torch_utils = _module
__about__ = _module
rest_api = _module
application = _module
controller = _module
document = _module
http_error = _module
feedback = _module
file_upload = _module
health = _module
search = _module
pipeline = _module
custom_component = _module
test = _module
test_rest_api = _module
embeddings_slice = _module
shuffle_passages = _module
model_distillation = _module
nq_to_squad = _module
results_to_json = _module
retriever_simplified = _module
run = _module
templates = _module
test_base = _module
test_deepsetcloud = _module
test_document_store = _module
test_elasticsearch = _module
test_faiss = _module
test_knowledge_graph = _module
test_memory = _module
test_milvus = _module
test_opensearch = _module
test_pinecone = _module
test_search_engine = _module
test_sql = _module
test_sql_based = _module
test_weaviate = _module
_test_feature_extraction_end2end = _module
test_distillation = _module
test_dpr = _module
test_feature_extraction = _module
test_inference = _module
test_language = _module
test_prediction_head = _module
test_processor = _module
test_processor_save_load = _module
test_question_answering = _module
test_audio = _module
test_connector = _module
test_document_classifier = _module
test_document_merger = _module
test_extractor = _module
test_extractor_translation = _module
test_file_converter = _module
test_filetype_classifier = _module
test_generator = _module
test_label_generator = _module
test_other = _module
test_preprocessor = _module
test_prompt_node = _module
test_query_classifier = _module
test_question_generator = _module
test_ranker = _module
test_reader = _module
test_retriever = _module
test_summarizer = _module
test_table_reader = _module
test_translator = _module
others = _module
test_schema = _module
test_squad_data = _module
test_telemetry = _module
test_utils = _module
test_eval = _module
test_eval_batch = _module
test_pipeline = _module
test_pipeline_debug_and_validation = _module
test_pipeline_extractive_qa = _module
test_pipeline_yaml = _module
test_ray = _module
test_standard_pipelines = _module
test_ui_utils = _module
ui = _module
webapp = _module

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


from typing import Any


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


from typing import Generator


import time


import logging


from copy import deepcopy


from collections import defaultdict


import re


import numpy as np


import torch


from typing import TYPE_CHECKING


from typing import Tuple


import random


from itertools import groupby


from torch.utils.data import ConcatDataset


from torch.utils.data import Dataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from math import ceil


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


import numbers


from torch.utils.data import TensorDataset


from typing import Iterable


from typing import Type


import uuid


import inspect


from inspect import signature


from abc import ABC


from abc import abstractmethod


from torch.nn import DataParallel


from typing import Set


import copy


from typing import Callable


import numpy


from torch import nn


from torch.nn.parallel import DistributedDataParallel


from torch import optim


from torch.nn import CrossEntropyLoss


from torch.nn import NLLLoss


from scipy.special import expit


from torch.optim.lr_scheduler import _LRScheduler


from torch.nn import MSELoss


from torch.nn import Linear


from torch.nn import Module


from torch.nn import ModuleList


import torch.nn.functional as F


from torch.optim import Optimizer


from functools import wraps


from itertools import islice


import torch.distributed as dist


from torch import multiprocessing as mp


from collections.abc import Callable


import itertools


from string import Template


from typing import Iterator


from time import perf_counter


import pandas as pd


from copy import copy


from torch.nn import functional as F


from torch.utils.data import SequentialSampler


class NonPrivateParameters:
    param_names: List[str] = ['top_k', 'model_name_or_path', 'add_isolated_node_eval', 'fingerprint', 'type', 'uptime', 'run_total', 'run_total_window', 'message']

    @classmethod
    def apply_filter(cls, param_dicts: Dict[str, Any]) ->Dict[str, Any]:
        """
        Ensures that only the values of non-private parameters are sent in events. All other parameter values are filtered out before sending an event.
        If model_name_or_path is a local file path, it will be reduced to the name of the file. The directory names are not sent.

        :param param_dicts: the keyword arguments that need to be filtered before sending an event
        """
        tracked_params = {k: param_dicts[k] for k in cls.param_names if k in param_dicts}
        if 'model_name_or_path' in tracked_params:
            if Path(tracked_params['model_name_or_path']).is_file() or tracked_params['model_name_or_path'].count(os.path.sep) > 1:
                tracked_params['model_name_or_path'] = Path(tracked_params['model_name_or_path']).name
        return tracked_params


logger = logging.getLogger(__name__)


def _read_telemetry_config():
    """
    Loads the config from the file specified in CONFIG_PATH
    """
    global user_id
    try:
        if not CONFIG_PATH.is_file():
            return
        with open(CONFIG_PATH, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
            if 'user_id' in config and user_id is None:
                user_id = config['user_id']
    except Exception as e:
        logger.debug('Telemetry was not able to read the config file %s', CONFIG_PATH, exc_info=e)


def _write_telemetry_config():
    """
    Writes a config file storing the randomly generated user id and whether to write events to a log file.
    This method logs an info to inform the user about telemetry when it is used for the first time.
    """
    global user_id
    try:
        if not CONFIG_PATH.is_file():
            logger.info(f'Haystack sends anonymous usage data to understand the actual usage and steer dev efforts towards features that are most meaningful to users. You can opt-out at anytime by calling disable_telemetry() or by manually setting the environment variable HAYSTACK_TELEMETRY_ENABLED as described for different operating systems on the documentation page. More information at https://docs.haystack.deepset.ai/docs/telemetry')
            CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
        user_id = _get_or_create_user_id()
        config = {'user_id': user_id}
        with open(CONFIG_PATH, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    except Exception:
        logger.debug('Could not write config file to %s', CONFIG_PATH)
        send_custom_event(event='config saving failed')


HAYSTACK_TELEMETRY_ENABLED = 'HAYSTACK_TELEMETRY_ENABLED'


def is_telemetry_enabled() ->bool:
    """
    Returns False if telemetry is disabled via an environment variable, otherwise True.
    """
    telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_ENABLED, 'True')
    return telemetry_environ.lower() != 'false'


def _get_or_create_user_id() ->Optional[str]:
    """
    Randomly generates a user id or loads the id defined in the config file and returns it.
    Returns None if no id has been set previously and a new one cannot be stored because telemetry is disabled
    """
    global user_id
    if user_id is None:
        _read_telemetry_config()
        if user_id is None and is_telemetry_enabled():
            user_id = str(uuid.uuid4())
            _write_telemetry_config()
    return user_id


def _write_event_to_telemetry_log_file(distinct_id: str, event: str, properties: Dict[str, Any]):
    try:
        with open(LOG_PATH, 'a') as file_object:
            file_object.write(f'{event}, {properties}, {distinct_id}\n')
    except Exception as e:
        logger.debug('Telemetry was not able to write event to log file %s', LOG_PATH, exc_info=e)


HAYSTACK_EXECUTION_CONTEXT = 'HAYSTACK_EXECUTION_CONTEXT'


HAYSTACK_DOCKER_CONTAINER = 'HAYSTACK_DOCKER_CONTAINER'


def _get_execution_environment():
    """
    Identifies the execution environment that Haystack is running in.
    Options are: colab notebook, kubernetes, CPU/GPU docker container, test environment, jupyter notebook, python script
    """
    if os.environ.get('CI', 'False').lower() == 'true':
        execution_env = 'ci'
    elif 'google.colab' in sys.modules:
        execution_env = 'colab'
    elif 'KUBERNETES_SERVICE_HOST' in os.environ:
        execution_env = 'kubernetes'
    elif HAYSTACK_DOCKER_CONTAINER in os.environ:
        execution_env = os.environ.get(HAYSTACK_DOCKER_CONTAINER)
    elif 'pytest' in sys.modules:
        execution_env = 'test'
    else:
        try:
            execution_env = get_ipython().__class__.__name__
        except NameError:
            execution_env = 'script'
    return execution_env


def get_or_create_env_meta_data() ->Dict[str, Any]:
    """
    Collects meta data about the setup that is used with Haystack, such as: operating system, python version, Haystack version, transformers version, pytorch version, number of GPUs, execution environment, and the value stored in the env variable HAYSTACK_EXECUTION_CONTEXT.
    """
    global env_meta_data
    if not env_meta_data:
        env_meta_data = {'os_version': platform.release(), 'os_family': platform.system(), 'os_machine': platform.machine(), 'python_version': platform.python_version(), 'haystack_version': __version__, 'transformers_version': transformers.__version__, 'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda if torch.cuda.is_available() else 0, 'n_gpu': torch.cuda.device_count() if torch.cuda.is_available() else 0, 'n_cpu': os.cpu_count(), 'context': os.environ.get(HAYSTACK_EXECUTION_CONTEXT), 'execution_env': _get_execution_environment()}
    return env_meta_data


HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED = 'HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED'


def is_telemetry_logging_to_file_enabled() ->bool:
    """
    Returns False if logging telemetry events to a file is disabled via an environment variable, otherwise True.
    """
    telemetry_environ = os.environ.get(HAYSTACK_TELEMETRY_LOGGING_TO_FILE_ENABLED, 'False')
    return telemetry_environ.lower() != 'false'


def send_custom_event(event: str='', payload: Dict[str, Any]={}):
    """
    This method can be called directly from anywhere in Haystack to send an event.
    Enriches the given event with metadata and sends it to the posthog server if telemetry is enabled.
    If telemetry has just been disabled, a final event is sent and the config file and the log file are deleted

    :param event: Name of the event. Use a noun and a verb, e.g., "evaluation started", "component created"
    :param payload: A dictionary containing event meta data, e.g., parameter settings
    """
    global user_id
    try:

        def send_request(payload: Dict[str, Any]):
            """
            Prepares and sends an event in a post request to a posthog server
            Sending the post request within posthog.capture is non-blocking.

            :param payload: A dictionary containing event meta data, e.g., parameter settings
            """
            event_properties = {**NonPrivateParameters.apply_filter(payload), **get_or_create_env_meta_data()}
            if user_id is None:
                raise RuntimeError('User id was not initialized')
            try:
                posthog.capture(distinct_id=user_id, event=event, properties=event_properties)
            except Exception as e:
                logger.debug('Telemetry was not able to make a post request to posthog.', exc_info=e)
            if is_telemetry_enabled() and is_telemetry_logging_to_file_enabled():
                _write_event_to_telemetry_log_file(distinct_id=user_id, event=event, properties=event_properties)
        user_id = _get_or_create_user_id()
        if is_telemetry_enabled():
            send_request(payload=payload)
        elif CONFIG_PATH.exists():
            event = 'telemetry disabled'
            send_request(payload={})
            _delete_telemetry_file(TelemetryFileType.CONFIG_FILE)
            _delete_telemetry_file(TelemetryFileType.LOG_FILE)
        else:
            return
    except Exception as e:
        logger.debug('Telemetry was not able to send an event.', exc_info=e)


class HaystackError(Exception):
    """
    Any error generated by Haystack.

    This error wraps its source transparently in such a way that its attributes
    can be accessed directly: for example, if the original error has a `message` attribute,
    `HaystackError.message` will exist and have the expected content.
    If send_message_in_event is set to True (default), the message will be sent as part of a telemetry event reporting the error.
    The messages of errors that might contain user-specific information will not be sent, e.g., DocumentStoreError or OpenAIError.
    """

    def __init__(self, message: Optional[str]=None, docs_link: Optional[str]=None, send_message_in_event: bool=True):
        payload = {'message': message} if send_message_in_event else {}
        send_custom_event(event=f'{type(self).__name__} raised', payload=payload)
        super().__init__()
        if message:
            self.message = message
        self.docs_link = None

    def __getattr__(self, attr):
        getattr(self.__cause__, attr)

    def __str__(self):
        if self.docs_link:
            docs_message = f'\n\nCheck out the documentation at {self.docs_link}'
            return self.message + docs_message
        return self.message

    def __repr__(self):
        return str(self)


class ModelingError(HaystackError):
    """Exception for issues raised by the modeling module"""

    def __init__(self, message: Optional[str]=None, docs_link: Optional[str]='https://haystack.deepset.ai/'):
        super().__init__(message=message, docs_link=docs_link)


OUTPUT_DIM_NAMES = ['dim', 'hidden_size', 'd_model']


class FeedForwardBlock(nn.Module):
    """
    A feed forward neural network of variable depth and width.
    """

    def __init__(self, layer_dims: List[int], **kwargs):
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

    def forward(self, X: torch.Tensor):
        logits = self.feed_forward(X)
        return logits


def _is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False


class AutoTokenizer:
    mocker: MagicMock = MagicMock()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.mocker.from_pretrained(*args, **kwargs)
        return cls()


SAMPLE = """
      .--.        _____                       _      
    .'_\\/_'.     / ____|                     | |     
    '. /\\ .'    | (___   __ _ _ __ ___  _ __ | | ___ 
      "||"       \\___ \\ / _` | '_ ` _ \\| '_ \\| |/ _ \\ 
       || /\\     ____) | (_| | | | | | | |_) | |  __/
    /\\ ||//\\)   |_____/ \\__,_|_| |_| |_| .__/|_|\\___|
   (/\\\\||/                             |_|           
______\\||/___________________________________________                     
"""


class Sample:
    """A single training/test sample. This should contain the input and the label. Is initialized with
    the human readable clear_text. Over the course of data preprocessing, this object is populated
    with tokenized and featurized versions of the data."""

    def __init__(self, id: str, clear_text: dict, tokenized: Optional[dict]=None, features: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]=None):
        """
        :param id: The unique id of the sample
        :param clear_text: A dictionary containing various human readable fields (e.g. text, label).
        :param tokenized: A dictionary containing the tokenized version of clear text plus helpful meta data: offsets (start position of each token in the original text) and start_of_word (boolean if a token is the first one of a word).
        :param features: A dictionary containing features in a vectorized format needed by the model to process this sample.
        """
        self.id = id
        self.clear_text = clear_text
        self.features = features
        self.tokenized = tokenized

    def __str__(self):
        if self.clear_text:
            clear_text_str = '\n \t'.join([(k + ': ' + str(v)) for k, v in self.clear_text.items()])
            if len(clear_text_str) > 3000:
                clear_text_str = clear_text_str[:3000] + f'\nTHE REST IS TOO LONG TO DISPLAY. Remaining chars :{len(clear_text_str) - 3000}'
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
            if len(tokenized_str) > 3000:
                tokenized_str = tokenized_str[:3000] + f'\nTHE REST IS TOO LONG TO DISPLAY. Remaining chars: {len(tokenized_str) - 3000}'
        else:
            tokenized_str = 'None'
        s = f'\n{SAMPLE}\nID: {self.id}\nClear Text: \n \t{clear_text_str}\nTokenized: \n \t{tokenized_str}\nFeatures: \n \t{feature_str}\n_____________________________________________________'
        return s


class SampleBasket:
    """An object that contains one source text and the one or more samples that will be processed. This
    is needed for tasks like question answering where the source text can generate multiple input - label
    pairs."""

    def __init__(self, id_internal: Optional[Union[int, str]], raw: dict, id_external: Optional[str]=None, samples: Optional[List[Sample]]=None):
        """
        :param id_internal: A unique identifying id. Used for identification within Haystack.
        :param external_id: Used for identification outside of Haystack. E.g. if another framework wants to pass along its own id with the results.
        :param raw: Contains the various data needed to form a sample. It is ideally in human readable form.
        :param samples: An optional list of Samples used to populate the basket at initialization.
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
            logger.debug("Could not determine type for feature '%s'. Converting now to a tensor of default type long.", t_name)
        cur_tensor = torch.as_tensor(np.array([sample[t_name] for sample in features]), dtype=torch.long)
        all_tensors.append(cur_tensor)
    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


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
        inputs = tokenizer(tokens_a, tokens_b, add_special_tokens=True, truncation=False, return_token_type_ids=True, is_split_into_words=False)
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


def truncate_sequences(seq_a: list, seq_b: Optional[list], tokenizer: AutoTokenizer, max_seq_len: int, truncation_strategy: str='longest_first', with_special_tokens: bool=True, stride: int=0) ->Tuple[List[Any], Optional[List[Any]], List[Any]]:
    """
    Reduces a single sequence or a pair of sequences to a maximum sequence length.
    The sequences can contain tokens or any other elements (offsets, masks ...).
    If `with_special_tokens` is enabled, it'll remove some additional tokens to have exactly
    enough space for later adding special tokens (CLS, SEP etc.)

    Supported truncation strategies:

    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under
        max_length starting from the longest one at each token (when there is a pair of input sequences).
        Overflowing tokens only contains overflow from the first sequence.
    - only_first: Only truncate the first sequence. raise an error if the first sequence is
        shorter or equal to than num_tokens_to_remove.
    - only_second: Only truncate the second sequence
    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)

    :param seq_a: First sequence of tokens/offsets/...
    :param seq_b: Optional second sequence of tokens/offsets/...
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :param max_seq_len:
    :param truncation_strategy: how the sequence(s) should be truncated down.
        Default: "longest_first" (see above for other options).
    :param with_special_tokens: If true, it'll remove some additional tokens to have exactly enough space
        for later adding special tokens (CLS, SEP etc.)
    :param stride: optional stride of the window during truncation
    :return: truncated seq_a, truncated seq_b, overflowing tokens
    """
    pair = seq_b is not None
    len_a = len(seq_a)
    len_b = len(seq_b) if seq_b is not None else 0
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=pair) if with_special_tokens else 0
    total_len = len_a + len_b + num_special_tokens
    overflowing_tokens = []
    if max_seq_len and total_len > max_seq_len:
        seq_a, seq_b, overflowing_tokens = tokenizer.truncate_sequences(seq_a, pair_ids=seq_b, num_tokens_to_remove=total_len - max_seq_len, truncation_strategy=truncation_strategy, stride=stride)
    return seq_a, seq_b, overflowing_tokens


DOWNSTREAM_TASK_MAP = {'squad20': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/squad20.tar.gz', 'covidqa': 'https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-downstream/covidqa.tar.gz'}


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


def _download_extract_downstream_data(input_file: str, proxies=None):
    full_path = Path(os.path.realpath(input_file))
    directory = full_path.parent
    taskname = directory.stem
    datadir = directory.parent
    logger.info('downloading and extracting file {} to dir {}'.format(taskname, datadir))
    if taskname not in DOWNSTREAM_TASK_MAP:
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
            tfile = tarfile.open(temp_file.name)
            tfile.extractall(datadir)


def _read_squad_file(filename: str, proxies=None):
    """Read a SQuAD json file"""
    if not os.path.exists(filename):
        logger.info("Couldn't find %s locally. Trying to download ...", filename)
        _download_extract_downstream_data(filename, proxies)
    with open(filename, 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)['data']
    return input_data


def get_passage_offsets(doc_offsets, doc_stride, passage_len_t, doc_text):
    """
    Get spans (start and end offsets) for passages by applying a sliding window function.
    The sliding window moves in steps of doc_stride.
    Returns a list of dictionaries which each describe the start, end and id of a passage
    that is formed when chunking a document using a sliding window approach."""
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
    """Returns the idx of the token at the given character idx"""
    if ch_idx >= np.max(token_offsets):
        idx = np.argmax(token_offsets)
    else:
        idx = np.argmax(token_offsets > ch_idx) - 1
    return idx


def _get_start_of_word_QA(word_ids):
    return [1] + list(np.ediff1d(np.asarray(word_ids, dtype='int16')))


class QACandidate:
    """
    A single QA candidate answer.
    """

    def __init__(self, answer_type: str, score: float, offset_answer_start: int, offset_answer_end: int, offset_unit: str, aggregation_level: str, probability: Optional[float]=None, n_passages_in_doc: Optional[int]=None, passage_id: Optional[str]=None, confidence: Optional[float]=None):
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

    def set_context_window(self, context_window_size: int, clear_text: str):
        window_str, start_ch, end_ch = self._create_context_window(context_window_size, clear_text)
        self.context_window = window_str
        self.offset_context_window_start = start_ch
        self.offset_context_window_end = end_ch

    def set_answer_string(self, token_offsets: List[int], document_text: str):
        pred_str, self.offset_answer_start, self.offset_answer_end = self._span_to_string(token_offsets, document_text)
        self.offset_unit = 'char'
        self._add_answer(pred_str)

    def _add_answer(self, string: str):
        """
        Set the answer string. This method will check that the answer given is valid given the start
        and end indices that are stored in the object.
        """
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

    def _create_context_window(self, context_window_size: int, clear_text: str) ->Tuple[str, int, int]:
        """
        Extract from the clear_text a window that contains the answer and (usually) some amount of text on either
        side of the answer. Useful for cases where the answer and its surrounding context needs to be
        displayed in a UI. If the self.context_window_size is smaller than the extracted answer, it will be
        enlarged so that it can contain the answer

        :param context_window_size: The size of the context window to be generated. Note that the window size may be increased if the answer is longer.
        :param clear_text: The text from which the answer is extracted
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

    def _span_to_string(self, token_offsets: List[int], clear_text: str) ->Tuple[str, int, int]:
        """
        Generates a string answer span using self.offset_answer_start and self.offset_answer_end. If the candidate
        is a no answer, an empty string is returned

        :param token_offsets: A list of ints which give the start character index of the corresponding token
        :param clear_text: The text from which the answer span is to be extracted
        :return: The string answer span, followed by the start and end character indices
        """
        if self.offset_unit != 'token':
            logger.error(f'QACandidate needs to have self.offset_unit=token before calling _span_to_string() (id = {self.passage_id})')
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
        cleaned_final_text = final_text.strip()
        if not cleaned_final_text:
            return '', 0, 0
        left_offset = len(final_text) - len(final_text.lstrip())
        if left_offset:
            start_ch = start_ch + left_offset
        end_ch = start_ch + len(cleaned_final_text)
        return cleaned_final_text, start_ch, end_ch

    def to_doc_level(self, start: int, end: int):
        """
        Populate the start and end indices with document level indices. Changes aggregation level to 'document'
        """
        self.offset_answer_start = start
        self.offset_answer_end = end
        self.aggregation_level = 'document'

    def to_list(self) ->List[Optional[Union[str, int, float]]]:
        return [self.answer, self.offset_answer_start, self.offset_answer_end, self.score, self.passage_id]


class Pred(ABC):
    """
    Abstract base class for predictions of every task
    """

    def __init__(self, id: str, prediction: List[Any], context: str):
        self.id = id
        self.prediction = prediction
        self.context = context

    def to_json(self):
        raise NotImplementedError


class QAPred(Pred):
    """
    A set of QA predictions for a passage or a document. The candidates are stored in QAPred.prediction which is a
    list of QACandidate objects. Also contains all attributes needed to convert the object into json format and also
    to create a context window for a UI
    """

    def __init__(self, id: str, prediction: List[QACandidate], context: str, question: str, token_offsets: List[int], context_window_size: int, aggregation_level: str, no_answer_gap: float, ground_truth_answer: Optional[str]=None, answer_types: List[str]=[]):
        """
        :param id: The id of the passage or document
        :param prediction: A list of QACandidate objects for the given question and document
        :param context: The text passage from which the answer can be extracted
        :param question: The question being posed
        :param token_offsets: A list of ints indicating the start char index of each token
        :param context_window_size: The number of chars in the text window around the answer
        :param aggregation_level: States whether this candidate and its indices are on a passage level (pre aggregation) or on a document level (post aggregation)
        :param no_answer_gap: How much the QuestionAnsweringHead.no_ans_boost needs to change to turn a no_answer to a positive answer
        :param ground_truth_answer: Ground truth answers
        :param answer_types: List of answer_types supported by this task e.g. ["span", "yes_no", "no_answer"]
        """
        super().__init__(id, prediction, context)
        self.question = question
        self.token_offsets = token_offsets
        self.context_window_size = context_window_size
        self.aggregation_level = aggregation_level
        self.answer_types = answer_types
        self.ground_truth_answer = ground_truth_answer
        self.no_answer_gap = no_answer_gap
        self.n_passages = self.prediction[0].n_passages_in_doc
        for qa_candidate in self.prediction:
            qa_candidate.set_answer_string(token_offsets, self.context)
            qa_candidate.set_context_window(self.context_window_size, self.context)

    def to_json(self, squad=False) ->Dict:
        """
        Converts the information stored in the object into a json format.

        :param squad: If True, no_answers are represented by the empty string instead of "no_answer"
        """
        answers = self._answers_to_json(self.id, squad)
        ret = {'task': 'qa', 'predictions': [{'question': self.question, 'id': self.id, 'ground_truth': self.ground_truth_answer, 'answers': answers, 'no_ans_gap': self.no_answer_gap}]}
        if squad:
            del ret['predictions'][0]['id']
            ret['predictions'][0]['question_id'] = self.id
        return ret

    def _answers_to_json(self, ext_id, squad=False) ->List[Dict]:
        """
        Convert all answers into a json format

        :param id: ID of the question document pair
        :param squad: If True, no_answers are represented by the empty string instead of "no_answer"
        """
        ret = []
        for qa_candidate in self.prediction:
            if squad and qa_candidate.answer == 'no_answer':
                answer_string = ''
            else:
                answer_string = qa_candidate.answer
            curr = {'score': qa_candidate.score, 'probability': None, 'answer': answer_string, 'offset_answer_start': qa_candidate.offset_answer_start, 'offset_answer_end': qa_candidate.offset_answer_end, 'context': qa_candidate.context_window, 'offset_context_start': qa_candidate.offset_context_window_start, 'offset_context_end': qa_candidate.offset_context_window_end, 'document_id': ext_id}
            ret.append(curr)
        return ret

    def to_squad_eval(self) ->Dict:
        return self.to_json(squad=True)


def try_get(keys, dictionary):
    try:
        for key in keys:
            if key in dictionary:
                ret = dictionary[key]
                if type(ret) == list:
                    ret = ret[0]
                return ret
    except Exception as e:
        logger.warning('Cannot extract from dict %s with error: %s', dictionary, e)
    return None


def is_supported_model(model_type: Optional[str]):
    """
    Returns whether the model type is supported by Haystack
    :param model_type: the model_type as found in the config file
    :return: whether the model type is supported by the Haystack
    """
    return model_type and model_type.lower() in HUGGINGFACE_CAPITALIZE


def capitalize_model_type(model_type: str) ->str:
    """
    Returns the proper capitalized version of the model type, that can be used to
    retrieve the model class from transformers.
    :param model_type: the model_type as found in the config file
    :return: the capitalized version of the model type, or the original name of not found.
    """
    return HUGGINGFACE_CAPITALIZE.get(model_type.lower(), model_type)


LANGUAGE_HINTS = ('german', 'german'), ('english', 'english'), ('chinese', 'chinese'), ('indian', 'indian'), ('french', 'french'), ('camembert', 'french'), ('polish', 'polish'), ('spanish', 'spanish'), ('umberto', 'italian'), ('multilingual', 'multilingual')


def _guess_language(name: str) ->str:
    """
    Looks for clues about the model language in the model name.
    """
    languages = [lang for hint, lang in LANGUAGE_HINTS if hint.lower() in name.lower()]
    if len(languages) > 0:
        language = languages[0]
    else:
        language = 'english'
    logger.info('Auto-detected model language: %s', language)
    return language


def silence_transformers_logs(from_pretrained_func):
    """
    A wrapper that raises the log level of Transformers to
    ERROR to hide some unnecessary warnings.
    """

    @wraps(from_pretrained_func)
    def quiet_from_pretrained_func(cls, *args, **kwargs):
        t_logger = logging.getLogger('transformers')
        original_log_level = t_logger.level
        t_logger.setLevel(logging.ERROR)
        result = from_pretrained_func(cls, *args, **kwargs)
        t_logger.setLevel(original_log_level)
        return result
    return quiet_from_pretrained_func


def loss_per_head_sum(loss_per_head: List[torch.Tensor], global_step: Optional[int]=None, batch: Optional[Dict]=None):
    """
    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
    Output: aggregated loss (tensor)
    """
    return sum(loss_per_head)


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


class WrappedDataParallel(DataParallel):
    """
    A way of adapting attributes of underlying class to parallel mode. See:
    https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html#dataparallel

    Gets into recursion errors. Workaround see:
    https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
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

class Test_deepset_ai_haystack(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

