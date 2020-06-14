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
visual = _module
ascii = _module
images = _module
text = _module
run_all_experiments = _module
setup = _module
conftest = _module
convert_result_to_csv = _module
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


import logging


from functools import partial


import torch


from torch.utils.data.sampler import SequentialSampler


import copy


import numpy


from torch import nn


from collections import OrderedDict


import numpy as np


import inspect


from torch.nn.parallel import DistributedDataParallel


from torch.nn import DataParallel


import itertools


from scipy.special import expit


from scipy.special import softmax


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


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
        raise Exception(
            f'More than one of the prediction heads have a {fn_name}() function'
            )


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
            preds_final = self.language_model.formatted_preds(logits=logits,
                **kwargs)
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
            for i, (head, preds_p_for_head, logits_for_head) in enumerate(zip
                (self.prediction_heads, preds_for_heads, logits_for_heads)):
                preds = head.formatted_preds(logits=logits_for_head,
                    preds_p=preds_p_for_head, **kwargs)
                preds_final[i].append(preds)
            merge_fn = pick_single_fn(self.prediction_heads,
                'merge_formatted_preds')
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
                logger.info(
                    'Removing the NextSentenceHead since next_sent_pred is set to False in the BertStyleLMProcessor'
                    )
                del self.prediction_heads[i]
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]['label_tensor_name']
            label_list = tasks[head.task_name]['label_list']
            if not label_list and require_labels:
                raise Exception(
                    f"The task '{head.task_name}' is missing a valid set of labels"
                    )
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
        model_files = [(load_dir / f) for f in files if '.bin' in f and 
            'prediction_head' in f]
        config_files = [(load_dir / f) for f in files if 'config.json' in f and
            'prediction_head' in f]
        model_files.sort()
        config_files.sort()
        if strict:
            error_str = (
                f'There is a mismatch in number of model files ({len(model_files)}) and config files ({len(config_files)}).This might be because the Language Model Prediction Head does not currently support saving and loading'
                )
            assert len(model_files) == len(config_files), error_str
        logger.info(
            f'Found files for loading {len(model_files)} prediction heads')
        return model_files, config_files


EMBEDDING_VOCAB_FILES_MAP = {}


def load_from_cache(pretrained_model_name_or_path, s3_dict, **kwargs):
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    s3_file = s3_dict[pretrained_model_name_or_path]
    try:
        resolved_file = cached_path(s3_file, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies, resume_download
            =resume_download)
        if resolved_file is None:
            raise EnvironmentError
    except EnvironmentError:
        if pretrained_model_name_or_path in s3_dict:
            msg = "Couldn't reach server at '{}' to download data.".format(
                s3_file)
        else:
            msg = (
                "Model name '{}' was not found in model name list. We assumed '{}' was a path, a model identifier, or url to a configuration file or a directory containing such a file but couldn't find any such file at this path or url."
                .format(pretrained_model_name_or_path, s3_file))
        raise EnvironmentError(msg)
    if resolved_file == s3_file:
        logger.info('loading file {}'.format(s3_file))
    else:
        logger.info('loading file {} from cache at {}'.format(s3_file,
            resolved_file))
    return resolved_file


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if (cp >= 33 and cp <= 47 or cp >= 58 and cp <= 64 or cp >= 91 and cp <=
        96 or cp >= 123 and cp <= 126):
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
        return cls.subclasses[prediction_head_name](layer_dims=layer_dims,
            class_weights=class_weights)

    def save_config(self, save_dir, head_num=0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :type save_dir: str or Path
        :param head_num: Which head to save
        :type head_num: int
        """
        self.generate_config()
        output_config_file = Path(save_dir
            ) / f'prediction_head_{head_num}_config.json'
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
            prediction_head.load_state_dict(torch.load(model_file,
                map_location=torch.device('cpu')), strict=strict)
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
            logger.info(
                f'Resizing input dimensions of {type(self).__name__} ({self.task_name}) from {old_dims} to {new_dims} to match language model'
                )
            self.feed_forward = FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file):
        if 'config.json' in str(config_file) and 'prediction_head' in str(
            config_file):
            head_num = int(''.join([char for char in os.path.basename(
                config_file) if char.isdigit()]))
            model_file = Path(os.path.dirname(config_file)
                ) / f'prediction_head_{head_num}.bin'
        else:
            raise ValueError(
                f"This doesn't seem to be a proper prediction_head config file: '{config_file}'"
                )
        return model_file

    def _set_name(self, name):
        self.task_name = name


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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_deepset_ai_FARM(_paritybench_base):
    pass
    def test_000(self):
        self._check(FeedForwardBlock(*[], **{'layer_dims': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})

