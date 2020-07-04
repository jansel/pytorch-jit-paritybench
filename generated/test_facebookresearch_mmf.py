import sys
_module = sys.modules[__name__]
del sys
conf = _module
mmf = _module
common = _module
batch_collator = _module
constants = _module
dataset_loader = _module
meter = _module
registry = _module
report = _module
sample = _module
test_reporter = _module
typings = _module
datasets = _module
base_dataset = _module
base_dataset_builder = _module
builders = _module
clevr = _module
builder = _module
dataset = _module
coco = _module
masked_builder = _module
masked_dataset = _module
conceptual_captions = _module
hateful_memes = _module
dataset = _module
mmimdb = _module
nlvr2 = _module
ocrvqa = _module
sbu_captions = _module
stvqa = _module
textcaps = _module
textvqa = _module
visual_dialog = _module
database = _module
original = _module
build_imdb = _module
extract_vocabulary = _module
visual_entailment = _module
visual_genome = _module
vizwiz = _module
vqa2 = _module
masked_q_vqa2_builder = _module
masked_q_vqa2_dataset = _module
ocr_builder = _module
ocr_dataset = _module
concat_dataset = _module
databases = _module
annotation_database = _module
features_database = _module
image_database = _module
readers = _module
feature_readers = _module
scene_graph_database = _module
mmf_dataset = _module
mmf_dataset_builder = _module
multi_dataset_loader = _module
processors = _module
bert_processors = _module
image_processors = _module
models = _module
ban = _module
base_model = _module
butd = _module
cnn_lstm = _module
fusions = _module
lorra = _module
m4c = _module
m4c_captioner = _module
mmbt = _module
mmf_bert = _module
pythia = _module
top_down_bottom_up = _module
unimodal = _module
vilbert = _module
visdial_multi_modal = _module
visual_bert = _module
modules = _module
attention = _module
decoders = _module
embeddings = _module
encoders = _module
fusions = _module
layers = _module
losses = _module
metrics = _module
optimizers = _module
schedulers = _module
trainers = _module
base_trainer = _module
utils = _module
build = _module
checkpoint = _module
configuration = _module
distributed = _module
download = _module
early_stopping = _module
env = _module
file_io = _module
flags = _module
general = _module
logger = _module
m4c_evaluators = _module
modeling = _module
phoc = _module
build_phoc = _module
process_answers = _module
text = _module
timer = _module
transform = _module
visualize = _module
vocab = _module
version = _module
mmf_cli = _module
hm_convert = _module
predict = _module
run = _module
extract_ocr_frcn_feature = _module
coco_eval = _module
textcaps_eval = _module
setup = _module
tests = _module
test_sample = _module
test_configs_for_keys = _module
test_zoo_urls = _module
test_base_dataset = _module
test_processors = _module
test_cnn_lstm = _module
test_fusions = _module
test_layers = _module
test_losses = _module
test_metrics = _module
test_utils = _module
test_checkpoint = _module
test_configuration = _module
test_distributed = _module
test_download = _module
test_file_io = _module
test_general = _module
test_model = _module
test_text = _module
test_timer = _module
tools = _module
scripts = _module
extract_bert_embeddings = _module
coco_caption_eval = _module
extract_features_vmb = _module
extract_resnet152_feat = _module
lmdb_conversion = _module
convert_gqa_to_vqa = _module
generate_test_data = _module
lib = _module
slurm = _module
sweep_visual_bert = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import copy


import numpy as np


import torch


from torchvision import transforms


from torch import nn


import collections


import warnings


from copy import deepcopy


import functools


import math


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


from torch.nn.utils.weight_norm import weight_norm


from functools import lru_cache


import torchvision


import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence


import time


import re


from collections import Counter


from itertools import chain


from collections import defaultdict


from torchtext import vocab


import torchvision.models as models


import torchvision.transforms as transforms


from torch.autograd import Variable


def built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.

    If a version_string is provided, this has to match, or the version
    is regarded as not built.

    Version_string are generally the dataset version + the date the file was
    last updated. If this doesn't match, dataset will be mark not built. This makes
    sure that if we update our features or anything else features are updated
    for the end user.
    """
    if version_string:
        fname = os.path.join(path, '.built.json')
        if not PathManager.isfile(fname):
            return False
        else:
            with PathManager.open(fname, 'r') as read:
                text = json.load(read)
            return text.get('version', None) == version_string
    else:
        return PathManager.isfile(os.path.join(path, '.built.json'))


def decompress(path, fname, delete_original=True):
    """
    Unpack the given archive file to the same directory.

    Args:
        path(str): The folder containing the archive. Will contain the contents.
        fname (str): The filename of the archive file.
        delete_original (bool, optional): If true, the archive will be deleted
                                          after extraction. Default to True.
    """
    print('Unpacking ' + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if delete_original:
        os.remove(fullpath)


def check_header(url, from_google=False):
    """
    Performs a HEAD request to check if the URL / Google Drive ID is live.
    """
    session = requests.Session()
    if from_google:
        URL = 'https://docs.google.com/uc?export=download'
        response = session.head(URL, params={'id': url}, stream=True)
    else:
        headers = {'User-Agent': 
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) ' +
            'AppleWebKit/537.36 (KHTML, like Gecko) ' +
            'Chrome/77.0.3865.90 Safari/537.36'}
        response = session.head(url, allow_redirects=True, headers=headers)
    status = response.status_code
    session.close()
    assert status == 200, ('The url {} is broken. If this is not your own url,'
         + ' please open up an issue on GitHub').format(url)


def move(path1, path2):
    """
    Rename the given file.
    """
    shutil.move(path1, path2)


def download(url, path, fname, redownload=True):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``True``).

    Returns whether download actually happened or not
    """
    outfile = os.path.join(path, fname)
    download = not PathManager.isfile(outfile) or redownload
    retry = 5
    exp_backoff = [(2 ** r) for r in reversed(range(retry))]
    pbar = None
    if download:
        check_header(url)
        print('[ Downloading: ' + url + ' to ' + outfile + ' ]')
        pbar = tqdm.tqdm(unit='B', unit_scale=True, desc=f'Downloading {fname}'
            )
    while download and retry >= 0:
        resume_file = outfile + '.part'
        resume = PathManager.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = 'ab'
        else:
            resume_pos = 0
            mode = 'wb'
        response = None
        with requests.Session() as session:
            try:
                header = {'Range': 'bytes=%d-' % resume_pos,
                    'Accept-Encoding': 'identity'} if resume else {}
                response = session.get(url, stream=True, timeout=5, headers
                    =header)
                if resume and response.headers.get('Accept-Ranges', 'none'
                    ) == 'none':
                    resume_pos = 0
                    mode = 'wb'
                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                total_size += resume_pos
                pbar.total = total_size
                done = resume_pos
                with PathManager.open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except (requests.exceptions.ConnectionError, requests.
                exceptions.ReadTimeout):
                retry -= 1
                pbar.clear()
                if retry >= 0:
                    print('Connection error, retrying. (%d retries left)' %
                        retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning(
            'Connection broken too many times. Stopped retrying.')
    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning('Received less data than specified in ' +
                'Content-Length header for ' + url +
                '. There may be a download problem.')
        move(resume_file, outfile)
    if pbar:
        pbar.close()
    return download


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_from_google_drive(gd_id, destination, redownload=True):
    """
    Use the requests package to download a file from Google Drive.
    """
    download = not PathManager.isfile(destination) or redownload
    URL = 'https://docs.google.com/uc?export=download'
    if not download:
        return download
    else:
        check_header(gd_id, from_google=True)
    with requests.Session() as session:
        response = session.get(URL, params={'id': gd_id}, stream=True)
        token = _get_confirm_token(response)
        if token:
            response.close()
            params = {'id': gd_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        CHUNK_SIZE = 32768
        with PathManager.open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        response.close()
    return download


class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.

    Originally taken from ParlAI, this file has been modified for MMF specific
    use cases.

    Any dataset/model that needs to download a file needs to have a list RESOURCES
    that have objects of this class as elements.

    The class automatically figures out if the file is from Google Drive.

    This class provides the following functionality:

    - Download a file from a URL / Google Drive
    - Decompress the file if compressed
    - Checksum for the downloaded file
    - Send HEAD request to validate URL or Google Drive link
    - If the file is present and checksum is same, it won't be redownloaded

    Raises:
        AssertionError: If while downloading checksum of the files fails.
    """
    GOOGLE_DRIVE_SUBSTR = 'drive.google'
    MMF_PREFIX = 'mmf://'
    MMF_PREFIX_REPLACEMENT = 'https://dl.fbaipublicfiles.com/mmf/data/'

    def __init__(self, url, file_name, hashcode=None, compressed=True,
        delete_original=False):
        """
        An object of this class needs to be created with:

        Args:
            url (string): URL or Google Drive id to download from
            file_name (string): File name that the file should be named
            hashcode (string, optional): SHA256 hashcode of the downloaded file.
                                         Defaults to None. Won't be checked if not
                                         passed.
            compressed (bool, optional): False if the file is not compressed.
                                         Defaults to True.
            delete_original (bool, optional): If compressed whether to delete original.
                                              Defaults to False.
        """
        self._url = self._parse_url(url)
        self._file_name = file_name
        self._hashcode = hashcode
        self._compressed = compressed
        self._from_google = self._url.find(self.GOOGLE_DRIVE_SUBSTR) != -1
        self._delete_original = delete_original

    def _parse_url(self, url):
        if url.find(self.MMF_PREFIX) == -1:
            return url
        else:
            return self.MMF_PREFIX_REPLACEMENT + url[len(self.MMF_PREFIX):]

    def checksum(self, download_path):
        """
        Checksum on a given file.

        Args:
            download_path (string): path to the downloaded file.
        """
        if self._hashcode is None:
            print(f'[ Checksum not provided, skipping for {self._file_name}]')
            return
        sha256_hash = hashlib.sha256()
        destination = os.path.join(download_path, self._file_name)
        if not PathManager.isfile(destination):
            return
        with PathManager.open(destination, 'rb') as f:
            print(f'[ Starting checksum for {self._file_name}]')
            for byte_block in iter(lambda : f.read(65536), b''):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self._hashcode:
                raise AssertionError(
                    f"""[ Checksum for {self._file_name} from 
{self._url}
does not match the expected checksum. Please try again. ]"""
                    )
            else:
                print(f'[ Checksum successful for {self._file_name}]')

    def download_file(self, download_path):
        downloaded = False
        redownload = False
        try:
            self.checksum(download_path)
        except AssertionError:
            print('[ Checksum changed for {}. Redownloading')
            redownload = True
        if self._from_google:
            downloaded = download_from_google_drive(self._url, os.path.join
                (download_path, self._file_name), redownload=redownload)
        else:
            downloaded = download(self._url, download_path, self._file_name,
                redownload=redownload)
        if downloaded:
            self.checksum(download_path)
            if self._compressed:
                decompress(download_path, self._file_name, self.
                    _delete_original)


def download_resource(resource, download_path):
    if isinstance(resource, collections.abc.Mapping):
        resource = DownloadableFile(**resource)
    assert isinstance(resource, DownloadableFile)
    resource.download_file(download_path)


def make_dir(path):
    """
    Make the directory and any nonexistent parent directories (`mkdir -p`).
    """
    if path != '':
        PathManager.mkdirs(path)


def mark_done(path, version_string=None):
    """
    Mark this path as prebuilt.

    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.

    Args:
        path (str): The file path to mark as built
        version_string (str): The version of this dataset
    """
    data = {}
    data['created_at'] = str(datetime.datetime.today())
    data['version'] = version_string
    with PathManager.open(os.path.join(path, '.built.json'), 'w') as f:
        json.dump(data, f)


def download_resources(resources, download_path, version):
    is_built = built(download_path, version_string=version)
    if not is_built:
        make_dir(download_path)
        if not isinstance(resources, collections.abc.Sequence):
            resources = [resources]
        if len(resources) == 0:
            return
        for resource in resources:
            download_resource(resource, download_path)
        mark_done(download_path, version_string=version)


def get_mmf_root():
    from mmf.common.registry import registry
    mmf_root = registry.get('mmf_root', no_warning=True)
    if mmf_root is None:
        mmf_root = os.path.dirname(os.path.abspath(__file__))
        mmf_root = os.path.abspath(os.path.join(mmf_root, '..'))
        registry.register('mmf_root', mmf_root)
    return mmf_root


def get_absolute_path(paths):
    if isinstance(paths, str):
        if os.path.isabs(paths):
            return paths
        possible_paths = [paths]
        from mmf.utils.configuration import get_mmf_env
        user_dir = get_mmf_env(key='user_dir')
        if user_dir:
            possible_paths.append(os.path.join(user_dir, paths))
        mmf_root = get_mmf_root()
        possible_paths.append(os.path.join(mmf_root, '..', paths))
        possible_paths.append(os.path.join(mmf_root, paths))
        for path in possible_paths:
            if PathManager.exists(path):
                if path.find('://') == -1:
                    return os.path.abspath(path)
                else:
                    return path
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError(
            'Paths passed to dataset should either be string or list')


def get_rank():
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


def synchronize():
    if not dist.is_nccl_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def download_pretrained_model(model_name, *args, **kwargs):
    import omegaconf
    from omegaconf import OmegaConf
    from mmf.utils.configuration import load_yaml, get_mmf_env
    model_zoo = load_yaml(get_mmf_env(key='model_zoo'))
    OmegaConf.set_struct(model_zoo, True)
    OmegaConf.set_readonly(model_zoo, True)
    data_dir = get_absolute_path(get_mmf_env('data_dir'))
    model_data_dir = os.path.join(data_dir, 'models')
    download_path = os.path.join(model_data_dir, model_name)
    try:
        model_config = OmegaConf.select(model_zoo, model_name)
    except omegaconf.errors.OmegaConfBaseException as e:
        print(f'No such model name {model_name} defined in mmf zoo')
        raise e
    if 'version' not in model_config or 'resources' not in model_config:
        try:
            model_config = model_config.defaults
            download_path = os.path.join(model_data_dir, model_name +
                '.defaults')
        except omegaconf.errors.OmegaConfBaseException as e:
            print(
                f"Model name {model_name} doesn't specify 'resources' and 'version' while no defaults have been provided"
                )
            raise e
    if 'zoo_requirements' in model_config:
        requirements = model_config.zoo_requirements
        if isinstance(requirements, str):
            requirements = [requirements]
        for item in requirements:
            download_pretrained_model(item, *args, **kwargs)
    version = model_config.version
    resources = model_config.resources
    if is_master():
        download_resources(resources, download_path, version)
    synchronize()
    return download_path


def _hack_imports():
    sys.modules['pythia'] = importlib.import_module('mmf')
    sys.modules['pythia.utils.configuration'] = importlib.import_module(
        'mmf.utils.configuration')


def load_yaml(f):
    abs_f = get_absolute_path(f)
    try:
        mapping = OmegaConf.load(abs_f)
        f = abs_f
    except FileNotFoundError as e:
        relative = os.path.abspath(os.path.join(get_mmf_root(), f))
        if not PathManager.isfile(relative):
            raise e
        else:
            f = relative
            mapping = OmegaConf.load(f)
    if mapping is None:
        mapping = OmegaConf.create()
    includes = mapping.get('includes', [])
    if not isinstance(includes, collections.abc.Sequence):
        raise AttributeError('Includes must be a list, {} provided'.format(
            type(includes)))
    include_mapping = OmegaConf.create()
    mmf_root_dir = get_mmf_root()
    for include in includes:
        original_include_path = include
        include = os.path.join(mmf_root_dir, include)
        if not PathManager.exists(include):
            include = os.path.join(os.path.dirname(f), original_include_path)
        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping,
            current_include_mapping)
    mapping.pop('includes', None)
    mapping = OmegaConf.merge(include_mapping, mapping)
    return mapping


def load_pretrained_model(model_name_or_path, *args, **kwargs):
    if PathManager.exists(model_name_or_path):
        download_path = model_name_or_path
        model_name = model_name_or_path
    else:
        download_path = download_pretrained_model(model_name_or_path, *args,
            **kwargs)
        model_name = model_name_or_path
    configs = glob.glob(os.path.join(download_path, '*.yaml'))
    assert len(configs
        ) <= 1, 'Multiple yaml files with the pretrained model. ' + "MMF doesn't know what to do."
    ckpts = []
    allowed_ckpt_types = '*.ckpt', '*.pth', '*.pt'
    for ckpt_type in allowed_ckpt_types:
        ckpts.extend(glob.glob(os.path.join(download_path, ckpt_type)))
    assert len(ckpts
        ) == 1, "None or multiple checkpoints files. MMF doesn't know what to do."
    _hack_imports()
    ckpt = torch.load(ckpts[0], map_location=lambda storage, loc: storage)
    if len(configs) == 0:
        assert 'config' in ckpt, "No configs provided with pretrained model  while checkpoint also doesn't have configuration."
        config = ckpt['config']
    else:
        config = load_yaml(configs[0])
    model_config = config.get('model_config', config)
    ckpt = ckpt.get('model', ckpt)
    model_config = model_config.get(model_name.split(os.path.sep)[-1].split
        ('.')[0])
    return {'config': model_config, 'checkpoint': ckpt, 'full_config': config}


class Registry:
    """Class for registry object which acts as central source of truth
    for MMF
    """
    mapping = {'builder_name_mapping': {}, 'trainer_name_mapping': {},
        'model_name_mapping': {}, 'metric_name_mapping': {},
        'loss_name_mapping': {}, 'fusion_name_mapping': {},
        'optimizer_name_mapping': {}, 'scheduler_name_mapping': {},
        'processor_name_mapping': {}, 'decoder_name_mapping': {}, 'state': {}}

    @classmethod
    def register_trainer(cls, name):
        """Register a trainer to registry with key 'name'

        Args:
            name: Key with which the trainer will be registered.

        Usage::

            from mmf.common.registry import registry
            from mmf.trainers.custom_trainer import CustomTrainer


            @registry.register_trainer("custom_trainer")
            class CustomTrainer():
                ...

        """

        def wrap(trainer_cls):
            cls.mapping['trainer_name_mapping'][name] = trainer_cls
            return trainer_cls
        return wrap

    @classmethod
    def register_builder(cls, name):
        """Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage::

            from mmf.common.registry import registry
            from mmf.datasets.base_dataset_builder import BaseDatasetBuilder


            @registry.register_builder("vqa2")
            class VQA2Builder(BaseDatasetBuilder):
                ...

        """

        def wrap(builder_cls):
            from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
            assert issubclass(builder_cls, BaseDatasetBuilder
                ), 'All builders must inherit BaseDatasetBuilder class'
            cls.mapping['builder_name_mapping'][name] = builder_cls
            return builder_cls
        return wrap

    @classmethod
    def register_metric(cls, name):
        """Register a metric to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage::

            from mmf.common.registry import registry
            from mmf.modules.metrics import BaseMetric


            @registry.register_metric("r@1")
            class RecallAt1(BaseMetric):
                ...

        """

        def wrap(func):
            from mmf.modules.metrics import BaseMetric
            assert issubclass(func, BaseMetric
                ), 'All Metric must inherit BaseMetric class'
            cls.mapping['metric_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_loss(cls, name):
        """Register a loss to registry with key 'name'

        Args:
            name: Key with which the loss will be registered.

        Usage::

            from mmf.common.registry import registry
            from torch import nn

            @registry.register_task("logit_bce")
            class LogitBCE(nn.Module):
                ...

        """

        def wrap(func):
            from torch import nn
            assert issubclass(func, nn.Module
                ), 'All loss must inherit torch.nn.Module class'
            cls.mapping['loss_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_fusion(cls, name):
        """Register a fusion technique to registry with key 'name'

        Args:
            name: Key with which the fusion technique will be registered

        Usage::

            from mmf.common.registry import registry
            from torch import nn

            @registry.register_fusion("linear_sum")
            class LinearSum():
                ...
        """

        def wrap(func):
            from torch import nn
            assert issubclass(func, nn.Module
                ), 'All Fusion must inherit torch.nn.Module class'
            cls.mapping['fusion_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_model(cls, name):
        """Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage::

            from mmf.common.registry import registry
            from mmf.models.base_model import BaseModel

            @registry.register_task("pythia")
            class Pythia(BaseModel):
                ...
        """

        def wrap(func):
            from mmf.models.base_model import BaseModel
            assert issubclass(func, BaseModel
                ), 'All models must inherit BaseModel class'
            cls.mapping['model_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_processor(cls, name):
        """Register a processor to registry with key 'name'

        Args:
            name: Key with which the processor will be registered.

        Usage::

            from mmf.common.registry import registry
            from mmf.datasets.processors import BaseProcessor

            @registry.register_task("glove")
            class GloVe(BaseProcessor):
                ...

        """

        def wrap(func):
            from mmf.datasets.processors.processors import BaseProcessor
            assert issubclass(func, BaseProcessor
                ), 'All Processor classes must inherit BaseProcessor class'
            cls.mapping['processor_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_optimizer(cls, name):

        def wrap(func):
            cls.mapping['optimizer_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_scheduler(cls, name):

        def wrap(func):
            cls.mapping['scheduler_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_decoder(cls, name):
        """Register a decoder to registry with key 'name'

        Args:
            name: Key with which the decoder will be registered.

        Usage::

            from mmf.common.registry import registry
            from mmf.utils.text import TextDecoder


            @registry.register_decoder("nucleus_sampling")
            class NucleusSampling(TextDecoder):
                ...

        """

        def wrap(decoder_cls):
            from mmf.utils.text import TextDecoder
            assert issubclass(decoder_cls, TextDecoder
                ), 'All decoders must inherit TextDecoder class'
            cls.mapping['decoder_name_mapping'][name] = decoder_cls
            return decoder_cls
        return wrap

    @classmethod
    def register(cls, name, obj):
        """Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from mmf.common.registry import registry

            registry.register("config", {})
        """
        path = name.split('.')
        current = cls.mapping['state']
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path[-1]] = obj

    @classmethod
    def get_trainer_class(cls, name):
        return cls.mapping['trainer_name_mapping'].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping['builder_name_mapping'].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping['model_name_mapping'].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping['processor_name_mapping'].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping['metric_name_mapping'].get(name, None)

    @classmethod
    def get_loss_class(cls, name):
        return cls.mapping['loss_name_mapping'].get(name, None)

    @classmethod
    def get_optimizer_class(cls, name):
        return cls.mapping['optimizer_name_mapping'].get(name, None)

    @classmethod
    def get_scheduler_class(cls, name):
        return cls.mapping['scheduler_name_mapping'].get(name, None)

    @classmethod
    def get_decoder_class(cls, name):
        return cls.mapping['decoder_name_mapping'].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        """Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        Usage::

            from mmf.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split('.')
        value = cls.mapping['state']
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break
        if 'writer' in cls.mapping['state'
            ] and value == default and no_warning is False:
            cls.mapping['state']['writer'].write(
                'Key {} is not present in registry, returning default value of {}'
                .format(original_name, default))
        return value

    @classmethod
    def unregister(cls, name):
        """Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping['state'].pop(name, None)


registry = Registry()


class BaseModel(nn.Module):
    """For integration with Pythia's trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._logged_warning = {'losses_present': False}
        self.writer = registry.get('writer')
        self._is_pretrained = False

    @property
    def is_pretrained(self):
        return self._is_pretrained

    @is_pretrained.setter
    def is_pretrained(self, x):
        self._is_pretrained = x

    def build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError(
            'Build method not implemented in the child model class.')

    def init_losses(self):
        """Initializes loss for the model based ``losses`` key. Automatically called by
        MMF internally after building the model.
        """
        losses = self.config.get('losses', [])
        if len(losses) == 0 and not self.is_pretrained:
            warnings.warn(
                'No losses are defined in model configuration. You are expected to return loss in your return dict from forward.'
                )
        self.losses = Losses(losses)

    @classmethod
    def config_path(cls):
        return None

    @classmethod
    def format_state_key(cls, key):
        """Can be implemented if something special needs to be done
        key when pretrained model is being load. This will adapt and return
        keys according to that. Useful for backwards compatibility. See
        updated load_state_dict below. For an example, see VisualBERT model's
        code.

        Args:
            key (string): key to be formatted

        Returns:
            string: formatted key
        """
        return key

    def load_state_dict(self, state_dict, *args, **kwargs):
        copied_state_dict = deepcopy(state_dict)
        for key in list(copied_state_dict.keys()):
            formatted_key = self.format_state_key(key)
            copied_state_dict[formatted_key] = copied_state_dict.pop(key)
        return super().load_state_dict(copied_state_dict, *args, **kwargs)

    def forward(self, sample_list, *args, **kwargs):
        """To be implemented by child class. Takes in a ``SampleList`` and
        returns back a dict.

        Args:
            sample_list (SampleList): SampleList returned by the DataLoader for
            current iteration

        Returns:
            Dict: Dict containing scores object.

        """
        raise NotImplementedError(
            'Forward of the child model class needs to be implemented.')

    def __call__(self, sample_list, *args, **kwargs):
        model_output = super().__call__(sample_list, *args, **kwargs)
        if self.is_pretrained:
            return model_output
        assert isinstance(model_output, collections.abc.Mapping
            ), 'A dict must be returned from the forward of the model.'
        if 'losses' in model_output:
            if not self._logged_warning['losses_present']:
                warnings.warn(
                    "'losses' already present in model output. No calculation will be done in base model."
                    )
                self._logged_warning['losses_present'] = True
            assert isinstance(model_output['losses'], collections.abc.Mapping
                ), "'losses' must be a dict."
        else:
            model_output['losses'] = self.losses(sample_list, model_output)
        return model_output

    def load_requirements(self, *args, **kwargs):
        requirements = self.config.get('zoo_requirements', [])
        if isinstance(requirements, str):
            requirements = [requirements]
        for item in requirements:
            download_pretrained_model(item, *args, **kwargs)

    def format_for_prediction(self, results, report):
        """Implement this method in models if it requires to modify prediction
        results using report fields. Note that the required fields in report
        should already be gathered in report.
        """
        return results

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        model_key = model_name.split('.')[0]
        model_cls = registry.get_model_class(model_key)
        assert model_cls == cls, f'Incorrect pretrained model key {model_name} for class {cls.__name__}'
        output = load_pretrained_model(model_name, *args, **kwargs)
        config, checkpoint = output['config'], output['checkpoint']
        if hasattr(cls, 'update_registry_for_pretrained'):
            cls.update_registry_for_pretrained(config, checkpoint, output)
        instance = cls(config)
        instance.is_pretrained = True
        instance.build()
        instance.load_state_dict(checkpoint)
        instance.eval()
        return instance


class OcrPtrNet(nn.Module):

    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()
        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size
        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)
        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)
        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)
        return scores


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)
    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


class PrevPredEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps
        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)
        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2
        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=
            ocr_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings
        return dec_emb


class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding.
    """

    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.
            hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_modal, start_token=None, end_token=None,
        position_ids=None, token_type_ids=None):
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)
        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat([start_token_embeds.unsqueeze(1),
                token_embeddings], dim=1)
        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat([token_embeddings,
                end_token_embeds.unsqueeze(1)], dim=1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long,
                device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(input_modal.
                size(0), seq_length)
        if token_type_ids is None:
            token_type_ids = torch.zeros((input_modal.size(0), seq_length),
                dtype=torch.long, device=input_modal.device)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (token_embeddings + position_embeddings +
            token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MMBTModel(nn.Module):
    """
        Outputs: `Tuple` comprising various elements depending on the configuration
            (config) and inputs:
            **last_hidden_state**: ``torch.FloatTensor`` of shape
                ``(batch_size, sequence_length, hidden_size)``. Sequence of
                hidden-states at the output of the last layer of the model.
            **pooler_output**: ``torch.FloatTensor`` of shape
                ``(batch_size, hidden_size)``. Last layer hidden-state of the
                first token of the sequence (classification token) further processed
                by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction
                (classification) objective during Bert pretraining. This output
                is usually *not* a good summary of the semantic content of the
                input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            **hidden_states**: (`optional`, returned when
                ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer +
                the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the
                initial embedding outputs.
            **attentions**: (`optional`, returned when
                ``config.output_attentions=True``) list of ``torch.FloatTensor``
                (one for each layer) of shape ``(batch_size, num_heads,
                sequence_length, sequence_length)``: Attentions weights after
                the attention softmax, used to compute the weighted average in the
                self-attention heads.
        Examples::
            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            mmbt = MMBTModel(config, transformer, encoder)
        """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.modal_encoder = ModalEmbeddings(config, encoder, transformer.
            embeddings)

    def forward(self, input_modal, input_ids=None, modal_start_tokens=None,
        modal_end_tokens=None, attention_mask=None, token_type_ids=None,
        modal_token_type_ids=None, position_ids=None, modal_position_ids=
        None, head_mask=None, inputs_embeds=None, encoder_hidden_states=
        None, encoder_attention_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
                )
        elif input_ids is not None:
            input_txt_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_txt_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')
        device = (input_ids.device if input_ids is not None else
            inputs_embeds.device)
        modal_embeddings = self.modal_encoder(input_modal, start_token=
            modal_start_tokens, end_token=modal_end_tokens, position_ids=
            modal_position_ids, token_type_ids=modal_token_type_ids)
        input_modal_shape = modal_embeddings.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.ones(input_txt_shape, dtype=torch.long,
                device=device)
        txt_embeddings = self.transformer.embeddings(input_ids=input_ids,
            position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds)
        embedding_output = torch.cat([modal_embeddings, txt_embeddings], 1)
        input_shape = embedding_output.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            attention_mask = torch.cat([torch.ones(input_modal_shape,
                device=device, dtype=torch.long), attention_mask], dim=1)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        else:
            encoder_attention_mask = torch.cat([torch.ones(
                input_modal_shape, device=device), encoder_attention_mask],
                dim=1)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, (None), :, :]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[(None), (None), :].repeat(batch_size,
                    seq_length, 1) <= seq_ids[(None), :, (None)]
                extended_attention_mask = causal_mask[:, (None), :, :
                    ] * attention_mask[:, (None), (None), :]
            else:
                extended_attention_mask = attention_mask[:, (None), (None), :]
        extended_attention_mask = extended_attention_mask
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, (
                None), :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, (
                None), (None), :]
        encoder_extended_attention_mask = encoder_extended_attention_mask
        encoder_extended_attention_mask = (1.0 -
            encoder_extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1
                    ).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers,
                    -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.transformer.encoder(embedding_output,
            attention_mask=extended_attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.transformer.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class MMBTConfig:
    """Configuration class to store the configuration of a `MMBT Model`.
    Args:
        config (:obj:`~transformers.PreTrainedConfig`):
            Config of the underlying Transformer models. Its values are
            copied over to use a single config.
        num_labels (:obj:`int` or :obj:`None`, optional, defaults to `None`):
            Size of final Linear layer for classification.
        modal_hidden_size (:obj:`int`, optional, defautls to 2048):
            Embedding dimension of the non-text modality encoder.
    """

    def __init__(self, config, num_labels=None, modal_hidden_size=2048):
        self.__dict__ = config.__dict__
        self.modal_hidden_size = modal_hidden_size
        if num_labels:
            self.num_labels = num_labels


def get_default_config_path():
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, '..', 'configs', 'defaults.yaml')


def import_user_module(user_dir: str, no_print: bool=False):
    """Given a user dir, this function imports it as a module.

    This user_module is expected to have an __init__.py at its root.
    You can use import_files to import your python files easily in
    __init__.py

    Args:
        user_dir (str): directory which has to be imported
        no_print (bool): This function won't print anything if set to true
    """
    if user_dir:
        user_dir = get_absolute_path(user_dir)
        module_parent, module_name = os.path.split(user_dir)
        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            if not no_print:
                print(f'Importing user_dir from {user_dir}')
            importlib.import_module(module_name)
            sys.path.pop(0)


def resolve_cache_dir(env_variable='MMF_CACHE_DIR', default='mmf'):
    try:
        from torch.hub import _get_torch_home
        torch_cache_home = _get_torch_home()
    except ImportError:
        torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', os.
            path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
    default_cache_path = os.path.join(torch_cache_home, default)
    cache_path = os.getenv(env_variable, default_cache_path)
    if not PathManager.exists(cache_path):
        try:
            PathManager.mkdirs(cache_path)
        except PermissionError:
            cache_path = os.path.join(get_mmf_root(), '.mmf_cache')
            PathManager.mkdirs(cache_path)
    return cache_path


def resolve_dir(env_variable, default='data'):
    default_dir = os.path.join(resolve_cache_dir(), default)
    dir_path = os.getenv(env_variable, default_dir)
    if not PathManager.exists(dir_path):
        PathManager.mkdirs(dir_path)
    return dir_path


class Configuration:

    def __init__(self, args=None, default_only=False):
        self.config = {}
        if not args:
            import argparse
            args = argparse.Namespace(opts=[])
            default_only = True
        self.args = args
        self._register_resolvers()
        self._default_config = self._build_default_config()
        if default_only:
            other_configs = {}
        else:
            other_configs = self._build_other_configs()
        self.config = OmegaConf.merge(self._default_config, other_configs)
        self.config = self._merge_with_dotlist(self.config, args.opts)
        self._update_specific(self.config)
        self.upgrade(self.config)
        registry.register('config', self.config)

    def _build_default_config(self):
        self.default_config_path = get_default_config_path()
        default_config = load_yaml(self.default_config_path)
        return default_config

    def _build_other_configs(self):
        opts_config = self._build_opt_list(self.args.opts)
        user_config = self._build_user_config(opts_config)
        self._opts_config = opts_config
        self._user_config = user_config
        self.import_user_dir()
        model_config = self._build_model_config(opts_config)
        dataset_config = self._build_dataset_config(opts_config)
        args_overrides = self._build_demjson_config(self.args.config_override)
        other_configs = OmegaConf.merge(model_config, dataset_config,
            user_config, args_overrides)
        return other_configs

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    def _build_user_config(self, opts):
        user_config = {}
        self.config_path = opts.config
        if self.config_path is not None:
            user_config = load_yaml(self.config_path)
        return user_config

    def import_user_dir(self):
        user_dir = self._default_config.env.user_dir
        user_config_user_dir = self._user_config.get('env', {}).get('user_dir',
            None)
        if user_config_user_dir:
            user_dir = user_config_user_dir
        opts_user_dir = self._opts_config.get('env', {}).get('user_dir', None)
        if opts_user_dir:
            user_dir = opts_user_dir
        if user_dir:
            import_user_module(user_dir)

    def _build_model_config(self, config):
        model = config.model
        if model is None:
            raise KeyError("Required argument 'model' not passed")
        model_cls = registry.get_model_class(model)
        if model_cls is None:
            warning = f"No model named '{model}' has been registered"
            warnings.warn(warning)
            return OmegaConf.create()
        default_model_config_path = model_cls.config_path()
        if default_model_config_path is None:
            warning = ("Model {}'s class has no default configuration provided"
                .format(model))
            warnings.warn(warning)
            return OmegaConf.create()
        return load_yaml(default_model_config_path)

    def _build_dataset_config(self, config):
        dataset = config.dataset
        datasets = config.datasets
        if dataset is None and datasets is None:
            raise KeyError("Required argument 'dataset|datasets' not passed")
        if datasets is None:
            config.datasets = dataset
            datasets = dataset.split(',')
        else:
            datasets = datasets.split(',')
        dataset_config = OmegaConf.create()
        for dataset in datasets:
            builder_cls = registry.get_builder_class(dataset)
            if builder_cls is None:
                warning = f"No dataset named '{dataset}' has been registered"
                warnings.warn(warning)
                continue
            default_dataset_config_path = builder_cls.config_path()
            if default_dataset_config_path is None:
                warning = (
                    "Dataset {}'s builder class has no default configuration "
                     + f'provided')
                warnings.warn(warning)
                continue
            dataset_config = OmegaConf.merge(dataset_config, load_yaml(
                default_dataset_config_path))
        return dataset_config

    def get_config(self):
        self._register_resolvers()
        return self.config

    def _build_demjson_config(self, demjson_string):
        if demjson_string is None:
            return OmegaConf.create()
        demjson_dict = demjson.decode(demjson_string)
        return OmegaConf.create(demjson_dict)

    def _get_args_config(self, args):
        args_dict = vars(args)
        return OmegaConf.create(args_dict)

    def _register_resolvers(self):
        OmegaConf.clear_resolvers()
        device_count = max(1, torch.cuda.device_count())
        OmegaConf.register_resolver('device_count', lambda : device_count)
        OmegaConf.register_resolver('resolve_cache_dir', resolve_cache_dir)
        OmegaConf.register_resolver('resolve_dir', resolve_dir)

    def _merge_with_dotlist(self, config, opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return config
        has_equal = opts[0].find('=') != -1
        if has_equal:
            opt_values = [opt.split('=') for opt in opts]
        else:
            assert len(opts) % 2 == 0, 'Number of opts should be multiple of 2'
            opt_values = zip(opts[0::2], opts[1::2])
        for opt, value in opt_values:
            if opt == 'dataset':
                opt = 'datasets'
            splits = opt.split('.')
            current = config
            for idx, field in enumerate(splits):
                array_index = -1
                if field.find('[') != -1 and field.find(']') != -1:
                    stripped_field = field[:field.find('[')]
                    array_index = int(field[field.find('[') + 1:field.find(
                        ']')])
                else:
                    stripped_field = field
                if stripped_field not in current:
                    raise AttributeError(
                        'While updating configuration option {} is missing from configuration at field {}'
                        .format(opt, stripped_field))
                if isinstance(current[stripped_field], collections.abc.Mapping
                    ):
                    current = current[stripped_field]
                elif isinstance(current[stripped_field], collections.abc.
                    Sequence) and array_index != -1:
                    current_value = current[stripped_field][array_index]
                    if not isinstance(current_value, (collections.abc.
                        Mapping, collections.abc.Sequence)):
                        print(f'Overriding option {opt} to {value}')
                        current[stripped_field][array_index
                            ] = self._decode_value(value)
                    else:
                        current = current_value
                elif idx == len(splits) - 1:
                    print(f'Overriding option {opt} to {value}')
                    current[stripped_field] = self._decode_value(value)
                else:
                    raise AttributeError('While updating configuration',
                        'option {} is not present after field {}'.format(
                        opt, stripped_field))
        return config

    def _decode_value(self, value):
        if not isinstance(value, str):
            return value
        if value == 'None':
            value = None
        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def freeze(self):
        OmegaConf.set_struct(self.config, True)

    def defrost(self):
        OmegaConf.set_struct(self.config, False)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return opts
        has_equal = opts[0].find('=') != -1
        if has_equal:
            return opts
        return [(opt + '=' + value) for opt, value in zip(opts[0::2], opts[
            1::2])]

    def pretty_print(self):
        if not self.config.training.log_detailed_config:
            return
        self.writer = registry.get('writer')
        self.writer.write('=====  Training Parameters    =====', 'info')
        self.writer.write(self._convert_node_to_json(self.config.training),
            'info')
        self.writer.write('======  Dataset Attributes  ======', 'info')
        datasets = self.config.datasets.split(',')
        for dataset in datasets:
            if dataset in self.config.dataset_config:
                self.writer.write(f'======== {dataset} =======', 'info')
                dataset_config = self.config.dataset_config[dataset]
                self.writer.write(self._convert_node_to_json(dataset_config
                    ), 'info')
            else:
                self.writer.write(
                    f"No dataset named '{dataset}' in config. Skipping",
                    'warning')
        self.writer.write('======  Optimizer Attributes  ======', 'info')
        self.writer.write(self._convert_node_to_json(self.config.optimizer),
            'info')
        if self.config.model not in self.config.model_config:
            raise ValueError(
                f'{self.config.model} not present in model attributes')
        self.writer.write(
            f'======  Model ({self.config.model}) Attributes  ======', 'info')
        self.writer.write(self._convert_node_to_json(self.config.
            model_config[self.config.model]), 'info')

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def _update_specific(self, config):
        self.writer = registry.get('writer')
        if config.learning_rate:
            if 'optimizer' in config and 'params' in config.optimizer:
                lr = config.learning_rate
                config.optimizer.params.lr = lr
        if not torch.cuda.is_available() and 'cuda' in config.training.device:
            warnings.warn(
                "Device specified is 'cuda' but cuda is not present. " +
                'Switching to CPU version.')
            config.training.device = 'cpu'
        return config

    def upgrade(self, config):
        mapping = {'training.resume_file': 'checkpoint.resume_file',
            'training.resume': 'checkpoint.resume', 'training.resume_best':
            'checkpoint.resume_best', 'training.load_pretrained':
            'checkpoint.resume_pretrained',
            'training.pretrained_state_mapping':
            'checkpoint.pretrained_state_mapping', 'training.run_type':
            'run_type'}
        for old, new in mapping.items():
            value = OmegaConf.select(config, old)
            if value:
                OmegaConf.update(config, new, value)


def get_global_config(key=None):
    config = registry.get('config')
    if config is None:
        configuration = Configuration()
        config = configuration.get_config()
        registry.register('config', config)
    if key:
        config = OmegaConf.select(config, key)
    return config


def get_mmf_cache_dir():
    config = get_global_config()
    cache_dir = config.env.cache_dir
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(get_mmf_root(), cache_dir)
    return cache_dir


class MMBTForPreTraining(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config
        pretraining_module = BertForPreTraining.from_pretrained(self.config
            .bert_model_name, config=self.encoder_config, cache_dir=os.path
            .join(get_mmf_cache_dir(), 'distributed_{}'.format(-1)))
        self.cls = deepcopy(pretraining_module.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we
            are cloning them instead.
        """
        if hasattr(self, 'cls'):
            self.bert.mmbt.transformer._tie_or_clone_weights(self.cls.
                predictions.decoder, self.bert.mmbt.transformer.embeddings.
                word_embeddings)

    def forward(self, sample_list):
        module_output = self.bert(sample_list)
        sequence_output, pooled_output = module_output[0], module_output[1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output,
            pooled_output)
        output = {}
        if (self.encoder_config.output_hidden_states or self.encoder_config
            .output_attentions):
            output['extras'] = module_output[2:]
        loss_key = f'{sample_list.dataset_name}/{sample_list.dataset_type}'
        if ('lm_label_ids' in sample_list and sample_list.lm_label_ids is not
            None):
            output['logits'] = prediction_scores
            lm_label_ids = sample_list.lm_label_ids
            text_scores = prediction_scores[:, -lm_label_ids.size(1):
                ].contiguous().view(-1, self.encoder_config.vocab_size)
            masked_lm_loss = self.loss_fct(text_scores, sample_list.
                lm_label_ids.contiguous().view(-1))
            output['losses'] = {}
            output['losses'][f'{loss_key}/masked_lm_loss'] = masked_lm_loss
        if ('image_text_alignment' in sample_list and sample_list.
            image_text_alignment is not None):
            output['seq_relationship_logits'] = seq_relationship_score
            alignment_loss = self.loss_fct(seq_relationship_score.
                contiguous().view(-1), sample_list.image_text_alignment.
                contiguous().view(-1))
            output['losses'][f'{loss_key}/alignment_loss'] = alignment_loss
        return output


class MMBTForClassification(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config
        self.dropout = nn.Dropout(self.encoder_config.hidden_dropout_prob)
        self.classifier = nn.Sequential(BertPredictionHeadTransform(self.
            encoder_config), nn.Linear(self.encoder_config.hidden_size,
            self.config.num_labels))

    def forward(self, sample_list):
        module_output = self.bert(sample_list)
        pooled_output = module_output[1]
        output = {}
        if (self.encoder_config.output_hidden_states or self.encoder_config
            .output_attentions):
            output['extras'] = module_output[2:]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)
        output['scores'] = reshaped_logits
        return output


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.visualization = config.visualization
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.visualization:
            attn_data = {'attn': attention_probs, 'queries': query_layer,
                'keys': key_layer}
        else:
            attn_data = None
        return context_layer, attn_data


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states,
            attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.v_hidden_size, config.v_num_attention_heads))
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(config.v_hidden_size / config.
            v_num_attention_heads)
        self.visualization = config.visualization
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)
        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding,
        txt_attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))
            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.visualization:
            attn_data = {'attn': attention_probs, 'queries': query_layer,
                'keys': key_layer}
        else:
            attn_data = None
        return context_layer, attn_data


class BertImageSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding,
        txt_attention_mask):
        self_output, attention_probs = self.self(input_tensor,
            attention_mask, txt_embedding, txt_attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


ACT2FN = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU}


class BertImageIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size
            )
        if isinstance(config.v_hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size
            )
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding,
        txt_attention_mask):
        attention_output, attention_probs = self.attention(hidden_states,
            attention_mask, txt_embedding, txt_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.bi_hidden_size, config.bi_num_attention_heads))
        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(config.bi_hidden_size / config.
            bi_num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)
        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2,
        attention_mask2, co_attention_mask=None, use_co_attention_mask=False):
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose
            (-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.
            attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.
            all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose
            (-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.
            attention_head_size)
        attention_scores2 = attention_scores2 + attention_mask2
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.
            all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        attn_data = None
        if self.visualization:
            attn_data = {'attn1': attention_probs1, 'queries1':
                query_layer2, 'keys1': key_layer1, 'attn2':
                attention_probs2, 'querues2': query_layer1, 'keys2': key_layer2
                }
        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)
        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)
        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2,
        input_tensor2):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)
        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)
        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)
        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.biattention = BertBiAttention(config)
        self.biOutput = BertBiOutput(config)
        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)
        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(self, input_tensor1, attention_mask1, input_tensor2,
        attention_mask2, co_attention_mask=None, use_co_attention_mask=False):
        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1, attention_mask1, input_tensor2, attention_mask2,
            co_attention_mask, use_co_attention_mask)
        attention_output1, attention_output2 = self.biOutput(bi_output2,
            input_tensor1, bi_output1, input_tensor2)
        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)
        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)
        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)
        self.layer = nn.ModuleList([deepcopy(layer) for _ in range(config.
            num_hidden_layers)])
        self.v_layer = nn.ModuleList([deepcopy(v_layer) for _ in range(
            config.v_num_hidden_layers)])
        self.c_layer = nn.ModuleList([deepcopy(connect_layer) for _ in
            range(len(config.v_biattention_id))])

    def forward(self, txt_embedding, image_embedding, txt_attention_mask,
        txt_attention_mask2, image_attention_mask, co_attention_mask=None,
        output_all_encoded_layers=True, output_all_attention_masks=False):
        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []
        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []
        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()
        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.
            t_biattention_id):
            v_end = v_layer_id
            t_end = t_layer_id
            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end
            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask)
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)
            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask)
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)
            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding, image_attention_mask,
                        txt_embedding, txt_attention_mask2)
                    v_start = self.fixed_v_layer
                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)
            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding, image_attention_mask, txt_embedding,
                    txt_attention_mask2)
                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)
            if count == 0 and self.in_batch_pairs:
                image_embedding = image_embedding.unsqueeze(0).expand(
                    batch_size, batch_size, num_regions, v_hidden_size
                    ).contiguous().view(batch_size * batch_size,
                    num_regions, v_hidden_size)
                image_attention_mask = image_attention_mask.unsqueeze(0
                    ).expand(batch_size, batch_size, 1, 1, num_regions
                    ).contiguous().view(batch_size * batch_size, 1, 1,
                    num_regions)
                txt_embedding = txt_embedding.unsqueeze(1).expand(batch_size,
                    batch_size, num_words, t_hidden_size).contiguous().view(
                    batch_size * batch_size, num_words, t_hidden_size)
                txt_attention_mask = txt_attention_mask.unsqueeze(1).expand(
                    batch_size, batch_size, 1, 1, num_words).contiguous().view(
                    batch_size * batch_size, 1, 1, num_words)
                co_attention_mask = co_attention_mask.unsqueeze(1).expand(
                    batch_size, batch_size, 1, num_regions, num_words
                    ).contiguous().view(batch_size * batch_size, 1,
                    num_regions, num_words)
            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(image_embedding.size(0
                    ), txt_embedding.size(1), txt_embedding.size(2))
                txt_attention_mask = txt_attention_mask.expand(image_embedding
                    .size(0), txt_attention_mask.size(1),
                    txt_attention_mask.size(2), txt_attention_mask.size(3))
            if self.with_coattention:
                image_embedding, txt_embedding, co_attention_probs = (self.
                    c_layer[count](image_embedding, image_attention_mask,
                    txt_embedding, txt_attention_mask, co_attention_mask,
                    use_co_attention_mask))
                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)
            v_start = v_end
            t_start = t_end
            count += 1
            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)
        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding, image_attention_mask, txt_embedding,
                txt_attention_mask2)
            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)
        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding,
                txt_attention_mask)
            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)
        return all_encoder_layers_t, all_encoder_layers_v, (
            all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c)


class BertTextPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, (0)]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImgPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImagePredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertImgPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(self, sequence_output_t, sequence_output_v, pooled_output_t,
        pooled_output_v):
        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError
        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v = self.imagePredictions(sequence_output_v)
        return prediction_scores_t, prediction_scores_v, seq_relationship_score


class BertImageFeatureEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.image_embeddings = nn.Linear(config.v_feature_size, config.
            v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, image_feature, image_location):
        img_embeddings = self.image_embeddings(image_feature)
        loc_embeddings = self.image_location_embeddings(image_location)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ViLBERTForPretraining(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = ViLBERTBase.from_pretrained(self.config.bert_model_name,
            config=BertConfig.from_dict(OmegaConf.to_container(self.config,
            resolve=True)), cache_dir=os.path.join(get_mmf_cache_dir(),
            'distributed_{}'.format(-1)))
        self.cls = BertPreTrainingHeads(config)
        self.vocab_size = self.config.vocab_size
        self.visual_target = config.visual_target
        self.num_negative = config.num_negative
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        if self.visual_target == 0:
            self.vis_criterion = nn.KLDivLoss(reduction='none')
        elif self.visual_target == 1:
            self.vis_criterion = nn.MSELoss(reduction='none')
        elif self.visual_target == 2:
            self.vis_criterion = CrossEntropyLoss()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)
            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning
            them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.
            embeddings.word_embeddings)

    def forward(self, input_ids, image_feature, image_location,
        token_type_ids=None, attention_mask=None, image_attention_mask=None,
        masked_lm_labels=None, image_label=None, image_target=None,
        next_sentence_label=None, output_all_attention_masks=False):
        (sequence_output_t, sequence_output_v, pooled_output_t,
            pooled_output_v, attention_weights) = (self.bert(input_ids,
            image_feature, image_location, token_type_ids, attention_mask,
            image_attention_mask, output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks))
        (prediction_scores_t, prediction_scores_v, seq_relationship_score) = (
            self.cls(sequence_output_t, sequence_output_v, pooled_output_t,
            pooled_output_v))
        output = {}
        if output_all_attention_masks:
            output['attention_weights'] = attention_weights
        if image_target is not None:
            if self.visual_target == 1:
                img_loss = self.vis_criterion(prediction_scores_v, image_target
                    )
                masked_img_loss = torch.sum(img_loss * (image_label == 1).
                    unsqueeze(2).float()) / max(torch.sum((image_label == 1
                    ).unsqueeze(2).expand_as(img_loss)), 1)
            elif self.visual_target == 0:
                img_loss = self.vis_criterion(F.log_softmax(
                    prediction_scores_v, dim=2), image_target)
                masked_img_loss = torch.sum(img_loss * (image_label == 1).
                    unsqueeze(2).float()) / max(torch.sum(image_label == 1), 0)
            elif self.visual_target == 2:
                num_across_batch = int(self.num_negative * 0.7)
                num_inside_batch = int(self.num_negative * 0.3)
                batch_size, num_regions, _ = prediction_scores_v.size()
                assert batch_size != 0
                row_across_index = input_ids.new(batch_size, num_regions,
                    num_across_batch).random_(0, batch_size - 1)
                col_across_index = input_ids.new(batch_size, num_regions,
                    num_across_batch).random_(0, num_regions)
                for i in range(batch_size - 1):
                    row_across_index[i][row_across_index[i] == i
                        ] = batch_size - 1
                final_across_index = (row_across_index * num_regions +
                    col_across_index)
                row_inside_index = input_ids.new(batch_size, num_regions,
                    num_inside_batch).zero_()
                col_inside_index = input_ids.new(batch_size, num_regions,
                    num_inside_batch).random_(0, num_regions - 1)
                for i in range(batch_size):
                    row_inside_index[i] = i
                for i in range(num_regions - 1):
                    col_inside_index[:, (i), :][col_inside_index[:, (i), :] ==
                        i] = num_regions - 1
                final_inside_index = (row_inside_index * num_regions +
                    col_inside_index)
                final_index = torch.cat((final_across_index,
                    final_inside_index), dim=2)
                predict_v = prediction_scores_v[image_label == 1]
                neg_index_v = final_index[image_label == 1]
                flat_image_target = image_target.view(batch_size *
                    num_regions, -1)
                negative_v = flat_image_target[neg_index_v]
                positive_v = image_target[image_label == 1]
                sample_v = torch.cat((positive_v.unsqueeze(1), negative_v),
                    dim=1)
                score = torch.bmm(sample_v, predict_v.unsqueeze(2)).squeeze(2)
                masked_img_loss = self.vis_criterion(score, input_ids.new(
                    score.size(0)).zero_())
            output['masked_img_loss'] = masked_img_loss.unsqueeze(0)
        masked_lm_loss = self.loss_fct(prediction_scores_t.view(-1, self.
            vocab_size), masked_lm_labels.view(-1))
        output['masked_lm_loss'] = masked_lm_loss.unsqueeze(0)
        return output


class ViLBERTForClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = ViLBERTBase.from_pretrained(self.config.bert_model_name,
            config=BertConfig.from_dict(OmegaConf.to_container(self.config,
            resolve=True)), cache_dir=os.path.join(get_mmf_cache_dir(),
            'distributed_{}'.format(-1)))
        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        classifier_config = deepcopy(config)
        classifier_config.hidden_size = config.bi_hidden_size
        if self.config.training_head_type == 'nlvr2':
            classifier_config.hidden_size *= 2
        self.classifier = nn.Sequential(BertPredictionHeadTransform(
            classifier_config), nn.Linear(classifier_config.hidden_size,
            self.num_labels))
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                self.bert.init_weights()
            self.classifier.apply(self.bert._init_weights)

    def forward(self, input_ids, image_feature, image_location,
        token_type_ids=None, attention_mask=None, image_attention_mask=None,
        masked_lm_labels=None, image_label=None, image_target=None,
        next_sentence_label=None, output_all_attention_masks=False):
        (sequence_output_t, sequence_output_v, pooled_output_t,
            pooled_output_v, attention_weights) = (self.bert(input_ids,
            image_feature, image_location, token_type_ids, attention_mask,
            image_attention_mask, output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks))
        output = {}
        if output_all_attention_masks:
            output['attention_weights'] = attention_weights
        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError
        if self.training_head_type == 'nlvr2':
            pooled_output = pooled_output.view(-1, pooled_output.size(1) * 2)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output['scores'] = reshaped_logits
        return output


class VisualBERTForPretraining(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.bert_model_name = getattr(self.config, 'bert_model_name', None)
        self.bert_config = BertConfig.from_dict(OmegaConf.to_container(self
            .config, resolve=True))
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states)
        else:
            self.bert = VisualBERTBase.from_pretrained(self.config.
                bert_model_name, config=self.bert_config, cache_dir=os.path
                .join(get_mmf_cache_dir(), 'distributed_{}'.format(-1)),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states)
        self.vocab_size = self.bert.config.vocab_size
        if self.bert_model_name is None:
            bert_masked_lm = BertForPreTraining(self.bert.config)
        else:
            bert_masked_lm = BertForPreTraining.from_pretrained(self.config
                .bert_model_name, cache_dir=os.path.join(get_mmf_cache_dir(
                ), 'distributed_{}'.format(-1)))
        self.cls = deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)
            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them
            instead.
        """
        self.bert._tie_or_clone_weights(self.cls.predictions.decoder, self.
            bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
        visual_embeddings=None, position_embeddings_visual=None,
        visual_embeddings_type=None, image_text_alignment=None,
        masked_lm_labels=None):
        sequence_output, pooled_output, attention_weights = self.bert(input_ids
            , attention_mask, token_type_ids, visual_embeddings,
            position_embeddings_visual, visual_embeddings_type,
            image_text_alignment)
        output_dict = {}
        if self.output_attentions:
            output_dict['attention_weights'] = attention_weights
        if self.output_hidden_states:
            output_dict['sequence_output'] = sequence_output
            output_dict['pooled_output'] = pooled_output
        prediction_scores, seq_relationship_score = self.cls(sequence_output,
            pooled_output)
        if masked_lm_labels is not None:
            output_dict['logits'] = prediction_scores
            masked_lm_loss = self.loss_fct(prediction_scores.contiguous().
                view(-1, self.vocab_size), masked_lm_labels.contiguous().
                view(-1))
            output_dict['masked_lm_loss'] = masked_lm_loss
            output_dict['loss'] = masked_lm_loss
        return output_dict


class VisualBERTForClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.bert_model_name = getattr(self.config, 'bert_model_name', None)
        self.bert_config = BertConfig.from_dict(OmegaConf.to_container(self
            .config, resolve=True))
        if self.bert_model_name is None:
            self.bert = VisualBERTBase(self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states)
        else:
            self.bert = VisualBERTBase.from_pretrained(self.config.
                bert_model_name, config=self.bert_config, cache_dir=os.path
                .join(get_mmf_cache_dir(), 'distributed_{}'.format(-1)),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states)
        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if self.config.training_head_type == 'nlvr2':
            self.bert.config.hidden_size *= 2
        self.classifier = nn.Sequential(BertPredictionHeadTransform(self.
            bert.config), nn.Linear(self.bert.config.hidden_size, self.
            config.num_labels))
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                self.bert.init_weights()
            self.classifier.apply(self.bert._init_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
        visual_embeddings=None, position_embeddings_visual=None,
        visual_embeddings_type=None, image_text_alignment=None,
        masked_lm_labels=None):
        sequence_output, pooled_output, attention_weights = self.bert(input_ids
            , attention_mask, token_type_ids, visual_embeddings,
            position_embeddings_visual, visual_embeddings_type,
            image_text_alignment)
        if self.training_head_type == 'nlvr2':
            b, h = pooled_output.size()
            pooled_output = torch.cat([pooled_output[:b // 2],
                pooled_output[b // 2:]], dim=1)
        output_dict = {}
        if self.output_attentions:
            output_dict['attention_weights'] = attention_weights
        if self.output_hidden_states:
            output_dict['sequence_output'] = sequence_output
            output_dict['pooled_output'] = pooled_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict['scores'] = reshaped_logits
        return output_dict


class AttentionLayer(nn.Module):

    def __init__(self, image_dim, question_dim, **kwargs):
        super().__init__()
        combine_type = kwargs['modal_combine']['type']
        combine_params = kwargs['modal_combine']['params']
        modal_combine_layer = ModalCombineLayer(combine_type, image_dim,
            question_dim, **combine_params)
        transform_type = kwargs['transform']['type']
        transform_params = kwargs['transform']['params']
        transform_layer = TransformLayer(transform_type,
            modal_combine_layer.out_dim, **transform_params)
        normalization = kwargs['normalization']
        self.module = TopDownAttention(modal_combine_layer, transform_layer,
            normalization)
        if hasattr(self.module, 'out_dim'):
            self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ConcatenationAttention(nn.Module):

    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa = GatedTanh(image_feat_dim + txt_rnn_embeding_dim, hidden_size)
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(question_embedding, 1
            ).expand(-1, num_location, -1)
        concat_feature = torch.cat((image_feat, question_embedding_expand),
            dim=2)
        raw_attention = self.lc(self.fa(concat_feature))
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class ProjectAttention(nn.Module):

    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size,
        dropout=0.2):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa_image = GatedTanh(image_feat_dim, hidden_size)
        self.fa_txt = GatedTanh(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        num_location = image_feat.shape[1]
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(question_fa, 1).expand(-1,
            num_location, -1)
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class DoubleProjectAttention(nn.Module):

    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size,
        dropout=0.2):
        super().__init__()
        self.att1 = ProjectAttention(image_feat_dim, txt_rnn_embeding_dim,
            hidden_size, dropout)
        self.att2 = ProjectAttention(image_feat_dim, txt_rnn_embeding_dim,
            hidden_size, dropout)
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attn_weights = att1 + att2
        attention_weights = nn.functional.softmax(raw_attn_weights, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class TopDownAttention(nn.Module):
    EPS = 1e-08

    def __init__(self, combination_layer, transform_module, normalization):
        super().__init__()
        self.combination_layer = combination_layer
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype
            ).unsqueeze(dim=0)
        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = mask.unsqueeze(dim=2).expand_as(attention)
        attention = attention.masked_fill(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        joint_feature = self.combination_layer(image_feat, question_embedding)
        raw_attn = self.transform(joint_feature)
        if self.normalization.lower() == 'softmax':
            attention = nn.functional.softmax(raw_attn, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention, dim=1,
                    keepdim=True)
                masked_attention_sum += masked_attention_sum.eq(0).float(
                    ) + self.EPS
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention
        elif self.normalization.lower() == 'sigmoid':
            attention = torch.sigmoid(raw_attn)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
        return masked_attention


class VisDialDiscriminator(nn.Module):

    def __init__(self, config, embedding):
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.emb_out_dim = embedding.text_out_dim
        self.hidden_dim = self.config.hidden_dim
        self.projection_layer = nn.Linear(self.emb_out_dim, self.hidden_dim)

    def forward(self, encoder_output, batch):
        answer_options_len = batch['answer_options_len']
        answer_options = batch['answer_options']
        max_seq_len = answer_options.size(-1)
        batch_size, ndialogues, noptions, seq_len = answer_options.size()
        answer_options = answer_options.view(-1, max_seq_len)
        answer_options_len = answer_options_len.view(-1)
        answer_options = self.embedding(answer_options)
        answer_options = self.projection_layer(answer_options)
        answer_options = answer_options.view(batch_size * ndialogues,
            noptions, self.hidden_dim)
        encoder_output = encoder_output.unsqueeze(1).expand(-1, noptions, -1)
        scores = torch.sum(answer_options * encoder_output, dim=2)
        return scores


class LanguageDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.language_lstm = nn.LSTMCell(in_dim + kwargs['hidden_dim'],
            kwargs['hidden_dim'], bias=True)
        self.fc = weight_norm(nn.Linear(kwargs['hidden_dim'], out_dim))
        self.dropout = nn.Dropout(p=kwargs['dropout'])
        self.init_weights(kwargs['fc_bias_init'])

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn):
        state = registry.get(f'{weighted_attn.device}_lstm_state')
        h1, c1 = state['td_hidden']
        h2, c2 = state['lm_hidden']
        h2, c2 = self.language_lstm(torch.cat([weighted_attn, h1], dim=1),
            (h2, c2))
        predictions = self.fc(self.dropout(h2))
        state['lm_hidden'] = h2, c2
        return predictions


class TextEmbedding(nn.Module):

    def __init__(self, emb_type, **kwargs):
        super().__init__()
        self.model_data_dir = kwargs.get('model_data_dir', None)
        self.embedding_dim = kwargs.get('embedding_dim', None)
        if emb_type == 'identity':
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == 'vocab':
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == 'projection':
            self.module = ProjectionEmbedding(**kwargs)
            self.module.text_out_dim = self.module.out_dim
        elif emb_type == 'preextracted':
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == 'bilstm':
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == 'attention':
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == 'torch':
            vocab_size = kwargs['vocab_size']
            embedding_dim = kwargs['embedding_dim']
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" %
                emb_type)
        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BaseVocab:
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    UNK_TOKEN = '<unk>'
    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_file=None, embedding_dim=300, data_dir=None, *
        args, **kwargs):
        """Vocab class to be used when you want to train word embeddings from
        scratch based on a custom vocab. This will initialize the random
        vectors for the vocabulary you pass. Get the vectors using
        `get_vectors` function. This will also create random embeddings for
        some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
        UNK - <unk>.

        Parameters
        ----------
        vocab_file : str
            Path of the vocabulary file containing one word per line
        embedding_dim : int
            Size of the embedding

        """
        self.type = 'base'
        self.word_dict = {}
        self.itos = {}
        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN
        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX
        index = len(self.itos.keys())
        self.total_predefined = len(self.itos.keys())
        if vocab_file is not None:
            if not os.path.isabs(vocab_file) and data_dir is not None:
                mmf_root = get_mmf_root()
                vocab_file = os.path.join(mmf_root, data_dir, vocab_file)
            if not PathManager.exists(vocab_file):
                raise RuntimeError('Vocab not found at ' + vocab_file)
            with PathManager.open(vocab_file, 'r') as f:
                for line in f:
                    self.itos[index] = line.strip()
                    self.word_dict[line.strip()] = index
                    index += 1
        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX
        self.stoi = defaultdict(self.get_unk_index)
        self.stoi.update(self.word_dict)
        self.vectors = torch.FloatTensor(self.get_size(), embedding_dim)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)

    def get_pad_index(self):
        return self.PAD_INDEX

    def get_pad_token(self):
        return self.PAD_TOKEN

    def get_start_index(self):
        return self.SOS_INDEX

    def get_start_token(self):
        return self.SOS_TOKEN

    def get_end_index(self):
        return self.EOS_INDEX

    def get_end_token(self):
        return self.EOS_TOKEN

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def get_vectors(self):
        return getattr(self, 'vectors', None)

    def get_embedding(self, cls, **embedding_kwargs):
        vector_dim = len(self.vectors[0])
        embedding_kwargs['vocab_size'] = self.get_size()
        embedding_dim = embedding_kwargs['embedding_dim']
        embedding_kwargs['embedding_dim'] = vector_dim
        embedding = None
        if cls == torch.nn.Embedding:
            embedding = torch.nn.Embedding(self.get_size(), vector_dim)
        else:
            embedding = cls(**embedding_kwargs)
        if hasattr(embedding, 'embedding'):
            embedding.embedding = torch.nn.Embedding.from_pretrained(self.
                vectors, freeze=False)
        else:
            embedding = torch.nn.Embedding.from_pretrained(self.vectors,
                freeze=False)
        if vector_dim == embedding_dim:
            return embedding
        else:
            return torch.nn.Sequential([embedding, torch.nn.Linear(
                vector_dim, embedding_dim)])


class CustomVocab(BaseVocab):

    def __init__(self, vocab_file, embedding_file, data_dir=None, *args, **
        kwargs):
        """Use this vocab class when you have a custom vocab as well as a
        custom embeddings file.

        This will inherit vocab class, so you will get predefined tokens with
        this one.

        IMPORTANT: To init your embedding, get your vectors from this class's
        object by calling `get_vectors` function

        Parameters
        ----------
        vocab_file : str
            Path of custom vocabulary
        embedding_file : str
            Path to custom embedding inititalization file
        data_dir : str
            Path to data directory if embedding file is not an absolute path.
            Default: None
        """
        super().__init__(vocab_file)
        self.type = 'custom'
        if not os.path.isabs(embedding_file) and data_dir is not None:
            mmf_root = get_mmf_root()
            embedding_file = os.path.join(mmf_root, data_dir, embedding_file)
        if not PathManager.exists(embedding_file):
            from mmf.common.registry import registry
            writer = registry.get('writer')
            error = "Embedding file path %s doesn't exist" % embedding_file
            if writer is not None:
                writer.write(error, 'error')
            raise RuntimeError(error)
        embedding_vectors = torch.from_numpy(np.load(embedding_file))
        self.vectors = torch.FloatTensor(self.get_size(), len(
            embedding_vectors[0]))
        for i in range(0, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i
        for i in range(4, self.get_size()):
            self.vectors[i] = embedding_vectors[i - 4]


class ExtractedVocab(BaseVocab):

    def __init__(self, base_path, emb_dim, *args, **kwargs):
        """Special vocab which is not really vocabulary but instead a class
        which returns embedding pre-extracted from files. Can be used load
        word embeddings from popular models like ELMo and BERT


        Parameters
        ----------
        base_path: str
            path containing saved files with embeddings one file per txt item
        """
        super().__init__(*args, **kwargs)
        self.type = 'extracted'
        self.emb_dim = emb_dim
        self.base_path = base_path

    def get_dim(self):
        return self.emb_dim


EMBEDDING_NAME_CLASS_MAPPING = {'glove': 'GloVe', 'fasttext': 'FastText'}


class IntersectedVocab(BaseVocab):

    def __init__(self, vocab_file, embedding_name, *args, **kwargs):
        """Use this vocab class when you have a custom vocabulary class but you
        want to use pretrained embedding vectos for it. This will only load
        the vectors which intersect with your vocabulary. Use the
        embedding_name specified in torchtext's pretrained aliases:
        ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d',
         'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d',
         'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
         'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d',
         'glove.6B.200d', 'glove.6B.300d']

        Parameters
        ----------
        vocab_file : str
            Vocabulary file containing list of words with one word per line
            which will be used to collect vectors
        embedding_name : str
            Embedding name picked up from the list of the pretrained aliases
            mentioned above
        """
        super().__init__(vocab_file, *args, **kwargs)
        self.type = 'intersected'
        name = embedding_name.split('.')[0]
        dim = embedding_name.split('.')[2][:-1]
        middle = embedding_name.split('.')[1]
        class_name = EMBEDDING_NAME_CLASS_MAPPING[name]
        if not hasattr(vocab, class_name):
            from mmf.common.registry import registry
            writer = registry.get('writer')
            error = 'Unknown embedding type: %s' % name, 'error'
            if writer is not None:
                writer.write(error, 'error')
            raise RuntimeError(error)
        params = [middle]
        if name == 'glove':
            params.append(int(dim))
        vector_cache = get_mmf_cache_dir()
        if is_master():
            vocab.pretrained_aliases[embedding_name](cache=vector_cache)
        synchronize()
        embedding = getattr(vocab, class_name)(*params, cache=vector_cache)
        self.vectors = torch.empty((self.get_size(), len(embedding.vectors[
            0])), dtype=torch.float)
        self.embedding_dim = len(embedding.vectors[0])
        for i in range(0, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i
        for i in range(4, self.get_size()):
            word = self.itos[i]
            embedding_index = embedding.stoi.get(word, None)
            if embedding_index is None:
                self.vectors[i] = self.vectors[self.UNK_INDEX]
            else:
                self.vectors[i] = embedding.vectors[embedding_index]

    def get_embedding_dim(self):
        return self.embedding_dim


class WordToVectorDict:

    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        return np.mean([self.model.get_word_vector(w) for w in word.split(
            ' ')], axis=0)


class ModelVocab(BaseVocab):

    def __init__(self, name, model_file, *args, **kwargs):
        """Special vocab which is not really vocabulary but instead a model
        which returns embedding directly instead of vocabulary. This is just
        an abstraction over a model which generates embeddings directly.
        For e.g. for fasttext model we encapsulate it inside this and provide
        it as a vocab so that the API of the vocab remains same.

        NOTE: stoi's functionality will remain same but it is actually calling
        a function to get word vectors. Currently, only fasttext is supported.

        Parameters
        ----------
        name : str
            Name of the embedding model which this vocab currently is loading
        model_file : str
            File from which model will be loaded. This API might need to be
            changed in future.
        """
        super().__init__(*args, **kwargs)
        self.type = 'model'
        if name != 'fasttext':
            raise ValueError('Model vocab only supports fasttext as of now')
        else:
            self._load_fasttext_model(model_file)

    def _load_fasttext_model(self, model_file):
        from fastText import load_model
        from mmf.common.registry import registry
        model_file = os.path.join(get_mmf_cache_dir(), model_file)
        registry.get('writer').write('Loading fasttext model now from %s' %
            model_file)
        self.model = load_model(model_file)
        self.stoi = WordToVectorDict(self.model)

    def get_embedding_dim(self):
        return self.model.get_dimension()


class PretrainedVocab(BaseVocab):

    def __init__(self, embedding_name, *args, **kwargs):
        """Use this if you want to use pretrained embedding. See description
        of IntersectedVocab to get a list of the embedding available from
        torchtext

        Parameters
        ----------
        embedding_name : str
            Name of the pretrained alias for the embedding to used
        """
        self.type = 'pretrained'
        if embedding_name not in vocab.pretrained_aliases:
            from mmf.common.registry import registry
            writer = registry.get('writer')
            error = 'Unknown embedding type: %s' % embedding_name, 'error'
            if writer is not None:
                writer.write(error, 'error')
            raise RuntimeError(error)
        vector_cache = get_mmf_cache_dir()
        if is_master():
            vocab.pretrained_aliases[embedding_name](cache=vector_cache)
        synchronize()
        embedding = vocab.pretrained_aliases[embedding_name](cache=vector_cache
            )
        self.UNK_INDEX = 3
        self.stoi = defaultdict(lambda : self.UNK_INDEX)
        self.itos = {}
        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN
        self.stoi[self.SOS_TOKEN] = self.SOS_INDEX
        self.stoi[self.EOS_TOKEN] = self.EOS_INDEX
        self.stoi[self.PAD_TOKEN] = self.PAD_INDEX
        self.stoi[self.UNK_TOKEN] = self.UNK_INDEX
        self.vectors = torch.FloatTensor(len(self.itos.keys()) + len(
            embedding.itos), len(embedding.vectors[0]))
        for i in range(4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i
        index = 4
        for word in embedding.stoi:
            self.itos[index] = word
            self.stoi[word] = index
            actual_index = embedding.stoi[word]
            self.vectors[index] = embedding.vectors[actual_index]
            index += 1


class Vocab:

    def __init__(self, *args, **params):
        vocab_type = params.get('type', 'pretrained')
        if vocab_type == 'random':
            if params['vocab_file'] is None:
                raise ValueError('No vocab path passed for vocab')
            self.vocab = BaseVocab(*args, **params)
        elif vocab_type == 'custom':
            if params['vocab_file'] is None or params['embedding_file'
                ] is None:
                raise ValueError(
                    'No vocab path or embedding_file passed for vocab')
            self.vocab = CustomVocab(*args, **params)
        elif vocab_type == 'pretrained':
            self.vocab = PretrainedVocab(*args, **params)
        elif vocab_type == 'intersected':
            if params['vocab_file'] is None or params['embedding_name'
                ] is None:
                raise ValueError(
                    'No vocab path or embedding_name passed for vocab')
            self.vocab = IntersectedVocab(*args, **params)
        elif vocab_type == 'extracted':
            if params['base_path'] is None or params['embedding_dim'] is None:
                raise ValueError(
                    'No base_path or embedding_dim passed for vocab')
            self.vocab = ExtractedVocab(*args, **params)
        elif vocab_type == 'model':
            if params['name'] is None or params['model_file'] is None:
                raise ValueError('No name or model_file passed for vocab')
            if params['name'] == 'fasttext':
                self.vocab = ModelVocab(*args, **params)
        else:
            raise ValueError('Unknown vocab type: %s' % vocab_type)
        self._dir_representation = dir(self)

    def __call__(self, *args, **kwargs):
        return self.vocab(*args, **kwargs)

    def __getattr__(self, name):
        if ('_dir_representation' in self.__dict__ and name in self.
            _dir_representation):
            return getattr(self, name)
        elif 'vocab' in self.__dict__ and hasattr(self.vocab, name):
            return getattr(self.vocab, name)
        else:
            type_vocab = 'Vocab'
            if 'vocab' in self.__dict__:
                type_vocab = type(self.vocab)
            raise AttributeError(
                f'{type_vocab} vocab type has no attribute {name}.')


class VocabEmbedding(nn.Module):

    def __init__(self, embedding_dim, **vocab_params):
        super().__init__()
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(nn.Embedding, embedding_dim=
            embedding_dim)

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout,
        bidirectional=False, rnn_type='GRU'):
        super().__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional
        if rnn_type == 'LSTM':
            rnn_cls = nn.LSTM
        elif rnn_type == 'GRU':
            rnn_cls = nn.GRU
        self.recurrent_encoder = rnn_cls(input_size=embedding_dim,
            hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        if self.bidirectional:
            return out[:, (-1)]
        forward_ = out[:, (-1), :self.num_hid]
        backward = out[:, (0), self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):

    def __init__(self, out_dim, base_path):
        super().__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + '.npy'))


class AttentionTextEmbedding(nn.Module):

    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs
        ):
        super().__init__()
        self.text_out_dim = hidden_dim * kwargs['conv2_out']
        bidirectional = kwargs.get('bidirectional', False)
        self.recurrent_unit = nn.LSTM(input_size=embedding_dim, hidden_size
            =hidden_dim // 2 if bidirectional else hidden_dim, num_layers=
            num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        conv1_out = kwargs['conv1_out']
        conv2_out = kwargs['conv2_out']
        kernel_size = kwargs['kernel_size']
        padding = kwargs['padding']
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=
            conv1_out, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=
            conv2_out, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)
        lstm_drop = self.dropout(lstm_out)
        lstm_reshape = lstm_drop.permute(0, 2, 1)
        qatt_conv1 = self.conv1(lstm_reshape)
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)
        return qtt_feature_concat


class ProjectionEmbedding(nn.Module):

    def __init__(self, module, in_dim, out_dim, **kwargs):
        super().__init__()
        if module == 'linear':
            self.layers = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        elif module == 'conv':
            last_out_channels = in_dim
            layers = []
            for conv in kwargs['convs']:
                layers.append(nn.Conv1d(in_channels=last_out_channels, **conv))
                last_out_channels = conv['out_channels']
            self.layers = nn.ModuleList(*layers)
            self.out_dim = last_out_channels
        else:
            raise TypeError(
                "Unknown module type for 'ProjectionEmbedding',use either 'linear' or 'conv'"
                )

    def forward(self, x):
        return self.layers(x)


class ImageFeatureEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()
        self.image_attention_model = AttentionLayer(img_dim, question_dim,
            **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims,
        extra=None):
        if extra is None:
            extra = {}
        attention = self.image_attention_model(image_feat_variable,
            question_embedding, image_dims)
        att_reshape = attention.permute(0, 2, 1)
        order_vectors = getattr(extra, 'order_vectors', None)
        if order_vectors is not None:
            image_feat_variable = torch.cat([image_feat_variable,
                order_vectors], dim=-1)
        tmp_embedding = torch.bmm(att_reshape, image_feat_variable)
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)
        return image_embedding, attention


class MultiHeadImageFeatureEmbedding(nn.Module):

    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()
        self.module = nn.MultiheadAttention(embed_dim=question_dim, kdim=
            img_dim, vdim=img_dim, **kwargs)
        self.out_dim = question_dim

    def forward(self, image_feat_variable, question_embedding, image_dims,
        extra=None):
        if extra is None:
            extra = {}
        image_feat_variable = image_feat_variable.transpose(0, 1)
        question_embedding = question_embedding.unsqueeze(1).transpose(0, 1)
        output, weights = self.module(question_embedding,
            image_feat_variable, image_feat_variable)
        output = output.transpose(0, 1)
        return output.squeeze(), weights


class ImageFinetune(nn.Module):

    def __init__(self, in_dim, weights_file, bias_file):
        super().__init__()
        with PathManager.open(weights_file, 'rb') as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, 'rb') as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]
        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3


class ImageFeatureEncoder(nn.Module):

    def __init__(self, encoder_type, in_dim, **kwargs):
        super().__init__()
        if encoder_type == 'default':
            self.module = Identity()
            self.module.in_dim = in_dim
            self.module.out_dim = in_dim
        elif encoder_type == 'projection':
            module_type = kwargs.pop('module', 'linear')
            self.module = ProjectionEmbedding(module_type, in_dim, **kwargs)
        elif encoder_type == 'finetune_faster_rcnn_fpn_fc7':
            self.module = FinetuneFasterRcnnFpnFc7(in_dim, **kwargs)
        else:
            raise NotImplementedError('Unknown Image Encoder: %s' %
                encoder_type)
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FinetuneFasterRcnnFpnFc7(nn.Module):

    def __init__(self, in_dim, weights_file, bias_file, model_data_dir, *
        args, **kwargs):
        super().__init__()
        model_data_dir = get_absolute_path(model_data_dir)
        if not os.path.isabs(weights_file):
            weights_file = os.path.join(model_data_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(model_data_dir, bias_file)
        if not PathManager.exists(bias_file) or not PathManager.exists(
            weights_file):
            download_path = download_pretrained_model('detectron')
            weights_file = get_absolute_path(os.path.join(download_path,
                'fc7_w.pkl'))
            bias_file = get_absolute_path(os.path.join(download_path,
                'fc7_b.pkl'))
        with PathManager.open(weights_file, 'rb') as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, 'rb') as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]
        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3


class ImageEncoder(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self._type = config.type
        params = config.params
        if self._type == 'default':
            self.module = nn.Identity()
            self.module.out_dim = params.in_dim
        elif self._type == 'resnet152':
            self.module = ResNet152ImageEncoder(params)
        else:
            raise NotImplementedError('Unknown Image Encoder: %s' % self._type)

    @property
    def out_dim(self):
        return self.module.out_dim

    def forward(self, image):
        return self.module(image)


class ResNet152ImageEncoder(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torchvision.models.resnet152(pretrained=config.get(
            'pretrained', True))
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        pool_func = (nn.AdaptiveAvgPool2d if config.pool_type == 'avg' else
            nn.AdaptiveMaxPool2d)
        if config.num_output_features == -1:
            self.pool = nn.Identity()
        elif config.num_output_features in [1, 2, 3, 5, 7]:
            self.pool = pool_func((config.num_output_features, 1))
        elif config.num_output_features == 4:
            self.pool = pool_func((2, 2))
        elif config.num_output_features == 6:
            self.pool = pool_func((3, 2))
        elif config.num_output_features == 8:
            self.pool = pool_func((4, 2))
        elif config.num_output_features == 9:
            self.pool = pool_func((3, 3))
        self.out_dim = 2048

    def forward(self, x):
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out


class TextEncoder(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self._type = config.type
        if self._type == 'identity':
            self.module = nn.Identity()
        elif self._type == 'transformer':
            self._module = TransformerEncoder(config.params)
            self.module = self._module.module
        elif self._type == 'embedding':
            self.module = TextEmbeddingEncoder(config.params)
        else:
            raise NotImplementedError(f'Unknown Text Encoder {self._type}')

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TextEmbeddingEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._operator = config.operator
        self._embedding_params = config.embedding_params
        self.module = TextEmbedding(self._embedding_params.type, **self.
            _embedding_params.params)

    def forward(self, x):
        x = self.module(x)
        if self._operator == 'sum':
            x = x.sum(dim=1)
        elif self._operator == 'concat':
            x = torch.cat(x, dim=1)
        elif self._operator == 'mul':
            x = torch.prod(x, dim=1)
        return x.squeeze()


class TransformerEncoder(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.module = AutoModel.from_pretrained(self.config.bert_model_name,
            config=self._build_encoder_config(config), cache_dir=os.path.
            join(get_mmf_cache_dir(), 'distributed_{}'.format(-1)))
        self.embeddings = self.module.embeddings
        self.config = self.module.config

    def _build_encoder_config(self, config):
        return AutoConfig.from_pretrained(self.config.bert_model_name, **
            OmegaConf.to_container(self.config))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)[1]


def build_image_encoder(config, direct_features=False, **kwargs):
    from mmf.modules.encoders import ImageFeatureEncoder, ImageEncoder
    if direct_features:
        module = ImageFeatureEncoder(config.type, **config.params)
    else:
        module = ImageEncoder(config)
    return module.module


def build_text_encoder(config, *args, **kwargs):
    from mmf.modules.encoders import TextEncoder
    text_encoder = TextEncoder(config, *args, **kwargs)
    return text_encoder.module


class MultiModalEncoderBase(nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self._modal_encoder_config = self.config.get('modal_encoder', None)
        self._is_direct_features_input = self.config.get(
            'direct_features_input', False)
        self.build()
        self.modal_hidden_size = self.config.get('modal_hidden_size', None)
        self.text_hidden_size = self.config.get('text_hidden_size', None)

    def build(self):
        encoders = self._build_encoders(self.config)
        self.text_encoder, self.modal_encoder = encoders[0], encoders[1]
        self._encoder_config = None
        if self.text_encoder:
            self._encoder_config = self.text_encoder.config

    @property
    def encoder_config(self):
        return self._encoder_config

    def _build_encoders(self, config):
        text_encoder = None
        if config.get('text_encoder', None):
            text_encoder = build_text_encoder(config.text_encoder)
        modal_encoder = None
        if config.get('modal_encoder', None):
            modal_encoder = self._build_modal_encoder(config.modal_encoder)
        return text_encoder, modal_encoder

    def _build_modal_encoder(self, config):
        return build_image_encoder(config, direct_features=self.
            _is_direct_features_input)


class CompactBilinearPooling(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=True):
        super().__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        self.sketch1 = nn.Parameter(self.generate_sketch_matrix(torch.
            randint(output_dim, size=(input_dim1,)), 2 * torch.randint(2,
            size=(input_dim1,)) - 1, input_dim1, output_dim), requires_grad
            =False)
        self.sketch2 = nn.Parameter(self.generate_sketch_matrix(torch.
            randint(output_dim, size=(input_dim2,)), 2 * torch.randint(2,
            size=(input_dim2,)) - 1, input_dim2, output_dim), requires_grad
            =False)

    def generate_sketch_matrix(self, rand_h, rand_s, input_dim, output_dim):
        return torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim,
            out=torch.LongTensor()), rand_h.long()]), rand_s.float(), [
            input_dim, output_dim]).to_dense()

    def forward(self, x1, x2):
        assert len(x1.shape) == len(x2.shape)
        if len(x1.shape) == 4 and len(x2.shape) == 4:
            fft1 = torch.rfft(x1.permute(0, 2, 3, 1).matmul(self.sketch1),
                signal_ndim=1)
            fft2 = torch.rfft(x2.permute(0, 2, 3, 1).matmul(self.sketch2),
                signal_ndim=1)
        else:
            fft1 = torch.rfft(x1.matmul(self.sketch1), signal_ndim=1)
            fft2 = torch.rfft(x2.matmul(self.sketch2), signal_ndim=1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 
            1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] *
            fft2[..., 0]], dim=-1)
        cbp = torch.irfft(fft_product, signal_ndim=1, signal_sizes=(self.
            output_dim,)) * self.output_dim
        if len(x1.shape) == 4 and len(x2.shape) == 4:
            cbp = cbp.sum(dim=[1, 2]) if self.sum_pool else cbp.permute(0, 
                3, 1, 2)
        return cbp


class MLP(nn.Module):

    def __init__(self, input_dim, dimensions, activation='relu', dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


@registry.register_fusion('block')
class Block(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20, rank
        =15, shared=False, dropout_input=0.0, dropout_pre_lin=0.0,
        dropout_output=0.0, pos_norm='before_cat'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size * rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)), self.
            merge_linears0, self.merge_linears1):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            m = m0(x0_c) * m1(x1_c)
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('block_tucker')
class BlockTucker(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20,
        shared=False, dropout_input=0.0, dropout_pre_lin=0.0,
        dropout_output=0.0, pos_norm='before_cat'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if self.shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        bilinears = []
        for size in self.sizes_list:
            bilinears.append(nn.Bilinear(size, size, size))
        self.bilinears = nn.ModuleList(bilinears)
        self.linear_out = nn.Linear(self.mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.dropout_input:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        zs = []
        for chunk_id, bilinear in enumerate(self.bilinears):
            x0_c = x0_chunks[chunk_id]
            x1_c = x1_chunks[chunk_id]
            z = bilinear(x0_c, x1_c)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z, p=2)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('mutan')
class Mutan(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, rank=15, shared
        =False, normalize=False, dropout_input=0.0, dropout_pre_lin=0.0,
        dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = rank
        self.output_dim = output_dim
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.normalize = normalize
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim * rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim * rank)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        m0 = self.merge_linear0(x0)
        m1 = self.merge_linear1(x1)
        m = m0 * m1
        m = m.view(-1, self.rank, self.mm_dim)
        z = torch.sum(m, 1)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('tucker')
class Tucker(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, shared=False,
        normalize=False, dropout_input=0.0, dropout_pre_lin=0.0,
        dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.bilinear = nn.Bilinear(mm_dim, mm_dim, mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = self.bilinear(x0, x1)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('mlb')
class MLB(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, activ_input=
        'relu', activ_output='relu', normalize=False, dropout_input=0.0,
        dropout_pre_lin=0.0, dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 * x1
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('mfb')
class MFB(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, factor=2,
        activ_input='relu', activ_output='relu', normalize=False,
        dropout_input=0.0, dropout_pre_norm=0.0, dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 * x1
        if self.dropout_pre_norm > 0:
            z = F.dropout(z, p=self.dropout_pre_norm, training=self.training)
        z = z.view(z.size(0), self.mm_dim, self.factor)
        z = z.sum(2)
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('mfh')
class MFH(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, factor=2,
        activ_input='relu', activ_output='relu', normalize=False,
        dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim * factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim * factor)
        self.linear_out = nn.Linear(mm_dim * 2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0_0(x[0])
        x1 = self.linear1_0(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z_0_skip = x0 * x1
        if self.dropout_pre_lin:
            z_0_skip = F.dropout(z_0_skip, p=self.dropout_pre_lin, training
                =self.training)
        z_0 = z_0_skip.view(z_0_skip.size(0), self.mm_dim, self.factor)
        z_0 = z_0.sum(2)
        if self.normalize:
            z_0 = torch.sqrt(F.relu(z_0)) - torch.sqrt(F.relu(-z_0))
            z_0 = F.normalize(z_0, p=2)
        x0 = self.linear0_1(x[0])
        x1 = self.linear1_1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z_1 = x0 * x1 * z_0_skip
        if self.dropout_pre_lin > 0:
            z_1 = F.dropout(z_1, p=self.dropout_pre_lin, training=self.training
                )
        z_1 = z_1.view(z_1.size(0), self.mm_dim, self.factor)
        z_1 = z_1.sum(2)
        if self.normalize:
            z_1 = torch.sqrt(F.relu(z_1)) - torch.sqrt(F.relu(-z_1))
            z_1 = F.normalize(z_1, p=2)
        cat_dim = z_0.dim() - 1
        z = torch.cat([z_0, z_1], cat_dim)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('mcb')
class MCB(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=16000, activ_output=
        'relu', dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_output = activ_output
        self.dropout_output = dropout_output
        self.mcb = CompactBilinearPooling(input_dims[0], input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        z = self.mcb(x[0], x[1])
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('linear_sum')
class LinearSum(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1200, activ_input=
        'relu', activ_output='relu', normalize=False, dropout_input=0.0,
        dropout_pre_lin=0.0, dropout_output=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        z = x0 + x1
        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z


@registry.register_fusion('concat_mlp')
class ConcatMLP(nn.Module):

    def __init__(self, input_dims, output_dim, dimensions=None, activation=
        'relu', dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        if dimensions is None:
            dimensions = [500, 500]
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        self.mlp = MLP(self.input_dim, self.dimensions, self.activation,
            self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.
            requires_grad)

    def forward(self, x):
        if x[0].dim() == 3 and x[1].dim() == 2:
            x[1] = x[1].unsqueeze(1).reshape_as(x[0])
        if x[1].dim() == 3 and x[0].dim() == 2:
            x[0] = x[0].unsqueeze(1).reshape_as(x[1])
        z = torch.cat(x, dim=x[0].dim() - 1)
        z = self.mlp(z)
        return z


class ConvNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding_size
        ='same', pool_stride=2, batch_norm=True):
        super().__init__()
        if padding_size == 'same':
            padding_size = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding=padding_size)
        self.max_pool2d = nn.MaxPool2d(pool_stride, stride=pool_stride)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.max_pool2d(nn.functional.leaky_relu(self.conv(x)))
        if self.batch_norm:
            x = self.batch_norm_2d(x)
        return x


class Flatten(nn.Module):

    def forward(self, input):
        if input.dim() > 1:
            input = input.view(input.size(0), -1)
        return input


class UnFlatten(nn.Module):

    def forward(self, input, sizes=None):
        if sizes is None:
            sizes = []
        return input.view(input.size(0), *sizes)


class GatedTanh(nn.Module):
    """
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\\in R^m => y \\in R^n
    	ilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = 	ilda(y) \\circ g
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = torch.tanh(self.fc(x))
        gated = torch.sigmoid(self.gate_fc(x))
        y = y_tilda * gated
        return y


class ReLUWithWeightNormFC(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ClassifierLayer(nn.Module):

    def __init__(self, classifier_type, in_dim, out_dim, **kwargs):
        super().__init__()
        if classifier_type == 'weight_norm':
            self.module = WeightNormClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == 'logit':
            self.module = LogitClassifier(in_dim, out_dim, **kwargs)
        elif classifier_type == 'language_decoder':
            self.module = LanguageDecoder(in_dim, out_dim, **kwargs)
        elif classifier_type == 'bert':
            self.module = BertClassifierHead(in_dim, out_dim, kwargs.get(
                'config', None)).module
        elif classifier_type == 'mlp':
            self.module = MLPClassifer(in_dim, out_dim, **kwargs)
        elif classifier_type == 'linear':
            self.module = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError('Unknown classifier type: %s' %
                classifier_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BertClassifierHead(nn.Module):

    def __init__(self, in_dim=768, out_dim=2, config=None, *args, **kwargs):
        super().__init__()
        if config is None:
            config = BertConfig.from_pretrained('bert-base-uncased')
        assert config.hidden_size == in_dim
        self.module = nn.Sequential(nn.Dropout(config.hidden_dropout_prob),
            BertPredictionHeadTransform(config), nn.Linear(in_dim, out_dim))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MLPClassifer(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None, num_layers=0,
        dropout=0.5, hidden_act='relu', batch_norm=True, **kwargs):
        super().__init__()
        activation = ACT2FN[hidden_act]
        self.layers = nn.ModuleList()
        if hidden_dim is None:
            hidden_dim = in_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LogitClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        text_non_linear_dim = kwargs['text_hidden_dim']
        image_non_linear_dim = kwargs['img_hidden_dim']
        self.f_o_text = ReLUWithWeightNormFC(input_dim, text_non_linear_dim)
        self.f_o_image = ReLUWithWeightNormFC(input_dim, image_non_linear_dim)
        self.linear_text = nn.Linear(text_non_linear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_non_linear_dim, num_ans_candidates)
        if 'pretrained_image' in kwargs and kwargs['pretrained_text'
            ] is not None:
            self.linear_text.weight.data.copy_(torch.from_numpy(kwargs[
                'pretrained_text']))
        if 'pretrained_image' in kwargs and kwargs['pretrained_image'
            ] is not None:
            self.linear_image.weight.data.copy_(torch.from_numpy(kwargs[
                'pretrained_image']))

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val
        return logit_value


class WeightNormClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super().__init__()
        layers = [weight_norm(nn.Linear(in_dim, hidden_dim), dim=None), nn.
            ReLU(), nn.Dropout(dropout, inplace=True), weight_norm(nn.
            Linear(hidden_dim, out_dim), dim=None)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ModalCombineLayer(nn.Module):

    def __init__(self, combine_type, img_feat_dim, txt_emb_dim, **kwargs):
        super().__init__()
        if combine_type == 'MFH':
            self.module = MFH(img_feat_dim, txt_emb_dim, **kwargs)
        elif combine_type == 'non_linear_element_multiply':
            self.module = NonLinearElementMultiply(img_feat_dim,
                txt_emb_dim, **kwargs)
        elif combine_type == 'two_layer_element_multiply':
            self.module = TwoLayerElementMultiply(img_feat_dim, txt_emb_dim,
                **kwargs)
        elif combine_type == 'top_down_attention_lstm':
            self.module = TopDownAttentionLSTM(img_feat_dim, txt_emb_dim,
                **kwargs)
        else:
            raise NotImplementedError('Not implemented combine type: %s' %
                combine_type)
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class MfbExpand(nn.Module):

    def __init__(self, img_feat_dim, txt_emb_dim, hidden_dim, dropout):
        super().__init__()
        self.lc_image = nn.Linear(in_features=img_feat_dim, out_features=
            hidden_dim)
        self.lc_ques = nn.Linear(in_features=txt_emb_dim, out_features=
            hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feat, question_embed):
        image1 = self.lc_image(image_feat)
        ques1 = self.lc_ques(question_embed)
        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            ques1_expand = torch.unsqueeze(ques1, 1).expand(-1,
                num_location, -1)
        else:
            ques1_expand = ques1
        joint_feature = image1 * ques1_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class MFH(nn.Module):

    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super().__init__()
        self.mfb_expand_list = nn.ModuleList()
        self.mfb_sqz_list = nn.ModuleList()
        self.relu = nn.ReLU()
        hidden_sizes = kwargs['hidden_sizes']
        self.out_dim = int(sum(hidden_sizes) / kwargs['pool_size'])
        self.order = kwargs['order']
        self.pool_size = kwargs['pool_size']
        for i in range(self.order):
            mfb_exp_i = MfbExpand(img_feat_dim=image_feat_dim, txt_emb_dim=
                ques_emb_dim, hidden_dim=hidden_sizes[i], dropout=kwargs[
                'dropout'])
            self.mfb_expand_list.append(mfb_exp_i)
            self.mfb_sqz_list.append(self.mfb_squeeze)

    def forward(self, image_feat, question_embedding):
        feature_list = []
        prev_mfb_exp = 1
        for i in range(self.order):
            mfb_exp = self.mfb_expand_list[i]
            mfb_sqz = self.mfb_sqz_list[i]
            z_exp_i = mfb_exp(image_feat, question_embedding)
            if i > 0:
                z_exp_i = prev_mfb_exp * z_exp_i
            prev_mfb_exp = z_exp_i
            z = mfb_sqz(z_exp_i)
            feature_list.append(z)
        cat_dim = len(feature_list[0].size()) - 1
        feature = torch.cat(feature_list, dim=cat_dim)
        return feature

    def mfb_squeeze(self, joint_feature):
        orig_feature_size = len(joint_feature.size())
        if orig_feature_size == 2:
            joint_feature = torch.unsqueeze(joint_feature, dim=1)
        batch_size, num_loc, dim = joint_feature.size()
        if dim % self.pool_size != 0:
            exit('the dim %d is not multiply of              pool_size %d' %
                (dim, self.pool_size))
        joint_feature_reshape = joint_feature.view(batch_size, num_loc, int
            (dim / self.pool_size), self.pool_size)
        iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)
        iatt_iq_sqrt = torch.sqrt(self.relu(iatt_iq_sumpool)) - torch.sqrt(self
            .relu(-iatt_iq_sumpool))
        iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)
        iatt_iq_l2 = nn.functional.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(batch_size, num_loc, int(dim / self.
            pool_size))
        if orig_feature_size == 2:
            iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)
        return iatt_iq_l2


class NonLinearElementMultiply(nn.Module):

    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super().__init__()
        self.fa_image = ReLUWithWeightNormFC(image_feat_dim, kwargs[
            'hidden_dim'])
        self.fa_txt = ReLUWithWeightNormFC(ques_emb_dim, kwargs['hidden_dim'])
        context_dim = kwargs.get('context_dim', None)
        if context_dim is not None:
            self.fa_context = ReLUWithWeightNormFC(context_dim, kwargs[
                'hidden_dim'])
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.out_dim = kwargs['hidden_dim']

    def forward(self, image_feat, question_embedding, context_embedding=None):
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)
        if len(image_feat.size()) == 3 and len(question_fa.size()) != 3:
            question_fa_expand = question_fa.unsqueeze(1)
        else:
            question_fa_expand = question_fa
        joint_feature = image_fa * question_fa_expand
        if context_embedding is not None:
            context_fa = self.fa_context(context_embedding)
            context_text_joint_feaure = context_fa * question_fa_expand
            joint_feature = torch.cat([joint_feature,
                context_text_joint_feaure], dim=1)
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class TopDownAttentionLSTM(nn.Module):

    def __init__(self, image_feat_dim, embed_dim, **kwargs):
        super().__init__()
        self.fa_image = weight_norm(nn.Linear(image_feat_dim, kwargs[
            'attention_dim']))
        self.fa_hidden = weight_norm(nn.Linear(kwargs['hidden_dim'], kwargs
            ['attention_dim']))
        self.top_down_lstm = nn.LSTMCell(embed_dim + image_feat_dim +
            kwargs['hidden_dim'], kwargs['hidden_dim'], bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.out_dim = kwargs['attention_dim']

    def forward(self, image_feat, embedding):
        image_feat_mean = image_feat.mean(1)
        state = registry.get(f'{image_feat.device}_lstm_state')
        h1, c1 = state['td_hidden']
        h2, c2 = state['lm_hidden']
        h1, c1 = self.top_down_lstm(torch.cat([h2, image_feat_mean,
            embedding], dim=1), (h1, c1))
        state['td_hidden'] = h1, c1
        image_fa = self.fa_image(image_feat)
        hidden_fa = self.fa_hidden(h1)
        joint_feature = self.relu(image_fa + hidden_fa.unsqueeze(1))
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class TwoLayerElementMultiply(nn.Module):

    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super().__init__()
        self.fa_image1 = ReLUWithWeightNormFC(image_feat_dim, kwargs[
            'hidden_dim'])
        self.fa_image2 = ReLUWithWeightNormFC(kwargs['hidden_dim'], kwargs[
            'hidden_dim'])
        self.fa_txt1 = ReLUWithWeightNormFC(ques_emb_dim, kwargs['hidden_dim'])
        self.fa_txt2 = ReLUWithWeightNormFC(kwargs['hidden_dim'], kwargs[
            'hidden_dim'])
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.out_dim = kwargs['hidden_dim']

    def forward(self, image_feat, question_embedding):
        image_fa = self.fa_image2(self.fa_image1(image_feat))
        question_fa = self.fa_txt2(self.fa_txt1(question_embedding))
        if len(image_feat.size()) == 3:
            num_location = image_feat.size(1)
            question_fa_expand = torch.unsqueeze(question_fa, 1).expand(-1,
                num_location, -1)
        else:
            question_fa_expand = question_fa
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


class TransformLayer(nn.Module):

    def __init__(self, transform_type, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if transform_type == 'linear':
            self.module = LinearTransform(in_dim, out_dim)
        elif transform_type == 'conv':
            self.module = ConvTransform(in_dim, out_dim, hidden_dim)
        else:
            raise NotImplementedError(
                'Unknown post combine transform type: %s' % transform_type)
        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class LinearTransform(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lc = weight_norm(nn.Linear(in_features=in_dim, out_features=
            out_dim), dim=None)
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)


class ConvTransform(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim,
            kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim,
            kernel_size=1)
        self.out_dim = out_dim

    def forward(self, x):
        if len(x.size()) == 3:
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)
        iatt_conv1 = self.conv1(x_reshape)
        iatt_relu = nn.functional.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)
        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)
        return iatt_conv3


class BCNet(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=None,
        k=3):
        super().__init__()
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        if dropout is None:
            dropout = [0.2, 0.5]
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0]
            )
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0]
            )
        self.dropout = nn.Dropout(dropout[1])
        if k > 1:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        if h_out is None:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim *
                self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None
                )

    def forward(self, v, q):
        if self.h_out is None:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = d_.transpose(1, 2).transpose(2, 3)
            return logits
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))
            logits = logits + self.h_bias
            return logits
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            return logits.transpose(2, 3).transpose(1, 2)

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)
        logits = logits.squeeze(3).squeeze(2)
        if self.k > 1:
            logits = logits.unsqueeze(1)
            logits = self.p_net(logits).squeeze(1) * self.k
        return logits


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act is not None:
                layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act is not None:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiAttention(nn.Module):

    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.2, 0.5]
        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse,
            dropout=dropout, k=3), name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)
        if v_mask:
            v_abs_sum = v.abs().sum(2)
            mask = (v_abs_sum == 0).unsqueeze(1).unsqueeze(3)
            mask = mask.expand(logits.size())
            logits.masked_fill_(mask, -float('inf'))
        expanded_logits = logits.view(-1, self.glimpse, v_num * q_num)
        p = nn.functional.softmax(expanded_logits, 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_config`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (ListConfig): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instanttions of each loss
                                   passed in config
    """

    def __init__(self, loss_list):
        super().__init__()
        self.losses = []
        self._evaluation_predict = registry.get('config').evaluation.predict
        for loss in loss_list:
            self.losses.append(MMFLoss(loss))

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if not hasattr(sample_list, 'targets'):
            if not self._evaluation_predict:
                warnings.warn(
                    "Sample list has not field 'targets', are you sure that your ImDB has labels? you may have wanted to run with evaluation.predict=true"
                    )
            return output
        for loss in self.losses:
            output.update(loss(sample_list, model_output, *args, **kwargs))
        registry_loss_key = '{}.{}.{}'.format('losses', sample_list.
            dataset_name, sample_list.dataset_type)
        registry.register(registry_loss_key, output)
        return output


class MMFLoss(nn.Module):
    """Internal MMF helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/vqa2/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set of dataset `vqa2`.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``MMFLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.writer = registry.get('writer')
        is_mapping = isinstance(params, collections.abc.MutableMapping)
        if is_mapping:
            if 'type' not in params:
                raise ValueError(
                    "Parameters to loss must have 'type' field tospecify type of loss to instantiate"
                    )
            else:
                loss_name = params['type']
        else:
            assert isinstance(params, str
                ), "loss must be a string or dictionary with 'type' key"
            loss_name = params
        self.name = loss_name
        loss_class = registry.get_loss_class(loss_name)
        if loss_class is None:
            raise ValueError(
                f'No loss named {loss_name} is registered to registry')
        if loss_name == 'multi':
            assert is_mapping
            self.loss_criterion = loss_class(params)
        else:
            if is_mapping:
                loss_params = params.get('params', {})
            else:
                loss_params = {}
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = self.loss_criterion(sample_list, model_output, *args, **kwargs)
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)
        if loss.dim() == 0:
            loss = loss.view(1)
        key = '{}/{}/{}'.format(sample_list.dataset_type, sample_list.
            dataset_name, self.name)
        return {key: loss}


@registry.register_loss('logit_bce')
class LogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output['scores']
        targets = sample_list['targets']
        loss = F.binary_cross_entropy_with_logits(scores, targets,
            reduction='mean')
        return loss * targets.size(1)


@registry.register_loss('bce')
class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output['scores']
        targets = sample_list['targets']
        loss = F.binary_cross_entropy(scores, targets, reduction='mean')
        return loss * targets.size(1)


@registry.register_loss('caption_cross_entropy')
class CaptionCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output['scores']
        targets = sample_list['targets']
        if hasattr(sample_list, 'caption_len'):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0,
                descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= '1.1':
            scores = pack_padded_sequence(scores, decode_lengths,
                batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths,
                batch_first=True).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths,
                batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths,
                batch_first=True)
        loss = F.cross_entropy(scores, targets)
        return loss


@registry.register_loss('nll_loss')
class NLLLoss(nn.Module):
    """Negative log likelikehood loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the negative log likelihood.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output['scores']
        targets = sample_list['targets']
        _, idx = targets.max(dim=1)
        loss = F.nll_loss(scores, idx, reduction='mean')
        return loss * targets.size(1)


@registry.register_loss('multi')
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        params (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: logit_bce
            weight: 0.3
            params: {}
          - type: attention_supervision
            weight: 0.7
            params: {}

    """

    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []
        self.writer = registry.get('writer')
        self.loss_names = []
        for loss_params in params['params']:
            self.loss_names.append(loss_params['type'])
            loss_fn = MMFLoss(loss_params)
            loss_weight = loss_params.get('weight', {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            loss += self.losses_weights[idx] * value
        return loss


@registry.register_loss('attention_supervision')
class AttentionSupervisionLoss(nn.Module):
    """Loss for attention supervision. Used in case you want to make attentions
    similar to some particular values.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = (lambda *args, **kwargs: nn.functional.
            binary_cross_entropy(*args, **kwargs))

    def forward(self, sample_list, model_output):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        context_attentions = model_output['attentions']
        attention_supervision = sample_list['info']['attention_supervision']
        loss = self.loss_fn(context_attentions[0], attention_supervision.
            float(), weight=attention_supervision.float())
        return loss * attention_supervision.size(1)


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)
    return torch.sum(res, dim=1, keepdim=True)


@registry.register_loss('weighted_softmax')
class WeightedSoftmaxLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output['scores']
        target_score = sample_list['targets']
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1e-06)
        tar = target_score / tar_sum
        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss('softmax_kldiv')
class SoftmaxKlDivLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output['scores']
        target_score = sample_list['targets']
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1e-06)
        tar = target_score / tar_sum
        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss('wrong')
class WrongLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output['scores']
        target_score = sample_list['targets']
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1e-06)
        tar = target_score / tar_sum
        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, reduction='mean')
        loss *= target_score.size(1)
        return loss


@registry.register_loss('bce_kl_combined')
class CombinedLoss(nn.Module):

    def __init__(self, weight_softmax):
        super().__init__()
        self.weight_softmax = weight_softmax

    def forward(self, sample_list, model_output):
        pred_score = model_output['scores']
        target_score = sample_list['targets']
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1e-06)
        tar = target_score / tar_sum
        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)
        loss2 = F.binary_cross_entropy_with_logits(pred_score, target_score,
            reduction='mean')
        loss2 *= target_score.size(1)
        loss = self.weight_softmax * loss1 + loss2
        return loss


@registry.register_loss('m4c_decoding_bce_with_mask')
class M4CDecodingBCEWithMaskLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.0])

    def forward(self, sample_list, model_output):
        scores = model_output['scores']
        targets = sample_list['targets']
        loss_mask = sample_list['train_loss_mask']
        assert scores.dim() == 3 and loss_mask.dim() == 2
        losses = F.binary_cross_entropy_with_logits(scores, targets,
            reduction='none')
        losses *= loss_mask.unsqueeze(-1)
        count = torch.max(torch.sum(loss_mask), self.one)
        loss = torch.sum(losses) / count
        return loss


@registry.register_loss('cross_entropy')
class CrossEntropyLoss(nn.Module):

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, sample_list, model_output):
        return self.loss_fn(model_output['scores'], sample_list.targets)


class OnlyBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.base_test = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.
            nn.Tanh(), torch.nn.Linear(4, 5))

    def format_state_key(self, key):
        return key


class TestDecoderModel(nn.Module):

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

    def build(self):
        return

    def init_hidden_state(self, features):
        h = features.new_zeros((features.size(0), self.config.classifier.
            params.hidden_dim), dtype=torch.float)
        c = features.new_zeros((features.size(0), self.config.classifier.
            params.hidden_dim), dtype=torch.float)
        return h, c

    def get_data_t(self, data, batch_size_t):
        data['texts'] = data['texts'][:batch_size_t]
        if 'state' in data:
            h1 = data['state']['td_hidden'][0][:batch_size_t]
            c1 = data['state']['td_hidden'][1][:batch_size_t]
            h2 = data['state']['lm_hidden'][0][:batch_size_t]
            c2 = data['state']['lm_hidden'][1][:batch_size_t]
        else:
            h1, c1 = self.init_hidden_state(data['texts'])
            h2, c2 = self.init_hidden_state(data['texts'])
        data['state'] = {'td_hidden': (h1, c1), 'lm_hidden': (h2, c2)}
        registry.register(f'{h1.device}_lstm_state', data['state'])
        return data, batch_size_t

    def forward(self, sample_list):
        scores = torch.rand(sample_list.get_batch_size(), 3127)
        decoder = registry.get_decoder_class(self.config.inference.type)(self
            .vocab, self.config)
        sample_list = decoder.init_batch(sample_list)
        batch_size = sample_list.image_feature_0.size(0)
        data = {}
        data['texts'] = sample_list.answers.new_full((batch_size, 1), self.
            vocab.SOS_INDEX, dtype=torch.long)
        timesteps = 10
        sample_list.add_field('targets', sample_list.answers[:, (0), 1:])
        output = None
        batch_size_t = batch_size
        for t in range(timesteps):
            data, batch_size_t = self.get_data_t(data, batch_size_t)
            output = torch.randn(1, 9491)
            if t == timesteps - 1:
                output = torch.ones(1, 9491) * -30
                output[0][2] = 10
            finish, data, batch_size_t = decoder.decode(t, data, output)
            if finish:
                break
        model_output = {'scores': scores}
        model_output['captions'] = decoder.get_result()
        return model_output


RESNET152_MODEL = models.resnet152(pretrained=True)


class ResNet152FeatModule(nn.Module):

    def __init__(self):
        super().__init__()
        modules = list(RESNET152_MODEL.children())[:-2]
        self.feature_module = nn.Sequential(*modules)

    def forward(self, x):
        return self.feature_module(x)

