import sys
_module = sys.modules[__name__]
del sys
base_demo = _module
base_english_demo = _module
benchmark_bm25 = _module
benchmark_sbert = _module
fast_sim_demo = _module
image_demo = _module
literal_sim_demo = _module
one_line_demo = _module
search_gradio_demo = _module
sim_gradio_demo = _module
setup = _module
similarities = _module
clip_model = _module
data_loader = _module
evaluation = _module
fastsim = _module
imagesim = _module
literalsim = _module
similarity = _module
utils = _module
distance = _module
get_file = _module
imagehash = _module
ngram_util = _module
rank_bm25 = _module
tfidf = _module
tokenizer = _module
util = _module
version = _module
test_fastsim = _module
test_image_qps = _module
test_imagesim = _module
test_literalsim = _module
test_sim_score = _module
test_text_qps = _module

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


from typing import List


from typing import Union


import numpy as np


import torch


import torch.nn.functional


from torch import nn


import math


from typing import Dict


import queue


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CLIPModel(nn.Module):

    def __init__(self, model_name: str='openai/clip-vit-base-patch32', processor_name=None):
        super(CLIPModel, self).__init__()
        if processor_name is None:
            processor_name = model_name
        self.model = transformers.CLIPModel.from_pretrained(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(processor_name)

    def __str__(self):
        return f'CLIPModel({self.model})'

    def forward(self, features):
        image_embeds = []
        text_embeds = []
        if 'pixel_values' in features:
            vision_outputs = self.model.vision_model(pixel_values=features['pixel_values'])
            image_embeds = self.model.visual_projection(vision_outputs[1])
        if 'input_ids' in features:
            text_outputs = self.model.text_model(input_ids=features.get('input_ids'), attention_mask=features.get('attention_mask', None), position_ids=features.get('position_ids', None), output_attentions=features.get('output_attentions', None), output_hidden_states=features.get('output_hidden_states', None))
            text_embeds = self.model.text_projection(text_outputs[1])
        sentence_embedding = []
        image_features = iter(image_embeds)
        text_features = iter(text_embeds)
        for idx, input_type in enumerate(features['image_text_info']):
            if input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))
        features['embedding'] = torch.stack(sentence_embedding).float()
        return features

    def tokenize(self, texts):
        images = []
        texts_values = []
        image_text_info = []
        for idx, data in enumerate(texts):
            if isinstance(data, Image.Image):
                images.append(data)
                image_text_info.append(0)
            else:
                texts_values.append(data)
                image_text_info.append(1)
        if len(texts_values) == 0:
            texts_values = None
        if len(images) == 0:
            images = None
        inputs = self.processor(text=texts_values, images=images, return_tensors='pt', padding=True)
        inputs['image_text_info'] = image_text_info
        return inputs

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str):
        return CLIPModel(model_name=input_path)

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])

    @staticmethod
    def batch_to_device(batch):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key]
        return batch

    def encode(self, sentences: Union[str, List[str]], batch_size: int=32, show_progress_bar: bool=False, convert_to_numpy: bool=True, normalize_embeddings: bool=False):
        """
        Computes sentence and images embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
            In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        self.model.eval()
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True
        self.model
        all_embeddings = []
        length_sorted_idx = np.argsort([(-self._text_length(sent)) for sent in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc='Batches', disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = self.batch_to_device(features)
            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features['embedding']
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        else:
            all_embeddings = torch.stack(all_embeddings)
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

