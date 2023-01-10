import sys
_module = sys.modules[__name__]
del sys
default_runtime = _module
ctw1500 = _module
icdar2015 = _module
icdar2017 = _module
synthtext = _module
toy_data = _module
dbnet_r18_fpnc = _module
dbnet_r50dcnv2_fpnc = _module
dbnetpp_r50dcnv2_fpnc = _module
drrg_r50_fpn_unet = _module
fcenet_r50_fpn = _module
fcenet_r50dcnv2_fpn = _module
ocr_mask_rcnn_r50_fpn_ohem = _module
ocr_mask_rcnn_r50_fpn_ohem_poly = _module
panet_r18_fpem_ffm = _module
panet_r50_fpem_ffm = _module
psenet_r50_fpnf = _module
textsnake_r50_fpn_unet = _module
dbnet_pipeline = _module
drrg_pipeline = _module
fcenet_pipeline = _module
maskrcnn_pipeline = _module
panet_pipeline = _module
psenet_pipeline = _module
textsnake_pipeline = _module
MJ_train = _module
ST_MJ_alphanumeric_train = _module
ST_MJ_train = _module
ST_SA_MJ_real_train = _module
ST_SA_MJ_train = _module
ST_charbox_train = _module
academic_test = _module
seg_toy_data = _module
abinet = _module
crnn = _module
crnn_tps = _module
master = _module
nrtr_modality_transform = _module
robust_scanner = _module
sar = _module
satrn = _module
seg = _module
abinet_pipeline = _module
crnn_pipeline = _module
crnn_tps_pipeline = _module
master_pipeline = _module
nrtr_pipeline = _module
sar_pipeline = _module
satrn_pipeline = _module
seg_pipeline = _module
schedule_adadelta_18e = _module
schedule_adadelta_5e = _module
schedule_adam_600e = _module
schedule_adam_step_12e = _module
schedule_adam_step_20e = _module
schedule_adam_step_5e = _module
schedule_adam_step_600e = _module
schedule_adam_step_6e = _module
schedule_sgd_100k_iters = _module
schedule_sgd_1200e = _module
schedule_sgd_1500e = _module
schedule_sgd_160e = _module
schedule_sgd_600e = _module
sdmgr_novisual_60e_wildreceipt = _module
sdmgr_novisual_60e_wildreceipt_openset = _module
sdmgr_unet16_60e_wildreceipt = _module
bert_softmax_cluener_18e = _module
dbnet_r18_fpnc_100k_iters_synthtext = _module
dbnet_r18_fpnc_1200e_icdar2015 = _module
dbnet_r50dcnv2_fpnc_100k_iters_synthtext = _module
dbnet_r50dcnv2_fpnc_1200e_icdar2015 = _module
dbnetpp_r50dcnv2_fpnc_100k_iter_synthtext = _module
dbnetpp_r50dcnv2_fpnc_1200e_icdar2015 = _module
drrg_r50_fpn_unet_1200e_ctw1500 = _module
fcenet_r50_fpn_1500e_icdar2015 = _module
fcenet_r50dcnv2_fpn_1500e_ctw1500 = _module
mask_rcnn_r50_fpn_160e_ctw1500 = _module
mask_rcnn_r50_fpn_160e_icdar2015 = _module
mask_rcnn_r50_fpn_160e_icdar2017 = _module
panet_r18_fpem_ffm_600e_ctw1500 = _module
panet_r18_fpem_ffm_600e_icdar2015 = _module
panet_r50_fpem_ffm_600e_icdar2017 = _module
psenet_r50_fpnf_600e_ctw1500 = _module
psenet_r50_fpnf_600e_icdar2015 = _module
psenet_r50_fpnf_600e_icdar2017 = _module
textsnake_r50_fpn_unet_1200e_ctw1500 = _module
abinet_academic = _module
abinet_vision_only_academic = _module
crnn_academic_dataset = _module
crnn_toy_dataset = _module
master_r31_12e_ST_MJ_SA = _module
master_toy_dataset = _module
nrtr_modality_transform_academic = _module
nrtr_modality_transform_toy_dataset = _module
nrtr_r31_1by16_1by8_academic = _module
nrtr_r31_1by8_1by4_academic = _module
robustscanner_r31_academic = _module
sar_r31_parallel_decoder_academic = _module
sar_r31_parallel_decoder_chinese = _module
sar_r31_parallel_decoder_toy_dataset = _module
sar_r31_sequential_decoder_academic = _module
satrn_academic = _module
satrn_small = _module
seg_r31_1by16_fpnocr_academic = _module
seg_r31_1by16_fpnocr_toy_dataset = _module
crnn_tps_academic_dataset = _module
ner_demo = _module
webcam_demo = _module
conf = _module
stats = _module
mmocr = _module
apis = _module
inference = _module
test = _module
train = _module
utils = _module
core = _module
deployment = _module
deploy_utils = _module
evaluation = _module
hmean = _module
hmean_ic13 = _module
hmean_iou = _module
kie_metric = _module
ner_metric = _module
ocr_metric = _module
mask = _module
visualize = _module
datasets = _module
base_dataset = _module
builder = _module
icdar_dataset = _module
kie_dataset = _module
ner_dataset = _module
ocr_dataset = _module
ocr_seg_dataset = _module
openset_kie_dataset = _module
pipelines = _module
box_utils = _module
crop = _module
custom_format_bundle = _module
dbnet_transforms = _module
kie_transforms = _module
loading = _module
ner_transforms = _module
ocr_seg_targets = _module
ocr_transforms = _module
test_time_aug = _module
textdet_targets = _module
base_textdet_targets = _module
dbnet_targets = _module
drrg_targets = _module
fcenet_targets = _module
panet_targets = _module
psenet_targets = _module
textsnake_targets = _module
transform_wrappers = _module
transforms = _module
text_det_dataset = _module
uniform_concat_dataset = _module
backend = _module
loader = _module
parser = _module
models = _module
builder = _module
common = _module
backbones = _module
unet = _module
detectors = _module
single_stage = _module
layers = _module
transformer_layers = _module
losses = _module
dice_loss = _module
focal_loss = _module
modules = _module
transformer_module = _module
kie = _module
extractors = _module
sdmgr = _module
heads = _module
sdmgr_head = _module
sdmgr_loss = _module
ner = _module
classifiers = _module
ner_classifier = _module
convertors = _module
ner_convertor = _module
decoders = _module
fc_decoder = _module
encoders = _module
bert_encoder = _module
masked_cross_entropy_loss = _module
masked_focal_loss = _module
activations = _module
bert = _module
textdet = _module
dense_heads = _module
db_head = _module
drrg_head = _module
fce_head = _module
head_mixin = _module
pan_head = _module
pse_head = _module
textsnake_head = _module
dbnet = _module
drrg = _module
fcenet = _module
ocr_mask_rcnn = _module
panet = _module
psenet = _module
single_stage_text_detector = _module
text_detector_mixin = _module
textsnake = _module
db_loss = _module
drrg_loss = _module
fce_loss = _module
pan_loss = _module
pse_loss = _module
textsnake_loss = _module
gcn = _module
local_graph = _module
proposal_local_graph = _module
necks = _module
fpem_ffm = _module
fpn_cat = _module
fpn_unet = _module
fpnf = _module
postprocess = _module
base_postprocessor = _module
db_postprocessor = _module
drrg_postprocessor = _module
fce_postprocessor = _module
pan_postprocessor = _module
pse_postprocessor = _module
textsnake_postprocessor = _module
textrecog = _module
nrtr_modality_transformer = _module
resnet = _module
resnet31_ocr = _module
resnet_abi = _module
shallow_cnn = _module
very_deep_vgg = _module
abi = _module
attn = _module
base = _module
ctc = _module
seg = _module
abinet_language_decoder = _module
abinet_vision_decoder = _module
base_decoder = _module
crnn_decoder = _module
master_decoder = _module
nrtr_decoder = _module
position_attention_decoder = _module
robust_scanner_decoder = _module
sar_decoder = _module
sar_decoder_with_bs = _module
sequence_attention_decoder = _module
abinet_vision_model = _module
base_encoder = _module
channel_reduction_encoder = _module
nrtr_encoder = _module
sar_encoder = _module
satrn_encoder = _module
transformer = _module
fusers = _module
abi_fuser = _module
seg_head = _module
conv_layer = _module
dot_product_attention_layer = _module
lstm_layer = _module
position_aware_layer = _module
robust_scanner_fusion_layer = _module
satrn_layers = _module
ce_loss = _module
ctc_loss = _module
mix_loss = _module
seg_loss = _module
fpn_ocr = _module
plugins = _module
common = _module
preprocessor = _module
base_preprocessor = _module
tps_preprocessor = _module
recognizer = _module
abinet = _module
base = _module
encode_decode_recognizer = _module
nrtr = _module
seg_recognizer = _module
box_util = _module
check_argument = _module
collect_env = _module
data_convert_util = _module
fileio = _module
img_util = _module
lmdb_util = _module
logger = _module
model = _module
ocr = _module
setup_env = _module
string_util = _module
version = _module
setup = _module
test_image_misc = _module
test_model_inference = _module
test_single_gpu_test = _module
test_utils = _module
test_deploy_utils = _module
test_end2end_vis = _module
test_base_dataset = _module
test_crop = _module
test_dbnet_transforms = _module
test_detect_dataset = _module
test_icdar_dataset = _module
test_kie_dataset = _module
test_loader = _module
test_loading = _module
test_ner_dataset = _module
test_ocr_dataset = _module
test_ocr_seg_dataset = _module
test_ocr_seg_target = _module
test_ocr_transforms = _module
test_openset_kie_dataset = _module
test_parser = _module
test_test_time_aug = _module
test_textdet_targets = _module
test_transform_wrappers = _module
test_transforms = _module
test_uniform_concat_dataset = _module
test_eval_utils = _module
test_hmean_detect = _module
test_hmean_ic13 = _module
test_hmean_iou = _module
test_detector = _module
test_kie_config = _module
test_attn_label_convertor = _module
test_base_label_convertor = _module
test_ctc_label_convertor = _module
test_loss = _module
test_modules = _module
test_ner_model = _module
test_ocr_backbone = _module
test_ocr_decoder = _module
test_ocr_encoder = _module
test_ocr_fuser = _module
test_ocr_head = _module
test_ocr_layer = _module
test_ocr_loss = _module
test_ocr_neck = _module
test_ocr_preprocessor = _module
test_panhead = _module
test_recog_config = _module
test_recognizer = _module
test_targets = _module
test_textdet_head = _module
test_textdet_neck = _module
test_data_converter = _module
test_box = _module
test_check_argument = _module
test_mask_utils = _module
test_model = _module
test_ocr = _module
test_setup_env = _module
test_string_util = _module
test_text_utils = _module
test_textio = _module
test_version_utils = _module
test_wrapper = _module
analyze_logs = _module
benchmark_processing = _module
curvedsyntext_converter = _module
extract_kaist = _module
labelme_converter = _module
closeset_to_openset = _module
art_converter = _module
bid_converter = _module
coco_to_line_dict = _module
cocotext_converter = _module
ctw1500_converter = _module
detext_converter = _module
funsd_converter = _module
hiertext_converter = _module
ic11_converter = _module
ic13_converter = _module
icdar_converter = _module
ilst_converter = _module
imgur_converter = _module
kaist_converter = _module
lsvt_converter = _module
lv_converter = _module
mtwi_converter = _module
naf_converter = _module
rctw_converter = _module
rects_converter = _module
sroie_converter = _module
synthtext_converter = _module
textocr_converter = _module
totaltext_converter = _module
vintext_converter = _module
openvino_converter = _module
seg_synthtext_converter = _module
svt_converter = _module
lmdb_converter = _module
deploy_test = _module
mmocr2torchserve = _module
mmocr_handler = _module
onnx2tensorrt = _module
pytorch2onnx = _module
test_torchserve = _module
det_test_imgs = _module
kie_test_imgs = _module
print_config = _module
publish_model = _module
recog_test_imgs = _module
test = _module
train = _module

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


import warnings


import numpy as np


import torch.distributed as dist


import copy


from typing import Any


from typing import Iterable


import math


from matplotlib import pyplot as plt


from torch.utils.data import Dataset


import torchvision.transforms.functional as TF


import torchvision.transforms as transforms


import torch.nn as nn


import torch.utils.checkpoint as cp


import torch.nn.functional as F


from torch import nn


from torch.nn import functional as F


from torch.nn import CrossEntropyLoss


import itertools


from torch.nn import init


from queue import PriorityQueue


from abc import ABCMeta


from abc import abstractmethod


from collections import OrderedDict


import torch.multiprocessing as mp


from numpy.testing import assert_array_equal


from functools import partial


import random


import time


def build_upsample_layer(cfg, *args, **kwargs):
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
                deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in UPSAMPLE_LAYERS:
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    else:
        upsample = UPSAMPLE_LAYERS.get(layer_type)
    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), upsample_cfg=dict(type='InterpConv'), dcn=None, plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.conv_block = conv_block(in_channels=2 * skip_channels, out_channels=out_channels, num_convs=num_convs, stride=stride, dilation=dilation, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, dcn=None, plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(cfg=upsample_cfg, in_channels=in_channels, out_channels=skip_channels, with_cp=with_cp, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(in_channels, skip_channels, kernel_size=1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self, in_channels, out_channels, num_convs=2, stride=1, dilation=1, with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), dcn=None, plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(ConvModule(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, stride=stride if i == 0 else 1, dilation=1 if i == 0 else dilation, padding=1 if i == 0 else dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)


class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, kernel_size=4, scale_factor=2):
        super().__init__()
        assert kernel_size - scale_factor >= 0 and (kernel_size - scale_factor) % 2 == 0, f'kernel_size should be greater than or equal to scale_factor and (kernel_size - scale_factor) should be even numbers, while the kernel size is {kernel_size} and scale_factor is {scale_factor}.'
        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        _, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(self, in_channels, out_channels, with_cp=False, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), *, conv_cfg=None, conv_first=False, kernel_size=1, stride=1, padding=0, upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=False)):
        super().__init__()
        self.with_cp = with_cp
        conv = ConvModule(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    Args:
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
    """

    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v
        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)
        self.attention = ScaledDotProductAttention(d_k ** 0.5, dropout)
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
        attn_out, _ = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward module.

    Args:
        d_in (int): The dimension of the input for feedforward
            network model.
        d_hid (int): The dimension of the feedforward
            network model.
        dropout (float): Dropout layer on feedforward output.
        act_cfg (dict): Activation cfg for feedforward module.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, act_cfg=dict(type='Relu')):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = build_activation_layer(act_cfg)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class TFDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'enc_dec_attn',
            'norm', 'ffn', 'norm') or ('norm', 'self_attn', 'norm',
            'enc_dec_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self, d_model=512, d_inner=256, n_head=8, d_k=64, d_v=64, dropout=0.1, qkv_bias=False, act_cfg=dict(type='mmcv.GELU'), operation_order=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)
        self.mlp = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = 'norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'
        assert self.operation_order in [('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'), ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')]

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm'):
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input, self_attn_mask)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output, enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)
            mlp_out = self.mlp(enc_dec_attn_out)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out)
        elif self.operation_order == ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'):
            dec_input_norm = self.norm1(dec_input)
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm, dec_input_norm, self_attn_mask)
            dec_attn_out += dec_input
            enc_dec_attn_in = self.norm2(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output, enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            mlp_out += enc_dec_attn_out
        return mlp_out


class DiceLoss(nn.Module):

    def __init__(self, eps=1e-06):
        super().__init__()
        assert isinstance(eps, float)
        self.eps = eps

    def forward(self, pred, target, mask=None):
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        if mask is not None:
            mask = mask.contiguous().view(mask.size()[0], -1)
            pred = pred * mask
            target = target * mask
        a = torch.sum(pred * target)
        b = torch.sum(pred)
        c = torch.sum(target)
        d = 2 * a / (b + c + self.eps)
        return 1 - d


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation.

    Args:
        gamma (float): The larger the gamma, the smaller
            the loss weight of easier samples.
        weight (float): A manual rescaling weight given to each
            class.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logit = F.log_softmax(input, dim=1)
        pt = torch.exp(logit)
        logit = (1 - pt) ** self.gamma * logit
        loss = F.nll_loss(logit, target, self.weight, ignore_index=self.ignore_index)
        return loss


class PositionalEncoding(nn.Module):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('position_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([(1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        self.device = x.device
        x = x + self.position_table[:, :x.size(1)].clone().detach()
        return self.dropout(x)


class GNNLayer(nn.Module):

    def __init__(self, node_dim=256, edge_dim=256):
        super().__init__()
        self.in_fc = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.coef_fc = nn.Linear(node_dim, 1)
        self.out_fc = nn.Linear(node_dim, node_dim)
        self.relu = nn.ReLU()

    def forward(self, nodes, edges, nums):
        start, cat_nodes = 0, []
        for num in nums:
            sample_nodes = nodes[start:start + num]
            cat_nodes.append(torch.cat([sample_nodes.unsqueeze(1).expand(-1, num, -1), sample_nodes.unsqueeze(0).expand(num, -1, -1)], -1).view(num ** 2, -1))
            start += num
        cat_nodes = torch.cat([torch.cat(cat_nodes), edges], -1)
        cat_nodes = self.relu(self.in_fc(cat_nodes))
        coefs = self.coef_fc(cat_nodes)
        start, residuals = 0, []
        for num in nums:
            residual = F.softmax(-torch.eye(num).unsqueeze(-1) * 1000000000.0 + coefs[start:start + num ** 2].view(num, num, -1), 1)
            residuals.append((residual * cat_nodes[start:start + num ** 2].view(num, num, -1)).sum(1))
            start += num ** 2
        nodes += self.relu(self.out_fc(torch.cat(residuals)))
        return nodes, cat_nodes


class Block(nn.Module):

    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20, rank=15, shared=False, dropout_input=0.0, dropout_pre_lin=0.0, dropout_output=0.0, pos_norm='before_cat'):
        super().__init__()
        self.rank = rank
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert pos_norm in ['before_cat', 'after_cat']
        self.pos_norm = pos_norm
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = self.linear0 if shared else nn.Linear(input_dims[1], mm_dim)
        self.merge_linears0 = nn.ModuleList()
        self.merge_linears1 = nn.ModuleList()
        self.chunks = self.chunk_sizes(mm_dim, chunks)
        for size in self.chunks:
            ml0 = nn.Linear(size, size * rank)
            self.merge_linears0.append(ml0)
            ml1 = ml0 if shared else nn.Linear(size, size * rank)
            self.merge_linears1.append(ml1)
        self.linear_out = nn.Linear(mm_dim, output_dim)

    def forward(self, x):
        x0 = self.linear0(x[0])
        x1 = self.linear1(x[1])
        bs = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = torch.split(x0, self.chunks, -1)
        x1_chunks = torch.split(x1, self.chunks, -1)
        zs = []
        for x0_c, x1_c, m0, m1 in zip(x0_chunks, x1_chunks, self.merge_linears0, self.merge_linears1):
            m = m0(x0_c) * m1(x1_c)
            m = m.view(bs, self.rank, -1)
            z = torch.sum(m, 1)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z)
            zs.append(z)
        z = torch.cat(zs, 1)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z)
        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z

    @staticmethod
    def chunk_sizes(dim, chunks):
        split_size = (dim + chunks - 1) // chunks
        sizes_list = [split_size] * chunks
        sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
        return sizes_list


class SDMGRLoss(nn.Module):
    """The implementation the loss of key information extraction proposed in
    the paper: Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction.

    https://arxiv.org/abs/2103.14470.
    """

    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=-100):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def forward(self, node_preds, edge_preds, gts):
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].contiguous().view(-1))
        node_gts = torch.cat(node_gts).long()
        edge_gts = torch.cat(edge_gts).long()
        node_valids = torch.nonzero(node_gts != self.ignore, as_tuple=False).view(-1)
        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).view(-1)
        return dict(loss_node=self.node_weight * self.loss_node(node_preds, node_gts), loss_edge=self.edge_weight * self.loss_edge(edge_preds, edge_gts), acc_node=accuracy(node_preds[node_valids], node_gts[node_valids]), acc_edge=accuracy(edge_preds[edge_valids], edge_gts[edge_valids]))


class MaskedCrossEntropyLoss(nn.Module):
    """The implementation of masked cross entropy loss.

    The mask has 1 for real tokens and 0 for padding tokens,
        which only keep active parts of the cross entropy loss.
    Args:
        num_labels (int): Number of classes in labels.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, num_labels=None, ignore_index=0):
        super().__init__()
        self.num_labels = num_labels
        self.criterion = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, img_metas):
        """Loss forword.
        Args:
            logits: Model output with shape [N, C].
            img_metas (dict): A dict containing the following keys:
                    - img (list]): This parameter is reserved.
                    - labels (list[int]): The labels for each word
                        of the sequence.
                    - texts (list): The words of the sequence.
                    - input_ids (list): The ids for each word of
                        the sequence.
                    - attention_mask (list): The mask for each word
                        of the sequence. The mask has 1 for real tokens
                        and 0 for padding tokens. Only real tokens are
                        attended to.
                    - token_type_ids (list): The tokens for each word
                        of the sequence.
        """
        labels = img_metas['labels']
        attention_masks = img_metas['attention_masks']
        if attention_masks is not None:
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss_cls': loss}


class MaskedFocalLoss(nn.Module):
    """The implementation of masked focal loss.

    The mask has 1 for real tokens and 0 for padding tokens,
        which only keep active parts of the focal loss
    Args:
        num_labels (int): Number of classes in labels.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, num_labels=None, ignore_index=0):
        super().__init__()
        self.num_labels = num_labels
        self.criterion = FocalLoss(ignore_index=ignore_index)

    def forward(self, logits, img_metas):
        """Loss forword.
        Args:
            logits: Model output with shape [N, C].
            img_metas (dict): A dict containing the following keys:
                    - img (list]): This parameter is reserved.
                    - labels (list[int]): The labels for each word
                        of the sequence.
                    - texts (list): The words of the sequence.
                    - input_ids (list): The ids for each word of
                        the sequence.
                    - attention_mask (list): The mask for each word
                        of the sequence. The mask has 1 for real tokens
                        and 0 for padding tokens. Only real tokens are
                        attended to.
                    - token_type_ids (list): The tokens for each word
                        of the sequence.
        """
        labels = img_metas['labels']
        attention_masks = img_metas['attention_masks']
        if attention_masks is not None:
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss_cls': loss}


class GeluNew(nn.Module):
    """Implementation of the gelu activation function currently in Google Bert
    repo (identical to OpenAI GPT).

    Also see https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    Args:
        vocab_size (int): Number of words supported.
        hidden_size (int): Hidden size.
        max_position_embeddings (int): Max positions embedding size.
        type_vocab_size (int): The size of type_vocab.
        layer_norm_eps (float): eps.
        hidden_dropout_prob (float): The dropout probability of hidden layer.
    """

    def __init__(self, vocab_size=21128, hidden_size=768, max_position_embeddings=128, type_vocab_size=2, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_emb = self.word_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        embeddings = words_emb + position_emb + token_type_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """Bert self attention module.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    """

    def __init__(self, hidden_size=768, num_attention_heads=12, output_attentions=False, attention_probs_dropout_prob=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple ofthe number of attention heads (%d)' % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.att_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.att_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.att_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    """Bert self output.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    """

    def __init__(self, hidden_size=768, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Bert Attention module implementation.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    """

    def __init__(self, hidden_size=768, num_attention_heads=12, output_attentions=False, attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
        super().__init__()
        self.self = BertSelfAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, output_attentions=output_attentions, attention_probs_dropout_prob=attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size=hidden_size, layer_norm_eps=layer_norm_eps, hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    """Bert BertIntermediate module implementation.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    """

    def __init__(self, hidden_size=768, intermediate_size=3072, hidden_act_cfg=dict(type='GeluNew')):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = build_activation_layer(hidden_act_cfg)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Bert output module.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    """

    def __init__(self, intermediate_size=3072, hidden_size=768, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Bert layer.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch.
    """

    def __init__(self, hidden_size=768, num_attention_heads=12, output_attentions=False, attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12, hidden_dropout_prob=0.1, intermediate_size=3072, hidden_act_cfg=dict(type='GeluNew')):
        super().__init__()
        self.attention = BertAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, output_attentions=output_attentions, attention_probs_dropout_prob=attention_probs_dropout_prob, layer_norm_eps=layer_norm_eps, hidden_dropout_prob=hidden_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act_cfg=hidden_act_cfg)
        self.output = BertOutput(intermediate_size=intermediate_size, hidden_size=hidden_size, layer_norm_eps=layer_norm_eps, hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    """The code is adapted from https://github.com/lonePatient/BERT-NER-
    Pytorch."""

    def __init__(self, output_attentions=False, output_hidden_states=False, num_hidden_layers=12, hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12, hidden_dropout_prob=0.1, intermediate_size=3072, hidden_act_cfg=dict(type='GeluNew')):
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.layer = nn.ModuleList([BertLayer(hidden_size=hidden_size, num_attention_heads=num_attention_heads, output_attentions=output_attentions, attention_probs_dropout_prob=attention_probs_dropout_prob, layer_norm_eps=layer_norm_eps, hidden_dropout_prob=hidden_dropout_prob, intermediate_size=intermediate_size, hidden_act_cfg=hidden_act_cfg) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):

    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """Implement Bert model for named entity recognition task.

    The code is adapted from https://github.com/lonePatient/BERT-NER-Pytorch
    Args:
        num_hidden_layers (int): The number of hidden layers.
        initializer_range (float):
        vocab_size (int): Number of words supported.
        hidden_size (int): Hidden size.
        max_position_embeddings (int): Max positionsembedding size.
        type_vocab_size (int): The size of type_vocab.
        layer_norm_eps (float): eps.
        hidden_dropout_prob (float): The dropout probability of hidden layer.
        output_attentions (bool):  Whether use the attentions in output
        output_hidden_states (bool): Whether use the hidden_states in output.
        num_attention_heads (int): The number of attention heads.
        attention_probs_dropout_prob (float): The dropout probability
            for the attention probabilities normalized from
            the attention scores.
        intermediate_size (int): The size of intermediate layer.
        hidden_act_cfg (str):  hidden layer activation
    """

    def __init__(self, num_hidden_layers=12, initializer_range=0.02, vocab_size=21128, hidden_size=768, max_position_embeddings=128, type_vocab_size=2, layer_norm_eps=1e-12, hidden_dropout_prob=0.1, output_attentions=False, output_hidden_states=False, num_attention_heads=12, attention_probs_dropout_prob=0.1, intermediate_size=3072, hidden_act_cfg=dict(type='GeluNew')):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size=vocab_size, hidden_size=hidden_size, max_position_embeddings=max_position_embeddings, type_vocab_size=type_vocab_size, layer_norm_eps=layer_norm_eps, hidden_dropout_prob=hidden_dropout_prob)
        self.encoder = BertEncoder(output_attentions=output_attentions, output_hidden_states=output_hidden_states, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, num_attention_heads=num_attention_heads, attention_probs_dropout_prob=attention_probs_dropout_prob, layer_norm_eps=layer_norm_eps, hidden_dropout_prob=hidden_dropout_prob, intermediate_size=intermediate_size, hidden_act_cfg=hidden_act_cfg)
        self.pooler = BertPooler(hidden_size=hidden_size)
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def forward(self, input_ids, attention_masks=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_masks is None:
            attention_masks = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        attention_masks = attention_masks[:, None, None]
        attention_masks = attention_masks
        attention_masks = (1.0 - attention_masks) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask[None, None, :, None, None]
            elif head_mask.dim() == 2:
                head_mask = head_mask[None, :, None, None]
            head_mask = head_mask
        else:
            head_mask = [None] * self.num_hidden_layers
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        sequence_output, *encoder_outputs = self.encoder(embedding_output, attention_masks, head_mask=head_mask)
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + tuple(encoder_outputs)
        return outputs

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """Initialize and prunes weights if needed."""
        self.apply(self._init_weights)


class DBLoss(nn.Module):
    """The class for implementing DBNet loss.

    This is partially adapted from https://github.com/MhLiao/DB.

    Args:
        alpha (float): The binary loss coef.
        beta (float): The threshold loss coef.
        reduction (str): The way to reduce the loss.
        negative_ratio (float): The ratio of positives to negatives.
        eps (float): Epsilon in the threshold loss function.
        bbce_loss (bool): Whether to use balanced bce for probability loss.
            If False, dice loss will be used instead.
    """

    def __init__(self, alpha=1, beta=1, reduction='mean', negative_ratio=3.0, eps=1e-06, bbce_loss=False):
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bbce_loss = bbce_loss
        self.dice_loss = DiceLoss(eps=eps)

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert isinstance(bitmasks, list)
        assert isinstance(target_sz, tuple)
        batch_size = len(bitmasks)
        num_levels = len(bitmasks[0])
        result_tensors = []
        for level_inx in range(num_levels):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                mask_sz = mask.shape
                pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            result_tensors.append(kernel)
        return result_tensors

    def balance_bce_loss(self, pred, gt, mask):
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        assert gt.max() <= 1 and gt.min() >= 0
        assert pred.max() <= 1 and pred.min() >= 0
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        return balance_loss

    def l1_thr_loss(self, pred, gt, mask):
        thr_loss = torch.abs((pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return thr_loss

    def forward(self, preds, downsample_ratio, gt_shrink, gt_shrink_mask, gt_thr, gt_thr_mask):
        """Compute DBNet loss.

        Args:
            preds (Tensor): The output tensor with size :math:`(N, 3, H, W)`.
            downsample_ratio (float): The downsample ratio for the
                ground truths.
            gt_shrink (list[BitmapMasks]): The mask list with each element
                being the shrunk text mask for one img.
            gt_shrink_mask (list[BitmapMasks]): The effective mask list with
                each element being the shrunk effective mask for one img.
            gt_thr (list[BitmapMasks]): The mask list with each element
                being the threshold text mask for one img.
            gt_thr_mask (list[BitmapMasks]): The effective mask list with
                each element being the threshold effective mask for one img.

        Returns:
            dict: The dict for dbnet losses with "loss_prob", "loss_db" and
            "loss_thresh".
        """
        assert isinstance(downsample_ratio, float)
        assert isinstance(gt_shrink, list)
        assert isinstance(gt_shrink_mask, list)
        assert isinstance(gt_thr, list)
        assert isinstance(gt_thr_mask, list)
        pred_prob = preds[:, 0, :, :]
        pred_thr = preds[:, 1, :, :]
        pred_db = preds[:, 2, :, :]
        feature_sz = preds.size()
        keys = ['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask']
        gt = {}
        for k in keys:
            gt[k] = eval(k)
            gt[k] = [item.rescale(downsample_ratio) for item in gt[k]]
            gt[k] = self.bitmasks2tensor(gt[k], feature_sz[2:])
            gt[k] = [item for item in gt[k]]
        gt['gt_shrink'][0] = (gt['gt_shrink'][0] > 0).float()
        if self.bbce_loss:
            loss_prob = self.balance_bce_loss(pred_prob, gt['gt_shrink'][0], gt['gt_shrink_mask'][0])
        else:
            loss_prob = self.dice_loss(pred_prob, gt['gt_shrink'][0], gt['gt_shrink_mask'][0])
        loss_db = self.dice_loss(pred_db, gt['gt_shrink'][0], gt['gt_shrink_mask'][0])
        loss_thr = self.l1_thr_loss(pred_thr, gt['gt_thr'][0], gt['gt_thr_mask'][0])
        results = dict(loss_prob=self.alpha * loss_prob, loss_db=loss_db, loss_thr=self.beta * loss_thr)
        return results


class DRRGLoss(nn.Module):
    """The class for implementing DRRG loss. This is partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape Text
    Detection <https://arxiv.org/abs/1908.05900>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem.
    """

    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio

    def balance_bce_loss(self, pred, gt, mask):
        """Balanced Binary-CrossEntropy Loss.

        Args:
            pred (Tensor): Shape of :math:`(1, H, W)`.
            gt (Tensor): Shape of :math:`(1, H, W)`.
            mask (Tensor): Shape of :math:`(1, H, W)`.

        Returns:
            Tensor: Balanced bce loss.
        """
        assert pred.shape == gt.shape == mask.shape
        assert torch.all(pred >= 0) and torch.all(pred <= 1)
        assert torch.all(gt >= 0) and torch.all(gt <= 1)
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.float().sum())
        gt = gt.float()
        if positive_count > 0:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = torch.sum(loss * positive.float())
            negative_loss = loss * negative.float()
            negative_count = min(int(negative.float().sum()), int(positive_count * self.ohem_ratio))
        else:
            positive_loss = torch.tensor(0.0, device=pred.device)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative.float()
            negative_count = 100
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss + torch.sum(negative_loss)) / (float(positive_count + negative_count) + 1e-05)
        return balance_loss

    def gcn_loss(self, gcn_data):
        """CrossEntropy Loss from gcn module.

        Args:
            gcn_data (tuple(Tensor, Tensor)): The first is the
                prediction with shape :math:`(N, 2)` and the
                second is the gt label with shape :math:`(m, n)`
                where :math:`m * n = N`.

        Returns:
            Tensor: CrossEntropy loss.
        """
        gcn_pred, gt_labels = gcn_data
        gt_labels = gt_labels.view(-1)
        loss = F.cross_entropy(gcn_pred, gt_labels)
        return loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)
        batch_size = len(bitmasks)
        num_masks = len(bitmasks[0])
        results = []
        for level_inx in range(num_masks):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                mask_sz = mask.shape
                pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            results.append(kernel)
        return results

    def forward(self, preds, downsample_ratio, gt_text_mask, gt_center_region_mask, gt_mask, gt_top_height_map, gt_bot_height_map, gt_sin_map, gt_cos_map):
        """Compute Drrg loss.

        Args:
            preds (tuple(Tensor)): The first is the prediction map
                with shape :math:`(N, C_{out}, H, W)`.
                The second is prediction from GCN module, with
                shape :math:`(N, 2)`.
                The third is ground-truth label with shape :math:`(N, 8)`.
            downsample_ratio (float): The downsample ratio.
            gt_text_mask (list[BitmapMasks]): Text mask.
            gt_center_region_mask (list[BitmapMasks]): Center region mask.
            gt_mask (list[BitmapMasks]): Effective mask.
            gt_top_height_map (list[BitmapMasks]): Top height map.
            gt_bot_height_map (list[BitmapMasks]): Bottom height map.
            gt_sin_map (list[BitmapMasks]): Sinusoid map.
            gt_cos_map (list[BitmapMasks]): Cosine map.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_height``, ``loss_sin``, ``loss_cos``, and ``loss_gcn``.
        """
        assert isinstance(preds, tuple)
        assert isinstance(downsample_ratio, float)
        assert check_argument.is_type_list(gt_text_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_center_region_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_top_height_map, BitmapMasks)
        assert check_argument.is_type_list(gt_bot_height_map, BitmapMasks)
        assert check_argument.is_type_list(gt_sin_map, BitmapMasks)
        assert check_argument.is_type_list(gt_cos_map, BitmapMasks)
        pred_maps, gcn_data = preds
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_top_height_map = pred_maps[:, 4, :, :]
        pred_bot_height_map = pred_maps[:, 5, :, :]
        feature_sz = pred_maps.size()
        device = pred_maps.device
        mapping = {'gt_text_mask': gt_text_mask, 'gt_center_region_mask': gt_center_region_mask, 'gt_mask': gt_mask, 'gt_top_height_map': gt_top_height_map, 'gt_bot_height_map': gt_bot_height_map, 'gt_sin_map': gt_sin_map, 'gt_cos_map': gt_cos_map}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 0.01:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key in ['gt_top_height_map', 'gt_bot_height_map']:
                    gt[key] = [(item * downsample_ratio) for item in gt[key]]
            gt[key] = [item for item in gt[key]]
        scale = torch.sqrt(1.0 / (pred_sin_map ** 2 + pred_cos_map ** 2 + 1e-08))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale
        loss_text = self.balance_bce_loss(torch.sigmoid(pred_text_region), gt['gt_text_mask'][0], gt['gt_mask'][0])
        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0]).float()
        negative_text_mask = ((1 - gt['gt_text_mask'][0]) * gt['gt_mask'][0]).float()
        loss_center_map = F.binary_cross_entropy(torch.sigmoid(pred_center_region), gt['gt_center_region_mask'][0].float(), reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center_positive = torch.sum(loss_center_map * text_mask) / torch.sum(text_mask)
        else:
            loss_center_positive = torch.tensor(0.0, device=device)
        loss_center_negative = torch.sum(loss_center_map * negative_text_mask) / torch.sum(negative_text_mask)
        loss_center = loss_center_positive + 0.5 * loss_center_negative
        center_mask = (gt['gt_center_region_mask'][0] * gt['gt_mask'][0]).float()
        if int(center_mask.sum()) > 0:
            map_sz = pred_top_height_map.size()
            ones = torch.ones(map_sz, dtype=torch.float, device=device)
            loss_top = F.smooth_l1_loss(pred_top_height_map / (gt['gt_top_height_map'][0] + 0.01), ones, reduction='none')
            loss_bot = F.smooth_l1_loss(pred_bot_height_map / (gt['gt_bot_height_map'][0] + 0.01), ones, reduction='none')
            gt_height = gt['gt_top_height_map'][0] + gt['gt_bot_height_map'][0]
            loss_height = torch.sum(torch.log(gt_height + 1) * (loss_top + loss_bot) * center_mask) / torch.sum(center_mask)
            loss_sin = torch.sum(F.smooth_l1_loss(pred_sin_map, gt['gt_sin_map'][0], reduction='none') * center_mask) / torch.sum(center_mask)
            loss_cos = torch.sum(F.smooth_l1_loss(pred_cos_map, gt['gt_cos_map'][0], reduction='none') * center_mask) / torch.sum(center_mask)
        else:
            loss_height = torch.tensor(0.0, device=device)
            loss_sin = torch.tensor(0.0, device=device)
            loss_cos = torch.tensor(0.0, device=device)
        loss_gcn = self.gcn_loss(gcn_data)
        results = dict(loss_text=loss_text, loss_center=loss_center, loss_height=loss_height, loss_sin=loss_sin, loss_cos=loss_cos, loss_gcn=loss_gcn)
        return results


class FCELoss(nn.Module):
    """The class for implementing FCENet loss.

    FCENet(CVPR2021): `Fourier Contour Embedding for Arbitrary-shaped Text
    Detection <https://arxiv.org/abs/2104.10442>`_

    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, fourier_degree, num_sample, ohem_ratio=3.0):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio

    def forward(self, preds, _, p3_maps, p4_maps, p5_maps):
        """Compute FCENet loss.

        Args:
            preds (list[list[Tensor]]): The outer list indicates images
                in a batch, and the inner list indicates the classification
                prediction map (with shape :math:`(N, C, H, W)`) and
                regression map (with shape :math:`(N, C, H, W)`).
            p3_maps (list[ndarray]): List of leval 3 ground truth target map
                with shape :math:`(C, H, W)`.
            p4_maps (list[ndarray]): List of leval 4 ground truth target map
                with shape :math:`(C, H, W)`.
            p5_maps (list[ndarray]): List of leval 5 ground truth target map
                with shape :math:`(C, H, W)`.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_reg_x`` and ``loss_reg_y``.
        """
        assert isinstance(preds, list)
        assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5, 'fourier degree not equal in FCEhead and FCEtarget'
        device = preds[0][0].device
        gts = [p3_maps, p4_maps, p5_maps]
        for idx, maps in enumerate(gts):
            gts[idx] = torch.from_numpy(np.stack(maps)).float()
        losses = multi_apply(self.forward_single, preds, gts)
        loss_tr = torch.tensor(0.0, device=device).float()
        loss_tcl = torch.tensor(0.0, device=device).float()
        loss_reg_x = torch.tensor(0.0, device=device).float()
        loss_reg_y = torch.tensor(0.0, device=device).float()
        for idx, loss in enumerate(losses):
            if idx == 0:
                loss_tr += sum(loss)
            elif idx == 1:
                loss_tcl += sum(loss)
            elif idx == 2:
                loss_reg_x += sum(loss)
            else:
                loss_reg_y += sum(loss)
        results = dict(loss_text=loss_tr, loss_center=loss_tcl, loss_reg_x=loss_reg_x, loss_reg_y=loss_reg_y)
        return results

    def forward_single(self, pred, gt):
        cls_pred = pred[0].permute(0, 2, 3, 1).contiguous()
        reg_pred = pred[1].permute(0, 2, 3, 1).contiguous()
        gt = gt.permute(0, 2, 3, 1).contiguous()
        k = 2 * self.fourier_degree + 1
        tr_pred = cls_pred[:, :, :, :2].view(-1, 2)
        tcl_pred = cls_pred[:, :, :, 2:].view(-1, 2)
        x_pred = reg_pred[:, :, :, 0:k].view(-1, k)
        y_pred = reg_pred[:, :, :, k:2 * k].view(-1, k)
        tr_mask = gt[:, :, :, :1].view(-1)
        tcl_mask = gt[:, :, :, 1:2].view(-1)
        train_mask = gt[:, :, :, 2:3].view(-1)
        x_map = gt[:, :, :, 3:3 + k].view(-1, k)
        y_map = gt[:, :, :, 3 + k:].view(-1, k)
        tr_train_mask = train_mask * tr_mask
        device = x_map.device
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())
        loss_tcl = torch.tensor(0.0).float()
        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(tcl_pred[tr_train_mask.bool()], tcl_mask[tr_train_mask.bool()].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask.bool()], tcl_mask[tr_neg_mask.bool()].long())
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg
        loss_reg_x = torch.tensor(0.0).float()
        loss_reg_y = torch.tensor(0.0).float()
        if tr_train_mask.sum().item() > 0:
            weight = (tr_mask[tr_train_mask.bool()].float() + tcl_mask[tr_train_mask.bool()].float()) / 2
            weight = weight.contiguous().view(-1, 1)
            ft_x, ft_y = self.fourier2poly(x_map, y_map)
            ft_x_pre, ft_y_pre = self.fourier2poly(x_pred, y_pred)
            loss_reg_x = torch.mean(weight * F.smooth_l1_loss(ft_x_pre[tr_train_mask.bool()], ft_x[tr_train_mask.bool()], reduction='none'))
            loss_reg_y = torch.mean(weight * F.smooth_l1_loss(ft_y_pre[tr_train_mask.bool()], ft_y[tr_train_mask.bool()], reduction='none'))
        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def ohem(self, predict, target, train_mask):
        device = train_mask.device
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()
        n_pos = pos.float().sum()
        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(self.ohem_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.0)
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            loss_neg, _ = torch.topk(loss_neg, n_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def fourier2poly(self, real_maps, imag_maps):
        """Transform Fourier coefficient maps to polygon maps.

        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)

        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """
        device = real_maps.device
        k_vect = torch.arange(-self.fourier_degree, self.fourier_degree + 1, dtype=torch.float, device=device).view(-1, 1)
        i_vect = torch.arange(0, self.num_sample, dtype=torch.float, device=device).view(1, -1)
        transform_matrix = 2 * np.pi / self.num_sample * torch.mm(k_vect, i_vect)
        x1 = torch.einsum('ak, kn-> an', real_maps, torch.cos(transform_matrix))
        x2 = torch.einsum('ak, kn-> an', imag_maps, torch.sin(transform_matrix))
        y1 = torch.einsum('ak, kn-> an', real_maps, torch.sin(transform_matrix))
        y2 = torch.einsum('ak, kn-> an', imag_maps, torch.cos(transform_matrix))
        x_maps = x1 - x2
        y_maps = y1 + y2
        return x_maps, y_maps


class PANLoss(nn.Module):
    """The class for implementing PANet loss. This was partially adapted from
    https://github.com/WenmuZhou/PAN.pytorch.

    PANet: `Efficient and Accurate Arbitrary-
    Shaped Text Detection with Pixel Aggregation Network
    <https://arxiv.org/abs/1908.05900>`_.

    Args:
        alpha (float): The kernel loss coef.
        beta (float): The aggregation and discriminative loss coef.
        delta_aggregation (float): The constant for aggregation loss.
        delta_discrimination (float): The constant for discriminative loss.
        ohem_ratio (float): The negative/positive ratio in ohem.
        reduction (str): The way to reduce the loss.
        speedup_bbox_thr (int):  Speed up if speedup_bbox_thr > 0
            and < bbox num.
    """

    def __init__(self, alpha=0.5, beta=0.25, delta_aggregation=0.5, delta_discrimination=3, ohem_ratio=3, reduction='mean', speedup_bbox_thr=-1):
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta_aggregation = delta_aggregation
        self.delta_discrimination = delta_discrimination
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.speedup_bbox_thr = speedup_bbox_thr

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)
        batch_size = len(bitmasks)
        num_masks = len(bitmasks[0])
        results = []
        for level_inx in range(num_masks):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                mask_sz = mask.shape
                pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            results.append(kernel)
        return results

    def forward(self, preds, downsample_ratio, gt_kernels, gt_mask):
        """Compute PANet loss.

        Args:
            preds (Tensor): The output tensor of size :math:`(N, 6, H, W)`.
            downsample_ratio (float): The downsample ratio between preds
                and the input img.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_kernel``,
            ``loss_aggregation`` and ``loss_discrimination``.
        """
        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert isinstance(downsample_ratio, float)
        pred_texts = preds[:, 0, :, :]
        pred_kernels = preds[:, 1, :, :]
        inst_embed = preds[:, 2:, :, :]
        feature_sz = preds.size()
        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
            gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            gt[key] = [item for item in gt[key]]
        loss_aggrs, loss_discrs = self.aggregation_discrimination_loss(gt['gt_kernels'][0], gt['gt_kernels'][1], inst_embed)
        sampled_mask = self.ohem_batch(pred_texts.detach(), gt['gt_kernels'][0], gt['gt_mask'][0])
        loss_texts = self.dice_loss_with_logits(pred_texts, gt['gt_kernels'][0], sampled_mask)
        sampled_masks_kernel = (gt['gt_kernels'][0] > 0.5).float() * gt['gt_mask'][0].float()
        loss_kernels = self.dice_loss_with_logits(pred_kernels, gt['gt_kernels'][1], sampled_masks_kernel)
        losses = [loss_texts, loss_kernels, loss_aggrs, loss_discrs]
        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError
        coefs = [1, self.alpha, self.beta, self.beta]
        losses = [(item * scale) for item, scale in zip(losses, coefs)]
        results = dict()
        results.update(loss_text=losses[0], loss_kernel=losses[1], loss_aggregation=losses[2], loss_discrimination=losses[3])
        return results

    def aggregation_discrimination_loss(self, gt_texts, gt_kernels, inst_embeds):
        """Compute the aggregation and discrimnative losses.

        Args:
            gt_texts (Tensor): The ground truth text mask of size
                :math:`(N, 1, H, W)`.
            gt_kernels (Tensor): The ground truth text kernel mask of
                size :math:`(N, 1, H, W)`.
            inst_embeds(Tensor): The text instance embedding tensor
                of size :math:`(N, 1, H, W)`.

        Returns:
            (Tensor, Tensor): A tuple of aggregation loss and discriminative
            loss before reduction.
        """
        batch_size = gt_texts.size()[0]
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
        assert inst_embeds.shape[1] == 4
        inst_embeds = inst_embeds.contiguous().reshape(batch_size, 4, -1)
        loss_aggrs = []
        loss_discrs = []
        for text, kernel, embed in zip(gt_texts, gt_kernels, inst_embeds):
            text_num = int(text.max().item())
            loss_aggr_img = []
            kernel_avgs = []
            select_num = self.speedup_bbox_thr
            if 0 < select_num < text_num:
                inds = np.random.choice(text_num, select_num, replace=False) + 1
            else:
                inds = range(1, text_num + 1)
            for i in inds:
                kernel_i = kernel == i
                if kernel_i.sum() == 0 or (text == i).sum() == 0:
                    continue
                avg = embed[:, kernel_i].mean(1)
                kernel_avgs.append(avg)
                embed_i = embed[:, text == i]
                distance = (embed_i - avg.reshape(4, 1)).norm(2, dim=0) - self.delta_aggregation
                hinge = torch.max(distance, torch.tensor(0, device=distance.device, dtype=torch.float)).pow(2)
                aggr = torch.log(hinge + 1).mean()
                loss_aggr_img.append(aggr)
            num_inst = len(loss_aggr_img)
            if num_inst > 0:
                loss_aggr_img = torch.stack(loss_aggr_img).mean()
            else:
                loss_aggr_img = torch.tensor(0, device=gt_texts.device, dtype=torch.float)
            loss_aggrs.append(loss_aggr_img)
            loss_discr_img = 0
            for avg_i, avg_j in itertools.combinations(kernel_avgs, 2):
                distance_ij = self.delta_discrimination - (avg_i - avg_j).norm(2)
                D_ij = torch.max(distance_ij, torch.tensor(0, device=distance_ij.device, dtype=torch.float)).pow(2)
                loss_discr_img += torch.log(D_ij + 1)
            if num_inst > 1:
                loss_discr_img /= num_inst * (num_inst - 1)
            else:
                loss_discr_img = torch.tensor(0, device=gt_texts.device, dtype=torch.float)
            if num_inst == 0:
                warnings.warn('num of instance is 0')
            loss_discrs.append(loss_discr_img)
        return torch.stack(loss_aggrs), torch.stack(loss_discrs)

    def dice_loss_with_logits(self, pred, target, mask):
        smooth = 0.001
        pred = torch.sigmoid(pred)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)
        pred = pred * mask
        target = target * mask
        a = torch.sum(pred * target, 1) + smooth
        b = torch.sum(pred * pred, 1) + smooth
        c = torch.sum(target * target, 1) + smooth
        d = 2 * a / (b + c)
        return 1 - d

    def ohem_img(self, text_score, gt_text, gt_mask):
        """Sample the top-k maximal negative samples and all positive samples.

        Args:
            text_score (Tensor): The text score of size :math:`(H, W)`.
            gt_text (Tensor): The ground truth text mask of size
                :math:`(H, W)`.
            gt_mask (Tensor): The effective region mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled pixel mask of size :math:`(H, W)`.
        """
        assert isinstance(text_score, torch.Tensor)
        assert isinstance(gt_text, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_score.shape) == 2
        assert text_score.shape == gt_text.shape
        assert gt_text.shape == gt_mask.shape
        pos_num = int(torch.sum(gt_text > 0.5).item()) - int(torch.sum((gt_text > 0.5) * (gt_mask <= 0.5)).item())
        neg_num = int(torch.sum(gt_text <= 0.5).item())
        neg_num = int(min(pos_num * self.ohem_ratio, neg_num))
        if pos_num == 0 or neg_num == 0:
            warnings.warn('pos_num = 0 or neg_num = 0')
            return gt_mask.bool()
        neg_score = text_score[gt_text <= 0.5]
        neg_score_sorted, _ = torch.sort(neg_score, descending=True)
        threshold = neg_score_sorted[neg_num - 1]
        sampled_mask = ((text_score >= threshold) + (gt_text > 0.5) > 0) * (gt_mask > 0.5)
        return sampled_mask

    def ohem_batch(self, text_scores, gt_texts, gt_mask):
        """OHEM sampling for a batch of imgs.

        Args:
            text_scores (Tensor): The text scores of size :math:`(H, W)`.
            gt_texts (Tensor): The gt text masks of size :math:`(H, W)`.
            gt_mask (Tensor): The gt effective mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled mask of size :math:`(H, W)`.
        """
        assert isinstance(text_scores, torch.Tensor)
        assert isinstance(gt_texts, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_scores.shape) == 3
        assert text_scores.shape == gt_texts.shape
        assert gt_texts.shape == gt_mask.shape
        sampled_masks = []
        for i in range(text_scores.shape[0]):
            sampled_masks.append(self.ohem_img(text_scores[i], gt_texts[i], gt_mask[i]))
        sampled_masks = torch.stack(sampled_masks)
        return sampled_masks


class TextSnakeLoss(nn.Module):
    """The class for implementing TextSnake loss. This is partially adapted
    from https://github.com/princewang1994/TextSnake.pytorch.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem.
    """

    def __init__(self, ohem_ratio=3.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio

    def balanced_bce_loss(self, pred, gt, mask):
        assert pred.shape == gt.shape == mask.shape
        positive = gt * mask
        negative = (1 - gt) * mask
        positive_count = int(positive.float().sum())
        gt = gt.float()
        if positive_count > 0:
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            positive_loss = torch.sum(loss * positive.float())
            negative_loss = loss * negative.float()
            negative_count = min(int(negative.float().sum()), int(positive_count * self.ohem_ratio))
        else:
            positive_loss = torch.tensor(0.0, device=pred.device)
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative.float()
            negative_count = 100
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss + torch.sum(negative_loss)) / (float(positive_count + negative_count) + 1e-05)
        return balance_loss

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)
        batch_size = len(bitmasks)
        num_masks = len(bitmasks[0])
        results = []
        for level_inx in range(num_masks):
            kernel = []
            for batch_inx in range(batch_size):
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                mask_sz = mask.shape
                pad = [0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]]
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            kernel = torch.stack(kernel)
            results.append(kernel)
        return results

    def forward(self, pred_maps, downsample_ratio, gt_text_mask, gt_center_region_mask, gt_mask, gt_radius_map, gt_sin_map, gt_cos_map):
        """
        Args:
            pred_maps (Tensor): The prediction map of shape
                :math:`(N, 5, H, W)`, where each dimension is the map of
                "text_region", "center_region", "sin_map", "cos_map", and
                "radius_map" respectively.
            downsample_ratio (float): Downsample ratio.
            gt_text_mask (list[BitmapMasks]): Gold text masks.
            gt_center_region_mask (list[BitmapMasks]): Gold center region
                masks.
            gt_mask (list[BitmapMasks]): Gold general masks.
            gt_radius_map (list[BitmapMasks]): Gold radius maps.
            gt_sin_map (list[BitmapMasks]): Gold sin maps.
            gt_cos_map (list[BitmapMasks]): Gold cos maps.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_radius``, ``loss_sin`` and ``loss_cos``.
        """
        assert isinstance(downsample_ratio, float)
        assert check_argument.is_type_list(gt_text_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_center_region_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_radius_map, BitmapMasks)
        assert check_argument.is_type_list(gt_sin_map, BitmapMasks)
        assert check_argument.is_type_list(gt_cos_map, BitmapMasks)
        pred_text_region = pred_maps[:, 0, :, :]
        pred_center_region = pred_maps[:, 1, :, :]
        pred_sin_map = pred_maps[:, 2, :, :]
        pred_cos_map = pred_maps[:, 3, :, :]
        pred_radius_map = pred_maps[:, 4, :, :]
        feature_sz = pred_maps.size()
        device = pred_maps.device
        mapping = {'gt_text_mask': gt_text_mask, 'gt_center_region_mask': gt_center_region_mask, 'gt_mask': gt_mask, 'gt_radius_map': gt_radius_map, 'gt_sin_map': gt_sin_map, 'gt_cos_map': gt_cos_map}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            if abs(downsample_ratio - 1.0) < 0.01:
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key == 'gt_radius_map':
                    gt[key] = [(item * downsample_ratio) for item in gt[key]]
            gt[key] = [item for item in gt[key]]
        scale = torch.sqrt(1.0 / (pred_sin_map ** 2 + pred_cos_map ** 2 + 1e-08))
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale
        loss_text = self.balanced_bce_loss(torch.sigmoid(pred_text_region), gt['gt_text_mask'][0], gt['gt_mask'][0])
        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0]).float()
        loss_center_map = F.binary_cross_entropy(torch.sigmoid(pred_center_region), gt['gt_center_region_mask'][0].float(), reduction='none')
        if int(text_mask.sum()) > 0:
            loss_center = torch.sum(loss_center_map * text_mask) / torch.sum(text_mask)
        else:
            loss_center = torch.tensor(0.0, device=device)
        center_mask = (gt['gt_center_region_mask'][0] * gt['gt_mask'][0]).float()
        if int(center_mask.sum()) > 0:
            map_sz = pred_radius_map.size()
            ones = torch.ones(map_sz, dtype=torch.float, device=device)
            loss_radius = torch.sum(F.smooth_l1_loss(pred_radius_map / (gt['gt_radius_map'][0] + 0.01), ones, reduction='none') * center_mask) / torch.sum(center_mask)
            loss_sin = torch.sum(F.smooth_l1_loss(pred_sin_map, gt['gt_sin_map'][0], reduction='none') * center_mask) / torch.sum(center_mask)
            loss_cos = torch.sum(F.smooth_l1_loss(pred_cos_map, gt['gt_cos_map'][0], reduction='none') * center_mask) / torch.sum(center_mask)
        else:
            loss_radius = torch.tensor(0.0, device=device)
            loss_sin = torch.tensor(0.0, device=device)
            loss_cos = torch.tensor(0.0, device=device)
        results = dict(loss_text=loss_text, loss_center=loss_center, loss_radius=loss_radius, loss_sin=loss_sin, loss_cos=loss_cos)
        return results


class MeanAggregator(nn.Module):

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim * 2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.aggregator = MeanAggregator()

    def forward(self, features, A):
        b, n, d = features.shape
        assert d == self.in_dim
        agg_feats = self.aggregator(features, A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', cat_feats, self.weight)
        out = F.relu(out + self.bias)
        return out


class GCN(nn.Module):
    """Graph convolutional network for clustering. This was from repo
    https://github.com/Zhongdao/gcn_clustering licensed under the MIT license.

    Args:
        feat_len(int): The input node feature length.
    """

    def __init__(self, feat_len):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(feat_len, affine=False).float()
        self.conv1 = GraphConv(feat_len, 512)
        self.conv2 = GraphConv(512, 256)
        self.conv3 = GraphConv(256, 128)
        self.conv4 = GraphConv(128, 64)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.PReLU(32), nn.Linear(32, 2))

    def forward(self, x, A, knn_inds):
        num_local_graphs, num_max_nodes, feat_len = x.shape
        x = x.view(-1, feat_len)
        x = self.bn0(x)
        x = x.view(num_local_graphs, num_max_nodes, feat_len)
        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k = knn_inds.size(-1)
        mid_feat_len = x.size(-1)
        edge_feat = torch.zeros((num_local_graphs, k, mid_feat_len), device=x.device)
        for graph_ind in range(num_local_graphs):
            edge_feat[graph_ind, :, :] = x[graph_ind, knn_inds[graph_ind]]
        edge_feat = edge_feat.view(-1, mid_feat_len)
        pred = self.classifier(edge_feat)
        return pred


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_conv1x1=False, plugins=None):
        super(BasicBlock, self).__init__()
        if use_conv1x1:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes * self.expansion, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes * self.expansion)
        self.with_plugins = False
        if plugins:
            if isinstance(plugins, dict):
                plugins = [plugins]
            self.with_plugins = True
            self.before_conv1_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'before_conv1']
            self.after_conv1_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv1']
            self.after_conv2_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_conv2']
            self.after_shortcut_plugin = [plugin['cfg'] for plugin in plugins if plugin['position'] == 'after_shortcut']
        self.planes = planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        if self.with_plugins:
            self.before_conv1_plugin_names = self.make_block_plugins(inplanes, self.before_conv1_plugin)
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugin)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugin)
            self.after_shortcut_plugin_names = self.make_block_plugins(planes, self.after_shortcut_plugin)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, out_channels=in_channels, postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    def forward(self, x):
        if self.with_plugins:
            x = self.forward_plugin(x, self.before_conv1_plugin_names)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv1_plugin_names)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_conv2_plugin_names)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_shortcut_plugin_names)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * self.expansion, 1, stride, bias=False), nn.BatchNorm2d(planes * self.expansion))
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class DotProductAttentionLayer(nn.Module):

    def __init__(self, dim_model=None):
        super().__init__()
        self.scale = dim_model ** -0.5 if dim_model is not None else 1.0

    def forward(self, query, key, value, mask=None):
        n, seq_len = mask.size()
        logits = torch.matmul(query.permute(0, 2, 1), key) * self.scale
        if mask is not None:
            mask = mask.view(n, 1, seq_len)
            logits = logits.masked_fill(mask, float('-inf'))
        weights = F.softmax(logits, dim=2)
        glimpse = torch.matmul(weights, value.transpose(1, 2))
        glimpse = glimpse.permute(0, 2, 1).contiguous()
        return glimpse


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class PositionAwareLayer(nn.Module):

    def __init__(self, dim_model, rnn_layers=2):
        super().__init__()
        self.dim_model = dim_model
        self.rnn = nn.LSTM(input_size=dim_model, hidden_size=dim_model, num_layers=rnn_layers, batch_first=True)
        self.mixer = nn.Sequential(nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1, padding=1), nn.ReLU(True), nn.Conv2d(dim_model, dim_model, kernel_size=3, stride=1, padding=1))

    def forward(self, img_feature):
        n, c, h, w = img_feature.size()
        rnn_input = img_feature.permute(0, 2, 3, 1).contiguous()
        rnn_input = rnn_input.view(n * h, w, c)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = rnn_output.view(n, h, w, c)
        rnn_output = rnn_output.permute(0, 3, 1, 2).contiguous()
        out = self.mixer(rnn_output)
        return out


class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
    """

    def __init__(self, ignore_index=-1, reduction='none', ignore_first_char=False):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.ignore_first_char = ignore_first_char

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        if self.ignore_first_char:
            targets = targets[:, 1:].contiguous()
            outputs = outputs[:, :-1, :]
        outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs, targets

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with a key ``padded_targets``, which is
                a tensor of shape :math:`(N, T)`. Each element is the index of
                a character.
            img_metas (None): Unused.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        outputs, targets = self.format(outputs, targets_dict)
        loss_ce = self.loss_ce(outputs, targets)
        losses = dict(loss_ce=loss_ce)
        return losses


class SARLoss(CELoss):
    """Implementation of loss module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ("none", "mean", "sum").

    Warning:
        SARLoss assumes that the first input token is always `<SOS>`.
    """

    def __init__(self, ignore_index=-1, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()
        return outputs, targets


class TFLoss(CELoss):
    """Implementation of loss module for transformer.

    Args:
        ignore_index (int, optional): The character index to be ignored in
            loss computation.
        reduction (str): Type of reduction to apply to the output,
            should be one of the following: ("none", "mean", "sum").
        flatten (bool): Whether to flatten the vectors for loss computation.

    Warning:
        TFLoss assumes that the first input token is always `<SOS>`.
    """

    def __init__(self, ignore_index=-1, reduction='none', flatten=True, **kwargs):
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)
        self.flatten = flatten

    def format(self, outputs, targets_dict):
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets_dict['padded_targets']
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
        else:
            outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs, targets


class CTCLoss(nn.Module):
    """Implementation of loss module for CTC-loss based text recognition.

    Args:
        flatten (bool): If True, use flattened targets, else padded targets.
        blank (int): Blank label. Default 0.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        zero_infinity (bool): Whether to zero infinite losses and
            the associated gradients. Default: False.
            Infinite losses mainly occur when the inputs
            are too short to be aligned to the targets.
    """

    def __init__(self, flatten=True, blank=0, reduction='mean', zero_infinity=False, **kwargs):
        super().__init__()
        assert isinstance(flatten, bool)
        assert isinstance(blank, int)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)
        self.flatten = flatten
        self.blank = blank
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with 3 keys ``target_lengths``,
                ``flatten_targets`` and ``targets``.

                - | ``target_lengths`` (Tensor): A tensor of shape :math:`(N)`.
                    Each item is the length of a word.

                - | ``flatten_targets`` (Tensor): Used if ``self.flatten=True``
                    (default). A tensor of shape
                    (sum(targets_dict['target_lengths'])). Each item is the
                    index of a character.

                - | ``targets`` (Tensor): Used if ``self.flatten=False``. A
                    tensor of :math:`(N, T)`. Empty slots are padded with
                    ``self.blank``.

            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            dict: The loss dict with key ``loss_ctc``.
        """
        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [img_meta.get('valid_ratio', 1.0) for img_meta in img_metas]
        outputs = torch.log_softmax(outputs, dim=2)
        bsz, seq_len = outputs.size(0), outputs.size(1)
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()
        if self.flatten:
            targets = targets_dict['flatten_targets']
        else:
            targets = torch.full(size=(bsz, seq_len), fill_value=self.blank, dtype=torch.long)
            for idx, tensor in enumerate(targets_dict['targets']):
                valid_len = min(tensor.size(0), seq_len)
                targets[idx, :valid_len] = tensor[:valid_len]
        target_lengths = targets_dict['target_lengths']
        target_lengths = torch.clamp(target_lengths, min=1, max=seq_len).long()
        input_lengths = torch.full(size=(bsz,), fill_value=seq_len, dtype=torch.long)
        if not self.flatten and valid_ratios is not None:
            input_lengths = [math.ceil(valid_ratio * seq_len) for valid_ratio in valid_ratios]
            input_lengths = torch.Tensor(input_lengths).long()
        loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths, target_lengths)
        losses = dict(loss_ctc=loss_ctc)
        return losses


class ABILoss(nn.Module):
    """Implementation of ABINet multiloss that allows mixing different types of
    losses with weights.

    Args:
        enc_weight (float): The weight of encoder loss. Defaults to 1.0.
        dec_weight (float): The weight of decoder loss. Defaults to 1.0.
        fusion_weight (float): The weight of fuser (aligner) loss.
            Defaults to 1.0.
        num_classes (int): Number of unique output language tokens.

    Returns:
        A dictionary whose key/value pairs are the losses of three modules.
    """

    def __init__(self, enc_weight=1.0, dec_weight=1.0, fusion_weight=1.0, num_classes=37, **kwargs):
        assert isinstance(enc_weight, float) or isinstance(enc_weight, int)
        assert isinstance(dec_weight, float) or isinstance(dec_weight, int)
        assert isinstance(fusion_weight, float) or isinstance(fusion_weight, int)
        super().__init__()
        self.enc_weight = enc_weight
        self.dec_weight = dec_weight
        self.fusion_weight = fusion_weight
        self.num_classes = num_classes

    def _flatten(self, logits, target_lens):
        flatten_logits = torch.cat([s[:target_lens[i]] for i, s in enumerate(logits)])
        return flatten_logits

    def _ce_loss(self, logits, targets):
        targets_one_hot = F.one_hot(targets, self.num_classes)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(targets_one_hot * log_prob).sum(dim=-1)
        return loss.mean()

    def _loss_over_iters(self, outputs, targets):
        """
        Args:
            outputs (list[Tensor]): Each tensor has shape (N, T, C) where N is
                the batch size, T is the sequence length and C is the number of
                classes.
            targets_dicts (dict): The dictionary with at least `padded_targets`
                defined.
        """
        iter_num = len(outputs)
        dec_outputs = torch.cat(outputs, dim=0)
        flatten_targets_iternum = targets.repeat(iter_num)
        return self._ce_loss(dec_outputs, flatten_targets_iternum)

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (dict): The output dictionary with at least one of
                ``out_enc``, ``out_dec`` and ``out_fusers`` specified.
            targets_dict (dict): The target dictionary containing the key
                ``padded_targets``, which represents target sequences in
                shape (batch_size, sequence_length).

        Returns:
            A loss dictionary with ``loss_visual``, ``loss_lang`` and
            ``loss_fusion``. Each should either be the loss tensor or ``0`` if
            the output of its corresponding module is not given.
        """
        assert 'out_enc' in outputs or 'out_dec' in outputs or 'out_fusers' in outputs
        losses = {}
        target_lens = [len(t) for t in targets_dict['targets']]
        flatten_targets = torch.cat([t for t in targets_dict['targets']])
        if outputs.get('out_enc', None):
            enc_input = self._flatten(outputs['out_enc']['logits'], target_lens)
            enc_loss = self._ce_loss(enc_input, flatten_targets) * self.enc_weight
            losses['loss_visual'] = enc_loss
        if outputs.get('out_decs', None):
            dec_logits = [self._flatten(o['logits'], target_lens) for o in outputs['out_decs']]
            dec_loss = self._loss_over_iters(dec_logits, flatten_targets) * self.dec_weight
            losses['loss_lang'] = dec_loss
        if outputs.get('out_fusers', None):
            fusion_logits = [self._flatten(o['logits'], target_lens) for o in outputs['out_fusers']]
            fusion_loss = self._loss_over_iters(fusion_logits, flatten_targets) * self.fusion_weight
            losses['loss_fusion'] = fusion_loss
        return losses


class SegLoss(nn.Module):
    """Implementation of loss module for segmentation based text recognition
    method.

    Args:
        seg_downsample_ratio (float): Downsample ratio of
            segmentation map.
        seg_with_loss_weight (bool): If True, set weight for
            segmentation loss.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """

    def __init__(self, seg_downsample_ratio=0.5, seg_with_loss_weight=True, ignore_index=255, **kwargs):
        super().__init__()
        assert isinstance(seg_downsample_ratio, (int, float))
        assert 0 < seg_downsample_ratio <= 1
        assert isinstance(ignore_index, int)
        self.seg_downsample_ratio = seg_downsample_ratio
        self.seg_with_loss_weight = seg_with_loss_weight
        self.ignore_index = ignore_index

    def seg_loss(self, out_head, gt_kernels):
        seg_map = out_head
        seg_target = [item[1].rescale(self.seg_downsample_ratio).to_tensor(torch.long, seg_map.device) for item in gt_kernels]
        seg_target = torch.stack(seg_target).squeeze(1)
        loss_weight = None
        if self.seg_with_loss_weight:
            N = torch.sum(seg_target != self.ignore_index)
            N_neg = torch.sum(seg_target == 0)
            weight_val = 1.0 * N_neg / (N - N_neg)
            loss_weight = torch.ones(seg_map.size(1), device=seg_map.device)
            loss_weight[1:] = weight_val
        loss_seg = F.cross_entropy(seg_map, seg_target, weight=loss_weight, ignore_index=self.ignore_index)
        return loss_seg

    def forward(self, out_neck, out_head, gt_kernels):
        """
        Args:
            out_neck (None): Unused.
            out_head (Tensor): The output from head whose shape
                is :math:`(N, C, H, W)`.
            gt_kernels (BitmapMasks): The ground truth masks.

        Returns:
            dict: A loss dictionary with the key ``loss_seg``.
        """
        losses = {}
        loss_seg = self.seg_loss(out_head, gt_kernels)
        losses['loss_seg'] = loss_seg
        return losses


class Maxpool2d(nn.Module):
    """A wrapper around nn.Maxpool2d().

    Args:
        kernel_size (int or tuple(int)): Kernel size for max pooling layer
        stride (int or tuple(int)): Stride for max pooling layer
        padding (int or tuple(int)): Padding for pooling layer
    """

    def __init__(self, kernel_size, stride, padding=0, **kwargs):
        super(Maxpool2d, self).__init__()
        self.model = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map

        Returns:
            Tensor: The tensor after Maxpooling layer.
        """
        return self.model(x)


class GCAModule(nn.Module):
    """GCAModule in MASTER.

    Args:
        in_channels (int): Channels of input tensor.
        ratio (float): Scale ratio of in_channels.
        n_head (int): Numbers of attention head.
        pooling_type (str): Spatial pooling type. Options are [``avg``,
            ``att``].
        scale_attn (bool): Whether to scale the attention map. Defaults to
            False.
        fusion_type (str): Fusion type of input and context. Options are
            [``channel_add``, ``channel_mul``, ``channel_concat``].
    """

    def __init__(self, in_channels, ratio, n_head, pooling_type='att', scale_attn=False, fusion_type='channel_add', **kwargs):
        super(GCAModule, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert in_channels % n_head == 0 and in_channels >= 8
        self.n_head = n_head
        self.in_channels = in_channels
        self.ratio = ratio
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.scale_attn = scale_attn
        self.single_header_inplanes = int(in_channels / n_head)
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            self.cat_conv = nn.Conv2d(2 * self.in_channels, self.in_channels, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.in_channels, self.planes, kernel_size=1), nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True), nn.Conv2d(self.planes, self.in_channels, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            x = x.view(batch * self.n_head, self.single_header_inplanes, height, width)
            input_x = x
            input_x = input_x.view(batch * self.n_head, self.single_header_inplanes, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch * self.n_head, 1, height * width)
            if self.scale_attn and self.n_head > 1:
                context_mask = context_mask / torch.sqrt(self.single_header_inplanes)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, self.n_head * self.single_header_inplanes, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.fusion_type == 'channel_mul':
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            channel_concat_term = self.channel_concat_conv(context)
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape
            out = torch.cat([out, channel_concat_term.expand(-1, -1, H, W)], dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.in_channels, H, W])
            out = nn.functional.relu(out)
        return out


class LocalizationNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from input
    (img_width x img_height)

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        num_img_channel (int): Number of channels of the input image.
    """

    def __init__(self, num_fiducial, num_img_channel):
        super().__init__()
        self.num_fiducial = num_fiducial
        self.num_img_channel = num_img_channel
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.num_img_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.AdaptiveAvgPool2d(1))
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.num_fiducial * 2)
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(num_fiducial / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(num_fiducial / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Batch input image of shape
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Predicted coordinates of fiducial points for input batch.
            The shape is :math:`(N, F, 2)` where :math:`F` is ``num_fiducial``.
        """
        batch_size = batch_img.size(0)
        features = self.conv(batch_img).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.num_fiducial, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multiplying T with P.

    Args:
        num_fiducial (int): Number of fiducial points of TPS-STN.
        rectified_img_size (tuple(int, int)):
            Size :math:`(H_r, W_r)` of the rectified image.
    """

    def __init__(self, num_fiducial, rectified_img_size):
        """Generate P_hat and inv_delta_C for later."""
        super().__init__()
        self.eps = 1e-06
        self.rectified_img_height = rectified_img_size[0]
        self.rectified_img_width = rectified_img_size[1]
        self.num_fiducial = num_fiducial
        self.C = self._build_C(self.num_fiducial)
        self.P = self._build_P(self.rectified_img_width, self.rectified_img_height)
        self.register_buffer('inv_delta_C', torch.tensor(self._build_inv_delta_C(self.num_fiducial, self.C)).float())
        self.register_buffer('P_hat', torch.tensor(self._build_P_hat(self.num_fiducial, self.C, self.P)).float())

    def _build_C(self, num_fiducial):
        """Return coordinates of fiducial points in rectified_img; C."""
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(num_fiducial / 2))
        ctrl_pts_y_top = -1 * np.ones(int(num_fiducial / 2))
        ctrl_pts_y_bottom = np.ones(int(num_fiducial / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_inv_delta_C(self, num_fiducial, C):
        """Return inv_delta_C which is needed to calculate T."""
        hat_C = np.zeros((num_fiducial, num_fiducial), dtype=float)
        for i in range(0, num_fiducial):
            for j in range(i, num_fiducial):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = hat_C ** 2 * np.log(hat_C)
        delta_C = np.concatenate([np.concatenate([np.ones((num_fiducial, 1)), C, hat_C], axis=1), np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1), np.concatenate([np.zeros((1, 3)), np.ones((1, num_fiducial))], axis=1)], axis=0)
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, rectified_img_width, rectified_img_height):
        rectified_img_grid_x = (np.arange(-rectified_img_width, rectified_img_width, 2) + 1.0) / rectified_img_width
        rectified_img_grid_y = (np.arange(-rectified_img_height, rectified_img_height, 2) + 1.0) / rectified_img_height
        P = np.stack(np.meshgrid(rectified_img_grid_x, rectified_img_grid_y), axis=2)
        return P.reshape([-1, 2])

    def _build_P_hat(self, num_fiducial, C, P):
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, num_fiducial, 1))
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat

    def build_P_prime(self, batch_C_prime, device='cuda'):
        """Generate Grid from batch_C_prime [batch_size x num_fiducial x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float()), dim=1)
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)
        return batch_P_prime


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input):
        return


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BidirectionalLSTM,
     lambda: ([], {'nIn': 4, 'nHidden': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'input_dims': [4, 4], 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeluNew,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GraphConv,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (LocalizationNetwork,
     lambda: ([], {'num_fiducial': 4, 'num_img_channel': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Maxpool2d,
     lambda: ([], {'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MeanAggregator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PositionAwareLayer,
     lambda: ([], {'dim_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 512])], {}),
     False),
    (ScaledDotProductAttention,
     lambda: ([], {'temperature': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_BatchNormXd,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_open_mmlab_mmocr(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

