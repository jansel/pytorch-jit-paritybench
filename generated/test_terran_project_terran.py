import sys
_module = sys.modules[__name__]
del sys
conf = _module
match = _module
video = _module
setup = _module
terran = _module
checkpoint = _module
cli = _module
defaults = _module
face = _module
detection = _module
retinaface = _module
anchors = _module
model = _module
wrapper = _module
recognition = _module
arcface = _module
model = _module
wrapper = _module
io = _module
image = _module
reader = _module
writer = _module
pose = _module
openpose = _module
model = _module
wrapper = _module
tracking = _module
vis = _module
cairo = _module
pillow = _module

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


import math


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torchvision.ops import nms


from sklearn.preprocessing import normalize


from enum import Enum


from collections import OrderedDict


class ConvSepBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1, return_both=False):
        """Building block for base network.

        Consists of common Conv, BN and ReLU sequence, followed by the same
        sequence but with a separable Conv.

        Paramters
        ---------
        return_both : bool
            Return the outputs of both inner components, the conv and the
            separable blocks. We do this because it's the conv block the one
            that's used as feature pyramid.

        """
        super().__init__()
        self.return_both = return_both
        self.conv_block = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=1, bias=False), nn.BatchNorm2d(out_c, momentum=0.9, eps=1e-05), nn.ReLU())
        self.sep_block = nn.Sequential(nn.Conv2d(out_c, out_c, 3, stride=stride, padding=1, groups=out_c, bias=False), nn.BatchNorm2d(out_c, momentum=0.9, eps=1e-05), nn.ReLU())

    def forward(self, x):
        conv = self.conv_block(x)
        sep = self.sep_block(conv)
        if self.return_both:
            out = conv, sep
        else:
            out = sep
        return out


class BaseNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.first_conv_block = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(8, momentum=0.9, eps=1e-05), nn.ReLU(), nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8, bias=False), nn.BatchNorm2d(8, momentum=0.9, eps=1e-05), nn.ReLU())
        self.scales = nn.ModuleList([nn.Sequential(ConvSepBlock(8, 16, stride=2), ConvSepBlock(16, 32), ConvSepBlock(32, 32, stride=2), ConvSepBlock(32, 64), ConvSepBlock(64, 64, stride=2, return_both=True)), nn.Sequential(ConvSepBlock(64, 128), ConvSepBlock(128, 128), ConvSepBlock(128, 128), ConvSepBlock(128, 128), ConvSepBlock(128, 128), ConvSepBlock(128, 128, stride=2, return_both=True))])
        self.final_conv = nn.Sequential(ConvSepBlock(128, 256), nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256, momentum=0.9, eps=1e-05), nn.ReLU())

    def forward(self, x):
        out = self.first_conv_block(x)
        feature_maps = []
        for scale in self.scales:
            conv, out = scale(out)
            feature_maps.append(conv)
        out = self.final_conv(out)
        feature_maps.append(out)
        return feature_maps


class ContextModule(nn.Module):
    """Context module to expand the receptive field of the feature map.

    Every point in the feature map will be a mixture of a 3x3, a 5x5 and a 7x7
    receptive fields. The first 128 channels correspond to the 3x3 one, and the
    remaining 64 and 64 to the rest.
    """

    def __init__(self):
        super().__init__()
        self.context_3x3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32, momentum=0.9, eps=2e-05), nn.ReLU())
        self.dimension_reducer = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.BatchNorm2d(16, momentum=0.9, eps=2e-05), nn.ReLU())
        self.context_5x5 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16, momentum=0.9, eps=2e-05), nn.ReLU())
        self.context_7x7 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16, momentum=0.9, eps=2e-05), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16, momentum=0.9, eps=2e-05), nn.ReLU())

    def forward(self, x):
        red = self.dimension_reducer(x)
        ctx_3x3 = self.context_3x3(x)
        ctx_5x5 = self.context_5x5(red)
        ctx_7x7 = self.context_7x7(red)
        out = torch.cat([ctx_3x3, ctx_5x5, ctx_7x7], dim=1)
        return out


class PyramidRefiner(nn.Module):
    """Refines the feature pyramids from the base network into usable form.

    Normalizes channel sizes, mixes them up, and runs them through the context
    module.
    """

    def __init__(self):
        super().__init__()
        self.conv_stride8 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64, momentum=0.9, eps=2e-05), nn.ReLU())
        self.conv_stride16 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64, momentum=0.9, eps=2e-05), nn.ReLU())
        self.conv_stride32 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64, momentum=0.9, eps=2e-05), nn.ReLU())
        self.aggr_stride8 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64, momentum=0.9, eps=2e-05), nn.ReLU())
        self.aggr_stride16 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64, momentum=0.9, eps=2e-05), nn.ReLU())
        self.context_stride8 = ContextModule()
        self.context_stride16 = ContextModule()
        self.context_stride32 = ContextModule()

    def forward(self, x):
        """Forward pass for refiner.

        Expects `x` to be an array of three tensors, one per feature. Returns
        the same.
        """
        stride8, stride16, stride32 = x
        proc_stride8 = self.conv_stride8(stride8)
        proc_stride16 = self.conv_stride16(stride16)
        proc_stride32 = self.conv_stride32(stride32)
        ups_stride32 = F.interpolate(proc_stride32, scale_factor=2)[:, :, :proc_stride16.shape[2], :proc_stride16.shape[3]]
        proc_stride16 = self.aggr_stride16(proc_stride16 + ups_stride32)
        ups_stride16 = F.interpolate(proc_stride16, scale_factor=2)[:, :, :proc_stride8.shape[2], :proc_stride8.shape[3]]
        proc_stride8 = self.aggr_stride8(proc_stride8 + ups_stride16)
        ctx_stride8 = self.context_stride8(proc_stride8)
        ctx_stride16 = self.context_stride16(proc_stride16)
        ctx_stride32 = self.context_stride32(proc_stride32)
        return [ctx_stride8, ctx_stride16, ctx_stride32]


class OutputsPredictor(nn.Module):
    """Uses the feature pyramid to predict the final deltas for the network."""

    def __init__(self):
        super().__init__()
        self.num_anchors = 2
        self.cls_stride8 = nn.Conv2d(64, 2 * self.num_anchors, 1)
        self.cls_stride16 = nn.Conv2d(64, 2 * self.num_anchors, 1)
        self.cls_stride32 = nn.Conv2d(64, 2 * self.num_anchors, 1)
        self.bbox_stride8 = nn.Conv2d(64, 4 * self.num_anchors, 1)
        self.bbox_stride16 = nn.Conv2d(64, 4 * self.num_anchors, 1)
        self.bbox_stride32 = nn.Conv2d(64, 4 * self.num_anchors, 1)
        self.landmark_stride8 = nn.Conv2d(64, 10 * self.num_anchors, 1)
        self.landmark_stride16 = nn.Conv2d(64, 10 * self.num_anchors, 1)
        self.landmark_stride32 = nn.Conv2d(64, 10 * self.num_anchors, 1)

    def forward(self, x):
        """Forward pass for output predictor.

        Expects `x` to hold one feature map per stride.
        """
        stride8, stride16, stride32 = x
        cls_score8 = self.cls_stride8(stride8)
        cls_score16 = self.cls_stride16(stride16)
        cls_score32 = self.cls_stride32(stride32)
        N, A, H, W = cls_score8.shape
        cls_prob8 = F.softmax(cls_score8.view(N, 2, -1, W), dim=1).view(N, A, H, W)
        N, A, H, W = cls_score16.shape
        cls_prob16 = F.softmax(cls_score16.view(N, 2, -1, W), dim=1).view(N, A, H, W)
        N, A, H, W = cls_score32.shape
        cls_prob32 = F.softmax(cls_score32.view(N, 2, -1, W), dim=1).view(N, A, H, W)
        bbox_pred8 = self.bbox_stride8(stride8)
        bbox_pred16 = self.bbox_stride16(stride16)
        bbox_pred32 = self.bbox_stride32(stride32)
        landmark_pred8 = self.landmark_stride8(stride8)
        landmark_pred16 = self.landmark_stride16(stride16)
        landmark_pred32 = self.landmark_stride32(stride32)
        return [cls_prob32, bbox_pred32, landmark_pred32, cls_prob16, bbox_pred16, landmark_pred16, cls_prob8, bbox_pred8, landmark_pred8]


def anchors_plane(anchor_ref, feat_h, feat_w, stride):
    """Builds the anchors plane for the given feature map shape.

    Based on an anchor reference, reproduces it at every point of the feature
    map, according to the specified stride. `anchor_ref` is set in real-image
    coordinates, so we must know what the stride for the current feature map
    is.

    Parameters
    ----------
    anchor_ref : torch.Tensor of size (A, 4)
        Coordinates for each of the `A` anchors, centered at the origin.
    feat_h : int
        Height of the feature map.
    feat_w : int
        Width of the feature map.
    stride : int
        Number of pixels every which to apply the anchor reference.

    Returns
    -------
    torch.Tensor of size (feat_h * feat_w * A, 4)
        The dtype and device of the returned tensor is based on `anchor_ref`.

    """
    device = anchor_ref.device
    dtype = anchor_ref.dtype
    shift_y, shift_x = torch.meshgrid(torch.arange(feat_h, dtype=dtype, device=device) * stride, torch.arange(feat_w, dtype=dtype, device=device) * stride)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)
    anchors = (anchor_ref[None, ...] + shifts[:, None, :]).reshape(-1, 4)
    return anchors


def decode_bboxes(anchors, deltas):
    """Apply the bbox delta predictions on the base anchor coordinates.

    Paramters
    ---------
    anchors : torch.Tensor of shape (A, 4)
    deltas : torch.Tensor of shape (N, A, 4)

    Returns
    -------
    torch.Tensor of shape (N, A, 4)
        Adjusted bounding boxes.

    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)
    dx = deltas[..., 0]
    dy = deltas[..., 1]
    dw = deltas[..., 2]
    dh = deltas[..., 3]
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    pred = deltas
    pred[..., 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred[..., 1] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred[..., 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred[..., 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)
    return pred


def decode_landmarks(anchors, deltas):
    """Apply the landmark delta predictions on the base anchor coordinates.

    Paramters
    ---------
    anchors : torch.Tensor of shape (A, 4)
    deltas : torch.Tensor of shape (N, A, 5, 2)

    Returns
    -------
    torch.Tensor of shape (N, A, 5, 2)
        Adjusted landmark coordinates.

    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)
    pred = deltas
    for i in range(5):
        pred[..., i, 0] = deltas[..., i, 0] * widths + ctr_x
        pred[..., i, 1] = deltas[..., i, 1] * heights + ctr_y
    return pred


default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def _mkanchors(ws, hs, ctr_x, ctr_y):
    """Given a vector of widths (ws) and heights (hs) around a center
    (ctr_x, ctr_y), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack([ctr_x - 0.5 * (ws - 1), ctr_y - 0.5 * (hs - 1), ctr_x + 0.5 * (ws - 1), ctr_y + 0.5 * (hs - 1)])
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    ctr_x = anchor[0] + 0.5 * (w - 1)
    ctr_y = anchor[1] + 0.5 * (h - 1)
    return w, h, ctr_x, ctr_y


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, ctr_x, ctr_y = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, ctr_x, ctr_y)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, ctr_x, ctr_y = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, ctr_x, ctr_y)
    return anchors


def generate_anchors(base_size, ratios, scales, stride):
    """Generate an anchor reference for the given properties."""
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
    return anchors


def generate_anchor_reference(settings=None, device=default_device):
    """Generate anchor reference per stride, for the given settings."""
    feature_strides = sorted(settings.keys(), reverse=True)
    anchors = []
    for stride in feature_strides:
        base_size = settings[stride]['base_size']
        ratios = np.array(settings[stride]['ratios'])
        scales = np.array(settings[stride]['scales'])
        anchor = torch.as_tensor(generate_anchors(base_size, ratios, scales, stride), dtype=torch.float32, device=default_device)
        anchors.append(anchor)
    return anchors


def _make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = torch.nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, torch.nn.ReLU(inplace=True)))
    return torch.nn.Sequential(OrderedDict(layers))


class BodyPoseModel(torch.nn.Module):

    def __init__(self):
        super(BodyPoseModel, self).__init__()
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1', 'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2', 'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1', 'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict({'conv1_1': [3, 64, 3, 1, 1], 'conv1_2': [64, 64, 3, 1, 1], 'pool1_stage1': [2, 2, 0], 'conv2_1': [64, 128, 3, 1, 1], 'conv2_2': [128, 128, 3, 1, 1], 'pool2_stage1': [2, 2, 0], 'conv3_1': [128, 256, 3, 1, 1], 'conv3_2': [256, 256, 3, 1, 1], 'conv3_3': [256, 256, 3, 1, 1], 'conv3_4': [256, 256, 3, 1, 1], 'pool3_stage1': [2, 2, 0], 'conv4_1': [256, 512, 3, 1, 1], 'conv4_2': [512, 512, 3, 1, 1], 'conv4_3_CPM': [512, 256, 3, 1, 1], 'conv4_4_CPM': [256, 128, 3, 1, 1]})
        block1_1 = OrderedDict({'conv5_1_CPM_L1': [128, 128, 3, 1, 1], 'conv5_2_CPM_L1': [128, 128, 3, 1, 1], 'conv5_3_CPM_L1': [128, 128, 3, 1, 1], 'conv5_4_CPM_L1': [128, 512, 1, 1, 0], 'conv5_5_CPM_L1': [512, 38, 1, 1, 0]})
        block1_2 = OrderedDict({'conv5_1_CPM_L2': [128, 128, 3, 1, 1], 'conv5_2_CPM_L2': [128, 128, 3, 1, 1], 'conv5_3_CPM_L2': [128, 128, 3, 1, 1], 'conv5_4_CPM_L2': [128, 512, 1, 1, 0], 'conv5_5_CPM_L2': [512, 19, 1, 1, 0]})
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2
        self.model0 = _make_layers(block0, no_relu_layers)
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict({('Mconv1_stage%d_L1' % i): [185, 128, 7, 1, 3], ('Mconv2_stage%d_L1' % i): [128, 128, 7, 1, 3], ('Mconv3_stage%d_L1' % i): [128, 128, 7, 1, 3], ('Mconv4_stage%d_L1' % i): [128, 128, 7, 1, 3], ('Mconv5_stage%d_L1' % i): [128, 128, 7, 1, 3], ('Mconv6_stage%d_L1' % i): [128, 128, 1, 1, 0], ('Mconv7_stage%d_L1' % i): [128, 38, 1, 1, 0]})
            blocks['block%d_2' % i] = OrderedDict({('Mconv1_stage%d_L2' % i): [185, 128, 7, 1, 3], ('Mconv2_stage%d_L2' % i): [128, 128, 7, 1, 3], ('Mconv3_stage%d_L2' % i): [128, 128, 7, 1, 3], ('Mconv4_stage%d_L2' % i): [128, 128, 7, 1, 3], ('Mconv5_stage%d_L2' % i): [128, 128, 7, 1, 3], ('Mconv6_stage%d_L2' % i): [128, 128, 1, 1, 0], ('Mconv7_stage%d_L2' % i): [128, 19, 1, 1, 0]})
        for k in blocks.keys():
            blocks[k] = _make_layers(blocks[k], no_relu_layers)
        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']
        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']

    def forward(self, x):
        out1 = self.model0(x)
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        return out6_1, out6_2


CHECKPOINT_PATH = 'checkpoints'


def get_terran_home(create_if_missing=True):
    """Returns Terran's homedir.

    Defaults to `DEFAULT_TERRAN_HOME`, which is `~/.terran`, but can be
    overridden with the `TERRAN_HOME` environment variable.

    Returns
    -------
    pathlib.Path
        Path pointing to the base Terran directory.

    """
    path = Path(os.environ.get('TERRAN_HOME', DEFAULT_TERRAN_HOME)).expanduser()
    if create_if_missing and not path.exists():
        path.mkdir(exist_ok=True)
    return path


def get_checkpoints_directory():
    """Returns checkpoint directory within Terran's homedir.

    If the path doesn't exists, creates it.

    Returns
    -------
    pathlib.Path
        Path pointing to the checkpoints directory.

    """
    path = get_terran_home() / CHECKPOINT_PATH
    path.mkdir(exist_ok=True)
    return path


def download_remote_checkpoint(db, checkpoint):
    if checkpoint['local_path'] and checkpoint['local_path'].exists():
        click.echo(f"Checkpoint file already present at {checkpoint['local_path']}. 'If you're running into any issues, try issuing a `terran checkpoint delete {checkpoint['id']}` trying attempting again.")
        return
    file_name = f"{checkpoint['id']}.pth"
    tempdir = tempfile.mkdtemp()
    path = Path(tempdir) / file_name
    response = requests.get(checkpoint['url'], stream=True)
    if response.status_code != 200:
        raise ValueError(f"Invalid checkpoint URL {checkpoint['url']}")
    length = int(response.headers.get('Content-Length'))
    chunk_size = 16 * 1024
    progressbar = click.progressbar(response.iter_content(chunk_size=chunk_size), length=length / chunk_size, label='Downloading checkpoint...')
    with open(path, 'wb') as f:
        with progressbar as content:
            for chunk in content:
                f.write(chunk)
    new_path = get_checkpoints_directory() / file_name
    shutil.move(path, new_path)
    checkpoint['status'] = 'DOWNLOADED'
    checkpoint['local_path'] = new_path
    shutil.rmtree(tempdir)
    click.echo('Checkpoint downloaded successfully.')


def get_checkpoint_by_class(db, class_path):
    """Returns checkpoint entry in `db` indicated by `class_path`.

    Parameters
    ----------
    class_path : str
        Fully specified path to class (e.g. `terran.pose.openpose.OpenPose`)
        of the model to get the checkpoint for.

    Returns
    -------
    Dict
        Checkpoint data contained in the database.

    """
    selected = [c for c in db['checkpoints'] if c['class'] == class_path]
    if len(selected) < 1:
        return None
    if len(selected) > 1:
        click.echo(f"Multiple checkpoints found for '{class_path}' ({len(selected)}). Returning first.")
    return selected[0]


CHECKPOINTS = [{'id': 'b5d77fff', 'name': 'RetinaFace', 'description': 'RetinaFace with mnet backbone.', 'task': 'face-detection', 'class': 'terran.face.detection.retinaface.RetinaFace', 'alias': 'gpu-realtime', 'default': True, 'performance': 1.0, 'evaluation': {'value': 0.76, 'metric': 'mAP', 'is_reported': False}, 'url': 'https://github.com/nagitsu/terran/releases/download/0.0.1/retinaface-mnet.pth'}, {'id': 'd206e4b0', 'name': 'ArcFace', 'description': 'ArcFace with Resnet 100 backbone.', 'task': 'face-recognition', 'class': 'terran.face.recognition.arcface.ArcFace', 'alias': 'gpu-realtime', 'default': True, 'performance': 0.9, 'evaluation': {'value': 0.8, 'metric': 'accuracy', 'is_reported': False}, 'url': 'https://github.com/nagitsu/terran/releases/download/0.0.1/arcface-resnet100.pth'}, {'id': '11a769ad', 'name': 'OpenPose', 'description': 'OpenPose with VGG backend, 2017 version. Has some modifications, improving computational efficiency by giving up mAP.', 'task': 'pose-estimation', 'class': 'terran.pose.openpose.OpenPose', 'alias': 'gpu-realtime', 'default': True, 'performance': 1.8, 'evaluation': {'value': 0.65, 'metric': 'mAP', 'is_reported': True}, 'url': 'https://github.com/nagitsu/terran/releases/download/0.0.1/openpose-body.pth'}]


def read_checkpoint_db():
    """Reads the checkpoints database file from disk."""
    local_checkpoints = set(path.stem for path in get_checkpoints_directory().glob('*.pth'))
    checkpoints = [{'status': 'DOWNLOADED' if checkpoint['id'] in local_checkpoints else 'NOT_DOWNLOADED', 'local_path': get_checkpoints_directory() / f"{checkpoint['id']}.pth" if checkpoint['id'] in local_checkpoints else None, **checkpoint} for checkpoint in CHECKPOINTS]
    return {'checkpoints': checkpoints}


def get_checkpoint_path(model_class_path, prompt=True):
    """Returns the local path to the model's weights.

    Goes through the list of checkpoints and returns the local path to the
    weights of the modell specified by `model_class`. If the weights are not
    downloaded, downloads them first.

    Parameters
    ----------
    model_class_path : str
        Fully specified path to class (e.g. `terran.pose.openpose.OpenPose`)
        of the model to get the checkpoint for.
    prompt : boolean
        If `True` and the checkpoint is not yet downloaded, prompt to download.

    Returns
    -------
    pathlib.Path
        Path to the `.pth` file containing the weights for the model.

    Raises
    ------
    ValueError
        If checkpoint is not found or is found but not downloaded, either due
        to aborting the prompt or disabling it in the first place.

    """
    db = read_checkpoint_db()
    checkpoint = get_checkpoint_by_class(db, model_class_path)
    can_prompt = sys.stdout.isatty()
    if not checkpoint:
        raise ValueError('Checkpoint not found.')
    if checkpoint['status'] == 'NOT_DOWNLOADED':
        if prompt and can_prompt:
            try:
                click.confirm('Checkpoint not present locally. Want to download it?', abort=True)
            except Exception:
                click.echo('Checkpoint not present locally. Downloading it')
        download_remote_checkpoint(db, checkpoint)
    return checkpoint['local_path']


def load_model():
    model = BodyPoseModel()
    model.load_state_dict(torch.load(get_checkpoint_path('terran.pose.openpose.OpenPose')))
    model.eval()
    return model


class RetinaFace:

    def __init__(self, device=default_device, nms_threshold=0.4):
        self.device = device
        self.nms_threshold = nms_threshold
        self.feature_strides = [32, 16, 8]
        self.anchor_settings = {(8): {'scales': (2, 1), 'base_size': 16, 'ratios': (1,)}, (16): {'scales': (8, 4), 'base_size': 16, 'ratios': (1,)}, (32): {'scales': (32, 16), 'base_size': 16, 'ratios': (1,)}}
        self.anchor_references = dict(zip(self.feature_strides, generate_anchor_reference(settings=self.anchor_settings, device=self.device)))
        self.num_anchors_per_stride = {stride: anchors.shape[0] for stride, anchors in self.anchor_references.items()}
        self.model = load_model()

    def call(self, images, threshold=0.5):
        """Run the detection.

        `images` is a (N, H, W, C)-shaped array (np.float32).

        (Padding must be performed outside.)
        """
        H, W = images.shape[1:3]
        data = torch.as_tensor(images, device=self.device, dtype=torch.float32).permute(0, 3, 1, 2).flip(1)
        with torch.no_grad():
            output = self.model(data)
        anchors_per_stride = {}
        for stride in self.feature_strides:
            height = math.ceil(H / stride)
            width = math.ceil(W / stride)
            anchor_ref = self.anchor_references[stride]
            anchors = anchors_plane(anchor_ref, height, width, stride)
            anchors_per_stride[stride] = anchors
        proposals_list = []
        scores_list = []
        landmarks_list = []
        for stride_idx, stride in enumerate(self.feature_strides):
            idx = stride_idx * 3
            anchors = anchors_per_stride[stride]
            A = self.num_anchors_per_stride[stride]
            N = output[idx].shape[0]
            scores = output[idx]
            scores = scores[:, A:, :, :]
            scores = scores.permute(0, 2, 3, 1).reshape(N, -1)
            bbox_deltas = output[idx + 1]
            bbox_pred_len = bbox_deltas.shape[1] // A
            bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape((N, -1, bbox_pred_len))
            landmark_deltas = output[idx + 2]
            landmark_pred_len = landmark_deltas.shape[1] // A
            landmark_deltas = landmark_deltas.permute(0, 2, 3, 1).reshape((N, -1, 5, landmark_pred_len // 5))
            proposals = decode_bboxes(anchors, bbox_deltas)
            landmarks = decode_landmarks(anchors, landmark_deltas)
            scores_list.append(scores)
            proposals_list.append(proposals)
            landmarks_list.append(landmarks)
        batch_scores = torch.cat(scores_list, dim=1)
        batch_proposals = torch.cat(proposals_list, dim=1)
        batch_landmarks = torch.cat(landmarks_list, dim=1)
        batch_objects = []
        for image_idx in range(images.shape[0]):
            scores = batch_scores[image_idx]
            proposals = batch_proposals[image_idx]
            landmarks = batch_landmarks[image_idx]
            order = torch.where(scores >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]
            landmarks = landmarks[order, :]
            if proposals.shape[0] == 0:
                batch_objects.append([])
                continue
            order = scores.argsort(descending=True)
            proposals = proposals[order]
            scores = scores[order]
            landmarks = landmarks[order]
            keep = nms(proposals, scores, self.nms_threshold)
            proposals = proposals[keep].numpy()
            scores = scores[keep].numpy()
            landmarks = landmarks[keep].numpy()
            batch_objects.append([{'bbox': b, 'landmarks': l, 'score': s} for s, b, l in zip(scores, proposals, landmarks)])
        return batch_objects


class Unit(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dimensions_match = in_c == out_c and stride == 1
        self.body = nn.Sequential(nn.BatchNorm2d(in_c, momentum=0.9, eps=2e-05), nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c, momentum=0.9, eps=2e-05), nn.PReLU(num_parameters=out_c), nn.Conv2d(out_c, out_c, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_c, momentum=0.9, eps=2e-05))
        if not self.dimensions_match:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c, momentum=0.9, eps=2e-05))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        body = self.body(x)
        shortcut = self.shortcut(x)
        return body + shortcut


class FaceResNet100(nn.Module):

    def __init__(self):
        super().__init__()
        self.units_per_stage = [3, 13, 30, 3]
        self.channels = [64, 64, 128, 256, 512]
        self.mean = 127.5
        self.std = 0.0078125
        self.initial_layer = nn.Sequential(nn.Conv2d(3, self.channels[0], 3, padding=1, bias=False), nn.BatchNorm2d(self.channels[0], momentum=0.9, eps=2e-05), nn.PReLU(num_parameters=self.channels[0]))
        stages = []
        for stage_idx, num_units in enumerate(self.units_per_stage):
            prev_c = self.channels[stage_idx]
            curr_c = self.channels[stage_idx + 1]
            num_units = self.units_per_stage[stage_idx]
            units = nn.Sequential(*[Unit(prev_c, curr_c, stride=2), *[Unit(curr_c, curr_c) for _ in range(num_units - 1)]])
            stages.append(units)
        self.stages = nn.ModuleList(stages)
        self.final_layer = nn.Sequential(nn.BatchNorm2d(self.channels[-1], momentum=0.9, eps=2e-05), nn.Dropout(0.4), nn.Flatten(), nn.Linear(7 * 7 * 512, 512), nn.BatchNorm1d(512, momentum=0.9, eps=2e-05))

    def forward(self, x):
        preprocessed = (x - self.mean) * self.std
        out = self.initial_layer(preprocessed)
        for stage in self.stages:
            out = stage(out)
        out = self.final_layer(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (BodyPoseModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ContextModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ConvSepBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Unit,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_terran_project_terran(_paritybench_base):
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

