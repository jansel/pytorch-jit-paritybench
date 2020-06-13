import sys
_module = sys.modules[__name__]
del sys
annotate = _module
annotate_attributes = _module
annotate_cars = _module
annotate_cars_mirrors = _module
annotate_crashables = _module
annotate_current_lane = _module
annotate_lanes_same_direction = _module
annotate_speed_segments = _module
annotate_steering_wheel = _module
annotate_street_boundaries = _module
annotate_street_markings = _module
common = _module
config = _module
lib = _module
actions = _module
ets2game = _module
ets2window = _module
plotting = _module
pykeylogger = _module
replay_memory = _module
rewards = _module
screenshot = _module
speed = _module
states = _module
steering_wheel = _module
util = _module
windowhandling = _module
add_steering_wheel_to_replay_memory = _module
collect_experiences_supervised = _module
collect_speed_segments = _module
recalculate_rewards = _module
batching = _module
generate_video_frames = _module
models = _module
plans = _module
train = _module
visualization = _module
train_semisupervised = _module
compress_annotations = _module
dataset = _module
models = _module
train = _module
train_steering_wheel = _module
models = _module
train = _module

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


import numpy as np


import torch


import torch.nn.functional as F


import collections


from scipy import misc


from scipy import ndimage


import torch.nn as nn


from torch.autograd.function import InplaceFunction


import torch.optim as optim


from torch.autograd import Variable


import random


import math


import copy


def add_white_noise(x, std, training):
    """Layer that adds white/gaussian noise to its input."""
    if training:
        noise = Variable(x.data.new().resize_as_(x.data).normal_(mean=0,
            std=std), volatile=x.volatile, requires_grad=False).type_as(x)
        x = x + noise
    return x


def init_weights(module):
    """Weight initializer."""
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Embedder(nn.Module):
    """Model that converts the (B, C, H, W) outputs from the semi-supervised
    embedder to vectors (B, S).
    This also adds information regarding speed, steering wheel, gear and
    previous actions."""

    def __init__(self):
        super(Embedder, self).__init__()

        def identity(v):
            return lambda x: x
        bn2d = nn.InstanceNorm2d
        bn1d = identity
        self.nb_previous_images = 2
        self.emb_sup_c1 = nn.Conv2d(512, 1024, kernel_size=3, padding=0,
            stride=1)
        self.emb_sup_c1_bn = bn2d(1024)
        self.emb_sup_c1_sd = nn.Dropout2d(0.0)
        self.emb_add_fc1 = nn.Linear(self.nb_previous_images + 1 + (self.
            nb_previous_images + 1) + (self.nb_previous_images + 1) + (self
            .nb_previous_images + 1) + self.nb_previous_images * 9, 128)
        self.emb_add_fc1_bn = bn1d(128)
        self.emb_fc1 = nn.Linear(1024 * 3 + 128, 512)
        self.emb_fc1_bn = bn1d(512)
        init_weights(self)

    def forward(self, embeddings_supervised, speeds, is_reverse,
        steering_wheel, steering_wheel_raw, multiactions_vecs):

        def act(x):
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x_emb_sup = embeddings_supervised
        x_emb_sup = act(self.emb_sup_c1_sd(self.emb_sup_c1_bn(self.
            emb_sup_c1(x_emb_sup))))
        x_emb_sup = x_emb_sup.view(-1, 1024 * 1 * 3)
        x_emb_sup = add_white_noise(x_emb_sup, 0.005, self.training)
        x_emb_add = torch.cat([speeds, is_reverse, steering_wheel,
            steering_wheel_raw, multiactions_vecs], 1)
        x_emb_add = act(self.emb_add_fc1_bn(self.emb_add_fc1(x_emb_add)))
        x_emb_add = add_white_noise(x_emb_add, 0.005, self.training)
        x_emb = torch.cat([x_emb_sup, x_emb_add], 1)
        x_emb = F.dropout(x_emb, p=0.05, training=self.training)
        embs = F.relu(self.emb_fc1_bn(self.emb_fc1(x_emb)))
        embs = add_white_noise(embs, 0.005, True)
        return embs

    def forward_dict(self, embeddings_supervised, inputs_reinforced_add):
        return self.forward(embeddings_supervised, inputs_reinforced_add[
            'speeds'], inputs_reinforced_add['is_reverse'],
            inputs_reinforced_add['steering_wheel'], inputs_reinforced_add[
            'steering_wheel_raw'], inputs_reinforced_add['multiactions_vecs'])


class DirectRewardPredictor(nn.Module):
    """Model that predicts the direct reward of a state.
    For (s, a, r), (s', a', r') this would predict r in s'. It does not predict
    r', because that is dependent on a'.
    The prediction happens via a softmax over bins instead of via regression.
    """

    def __init__(self, nb_bins):
        super(DirectRewardPredictor, self).__init__()

        def identity(v):
            return lambda x: x
        bn2d = nn.InstanceNorm2d
        bn1d = identity
        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = bn1d(128)
        self.fc2 = nn.Linear(128, nb_bins)
        init_weights(self)

    def forward(self, embeddings, softmax):

        def act(x):
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = act(self.fc1_bn(self.fc1(embeddings)))
        x = add_white_noise(x, 0.005, self.training)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        if softmax:
            return F.softmax(x)
        else:
            return x


class IndirectRewardPredictor(nn.Module):
    """Model that predicts indirect rewards / Q-values.
    For
      (s, a, r), (s', a', r'), ...
    that is
      r + g^1*r' + g^2*r'' + ...
    in state s,
    where g is gamma (discount term).
    The prediction happens via regression.
    It predicts one value per action a.
    The prediction is split into V (mean value of a state) and A (advantage
    of each action).
    """

    def __init__(self):
        super(IndirectRewardPredictor, self).__init__()

        def identity(v):
            return lambda x: x
        bn2d = nn.InstanceNorm2d
        bn1d = identity
        self.fc1 = nn.Linear(512, 128)
        self.fc1_bn = bn1d(128)
        self.fc_v = nn.Linear(128, 1)
        self.fc_advantage = nn.Linear(128, 9)
        init_weights(self)

    def forward(self, embeddings, return_v_adv=False):

        def act(x):
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)
        B, _ = embeddings.size()
        x = act(self.fc1_bn(self.fc1(embeddings)))
        x = add_white_noise(x, 0.005, self.training)
        x = F.dropout(x, p=0.1, training=self.training)
        x_v = self.fc_v(x)
        x_v_expanded = x_v.expand(B, 9)
        x_adv = self.fc_advantage(x)
        x_adv_mean = x_adv.mean(dim=1)
        x_adv_mean = x_adv_mean.expand(B, 9)
        x_adv = x_adv - x_adv_mean
        x = x_v_expanded + x_adv
        if return_v_adv:
            return x, (x_v, x_adv)
        else:
            return x


class AEDecoder(nn.Module):
    """Decoder part of an autoencoder that takes an embedding vector
    and converts it into an image."""

    def __init__(self):
        super(AEDecoder, self).__init__()

        def identity(v):
            return lambda x: x
        bn2d = nn.InstanceNorm2d
        bn1d = identity
        self.ae_fc1 = nn.Linear(512, 128 * 3 * 5)
        self.ae_fc1_bn = bn1d(128 * 3 * 5)
        self.ae_c1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.ae_c1_bn = bn2d(128)
        self.ae_c2 = nn.Conv2d(128, 128, kernel_size=3, padding=(0, 1))
        self.ae_c2_bn = bn2d(128)
        self.ae_c3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.ae_c3_bn = bn2d(128)
        self.ae_c4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        init_weights(self)

    def forward(self, embedding):

        def act(x):
            return F.relu(x, inplace=True)

        def up(x):
            m = nn.UpsamplingNearest2d(scale_factor=2)
            return m(x)
        x_ae = embedding
        x_ae = act(self.ae_fc1_bn(self.ae_fc1(x_ae)))
        x_ae = x_ae.view(-1, 128, 3, 5)
        x_ae = up(x_ae)
        x_ae = act(self.ae_c1_bn(self.ae_c1(x_ae)))
        x_ae = up(x_ae)
        x_ae = act(self.ae_c2_bn(self.ae_c2(x_ae)))
        x_ae = F.pad(x_ae, (0, 0, 1, 0))
        x_ae = up(x_ae)
        x_ae = act(self.ae_c3_bn(self.ae_c3(x_ae)))
        x_ae = up(x_ae)
        x_ae = F.pad(x_ae, (0, 0, 1, 0))
        x_ae = F.sigmoid(self.ae_c4(x_ae))
        return x_ae


def to_variable(inputs, volatile=False, requires_grad=True):
    if volatile:
        make_var = lambda x: Variable(x, volatile=True)
    else:
        make_var = lambda x: Variable(x, requires_grad=requires_grad)
    if isinstance(inputs, np.ndarray):
        return make_var(torch.from_numpy(inputs))
    elif isinstance(inputs, list):
        return [make_var(torch.from_numpy(el)) for el in inputs]
    elif isinstance(inputs, tuple):
        return [make_var(torch.from_numpy(el)) for el in inputs]
    elif isinstance(inputs, dict):
        return dict([(key, make_var(torch.from_numpy(inputs[key]))) for key in
            inputs])
    else:
        raise Exception('unknown input %s' % (type(inputs),))


class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()

        def identity(_):
            return lambda x: x
        bn2d = nn.InstanceNorm2d
        bn1d = nn.InstanceNorm1d
        self.nb_previous_images = 2
        self.emb_c1_curr = nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2
            )
        self.emb_c1_bn_curr = bn2d(128)
        self.emb_c1_sd_curr = nn.Dropout2d(0.0)
        self.emb_c2_curr = nn.Conv2d(128, 128, kernel_size=3, padding=1,
            stride=1)
        self.emb_c2_bn_curr = bn2d(128)
        self.emb_c2_sd_curr = nn.Dropout2d(0.0)
        self.emb_c3_curr = nn.Conv2d(128, 256, kernel_size=3, padding=1,
            stride=1)
        self.emb_c3_bn_curr = bn2d(256)
        self.emb_c3_sd_curr = nn.Dropout2d(0.0)
        self.emb_c1_prev = nn.Conv2d(self.nb_previous_images, 64,
            kernel_size=3, padding=1, stride=1)
        self.emb_c1_bn_prev = bn2d(64)
        self.emb_c1_sd_prev = nn.Dropout2d(0.0)
        self.emb_c2_prev = nn.Conv2d(64, 128, kernel_size=3, padding=1,
            stride=1)
        self.emb_c2_bn_prev = bn2d(128)
        self.emb_c2_sd_prev = nn.Dropout2d(0.0)
        self.emb_c4 = nn.Conv2d(256 + 128 + 4, 256, kernel_size=5, padding=
            2, stride=2)
        self.emb_c4_bn = bn2d(256)
        self.emb_c4_sd = nn.Dropout2d(0.0)
        self.emb_c5 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2)
        self.emb_c5_bn = bn2d(256)
        self.emb_c5_sd = nn.Dropout2d(0.0)
        self.emb_c6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.emb_c6_bn = bn2d(512)
        self.emb_c6_sd = nn.Dropout2d(0.0)
        self.emb_c7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.emb_c7_bn = bn2d(512)
        self.emb_c7_sd = nn.Dropout2d(0.0)
        self.maps_c1 = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.maps_c1_bn = bn2d(256)
        self.maps_c2 = nn.Conv2d(256, 256, kernel_size=5, padding=(0, 2))
        self.maps_c2_bn = bn2d(256)
        self.maps_c3 = nn.Conv2d(256, 8 + 3 + self.nb_previous_images + 1 +
            1, kernel_size=5, padding=2)
        atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
        ma_size = 9 + 9 + 9 + 9
        flipped_size = self.nb_previous_images
        self.vec_fc1 = nn.Linear(512 * 3 * 5, atts_size + ma_size +
            flipped_size, bias=False)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def downscale(self, img):
        return ia.imresize_single_image(img, (train.MODEL_HEIGHT, train.
            MODEL_WIDTH), interpolation='cubic')

    def downscale_prev(self, img):
        return ia.imresize_single_image(img, (train.MODEL_PREV_HEIGHT,
            train.MODEL_PREV_WIDTH), interpolation='cubic')

    def embed_state(self, previous_states, state, volatile=False,
        requires_grad=True, gpu=-1):
        prev_scrs = [self.downscale_prev(s.screenshot_rs) for s in
            previous_states]
        prev_scrs_y = [cv2.cvtColor(scr, cv2.COLOR_RGB2GRAY) for scr in
            prev_scrs]
        inputs = np.array(self.downscale(state.screenshot_rs), dtype=np.float32
            )
        inputs = inputs / 255.0
        inputs = inputs.transpose((2, 0, 1))
        inputs = inputs[np.newaxis, ...]
        inputs = to_cuda(to_variable(inputs, volatile=volatile,
            requires_grad=requires_grad), gpu)
        inputs_prev = np.dstack(prev_scrs_y)
        inputs_prev = inputs_prev.astype(np.float32) / 255.0
        inputs_prev = inputs_prev.transpose((2, 0, 1))
        inputs_prev = inputs_prev[np.newaxis, ...]
        inputs_prev = to_cuda(to_variable(inputs_prev, volatile=volatile,
            requires_grad=requires_grad), gpu)
        return self.embed(inputs, inputs_prev)

    def embed(self, inputs, inputs_prev):
        return self.forward(inputs, inputs_prev, only_embed=True)

    def forward(self, inputs, inputs_prev, only_embed=False):

        def act(x):
            return F.relu(x, inplace=True)

        def lrelu(x, negative_slope=0.2):
            return F.leaky_relu(x, negative_slope=negative_slope, inplace=True)

        def up(x, f=2):
            m = nn.UpsamplingNearest2d(scale_factor=f)
            return m(x)

        def maxp(x):
            return F.max_pool2d(x, 2)
        B = inputs.size(0)
        pos_x = np.tile(np.linspace(0, 1, 40).astype(np.float32).reshape(1,
            1, 40), (B, 1, 23, 1))
        pos_x = np.concatenate([pos_x, np.fliplr(pos_x)], axis=1)
        pos_y = np.tile(np.linspace(0, 1, 23).astype(np.float32).reshape(1,
            23, 1), (B, 1, 1, 40))
        pos_y = np.concatenate([pos_y, np.flipud(pos_y)], axis=1)
        """
        print(pos_x_curr[0, 0, 0, 0])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(1/4))-1])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(2/4))-1])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(3/4))-1])
        print(pos_x_curr[0, 0, 0, int(MODEL_WIDTH*(4/4))-1])
        from scipy import misc
        misc.imshow(
            np.vstack([
                np.squeeze(pos_x_curr[0].transpose((1, 2, 0))) * 255,
                np.squeeze(pos_y_curr[0].transpose((1, 2, 0))) * 255
            ])
        )
        """
        pos_x = to_cuda(to_variable(pos_x, volatile=inputs.volatile,
            requires_grad=inputs.requires_grad), Config.GPU)
        pos_y = to_cuda(to_variable(pos_y, volatile=inputs.volatile,
            requires_grad=inputs.requires_grad), Config.GPU)
        x_emb0_curr = inputs
        x_emb1_curr = lrelu(self.emb_c1_sd_curr(self.emb_c1_bn_curr(self.
            emb_c1_curr(x_emb0_curr))))
        x_emb2_curr = lrelu(self.emb_c2_sd_curr(self.emb_c2_bn_curr(self.
            emb_c2_curr(x_emb1_curr))))
        x_emb2_curr = F.pad(x_emb2_curr, (0, 0, 1, 0))
        x_emb2_curr = maxp(x_emb2_curr)
        x_emb3_curr = lrelu(self.emb_c3_sd_curr(self.emb_c3_bn_curr(self.
            emb_c3_curr(x_emb2_curr))))
        x_emb0_prev = inputs_prev
        x_emb1_prev = lrelu(self.emb_c1_sd_prev(self.emb_c1_bn_prev(self.
            emb_c1_prev(x_emb0_prev))))
        x_emb1_prev = F.pad(x_emb1_prev, (0, 0, 1, 0))
        x_emb1_prev = maxp(x_emb1_prev)
        x_emb2_prev = lrelu(self.emb_c2_sd_prev(self.emb_c2_bn_prev(self.
            emb_c2_prev(x_emb1_prev))))
        x_emb3 = torch.cat([x_emb3_curr, x_emb2_prev, pos_x, pos_y], 1)
        x_emb3 = F.pad(x_emb3, (0, 0, 1, 0))
        x_emb4 = lrelu(self.emb_c4_sd(self.emb_c4_bn(self.emb_c4(x_emb3))))
        x_emb5 = lrelu(self.emb_c5_sd(self.emb_c5_bn(self.emb_c5(x_emb4))))
        x_emb6 = lrelu(self.emb_c6_sd(self.emb_c6_bn(self.emb_c6(x_emb5))))
        x_emb7 = lrelu(self.emb_c7_sd(self.emb_c7_bn(self.emb_c7(x_emb6))))
        x_emb = x_emb7
        if only_embed:
            return x_emb
        else:
            x_maps = x_emb
            x_maps = up(x_maps, 4)
            x_maps = lrelu(self.maps_c1_bn(self.maps_c1(x_maps)))
            x_maps = up(x_maps, 4)
            x_maps = lrelu(self.maps_c2_bn(self.maps_c2(x_maps)))
            x_maps = F.pad(x_maps, (0, 0, 1, 0))
            x_maps = up(x_maps)
            x_maps = F.sigmoid(self.maps_c3(x_maps))
            ae_size = 3 + self.nb_previous_images
            x_grids = x_maps[:, 0:8, (...)]
            x_ae = x_maps[:, 8:8 + ae_size, (...)]
            x_flow = x_maps[:, 8 + ae_size:8 + ae_size + 1, (...)]
            x_canny = x_maps[:, 8 + ae_size + 1:8 + ae_size + 2, (...)]
            x_vec = x_emb
            x_vec = x_vec.view(-1, 512 * 3 * 5)
            x_vec = F.dropout(x_vec, p=0.5, training=self.training)
            x_vec = F.sigmoid(self.vec_fc1(x_vec))
            atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
            ma_size = 9 + 9 + 9 + 9
            x_atts = x_vec[:, 0:atts_size]
            x_ma = x_vec[:, atts_size:atts_size + ma_size]
            x_flipped = x_vec[:, atts_size + ma_size:]
            return (x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped,
                x_emb)

    def predict_grids(self, inputs, inputs_prev):
        x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb = (self
            .forward(inputs, inputs_prev))
        return x_grids


class PredictorWithShortcuts(nn.Module):

    def __init__(self):
        super(PredictorWithShortcuts, self).__init__()

        def identity(_):
            return lambda x: x
        bn2d = nn.InstanceNorm2d
        bn1d = nn.InstanceNorm1d
        self.nb_previous_images = 2
        self.emb_c1_curr = nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2
            )
        self.emb_c1_bn_curr = bn2d(128)
        self.emb_c1_sd_curr = nn.Dropout2d(0.0)
        self.emb_c2_curr = nn.Conv2d(128, 128, kernel_size=3, padding=1,
            stride=1)
        self.emb_c2_bn_curr = bn2d(128)
        self.emb_c2_sd_curr = nn.Dropout2d(0.0)
        self.emb_c3_curr = nn.Conv2d(128, 256, kernel_size=3, padding=1,
            stride=1)
        self.emb_c3_bn_curr = bn2d(256)
        self.emb_c3_sd_curr = nn.Dropout2d(0.0)
        self.emb_c1_prev = nn.Conv2d(self.nb_previous_images, 64,
            kernel_size=3, padding=1, stride=1)
        self.emb_c1_bn_prev = bn2d(64)
        self.emb_c1_sd_prev = nn.Dropout2d(0.0)
        self.emb_c2_prev = nn.Conv2d(64, 128, kernel_size=3, padding=1,
            stride=1)
        self.emb_c2_bn_prev = bn2d(128)
        self.emb_c2_sd_prev = nn.Dropout2d(0.0)
        self.emb_c4 = nn.Conv2d(256 + 128 + 4, 256, kernel_size=5, padding=
            2, stride=2)
        self.emb_c4_bn = bn2d(256)
        self.emb_c4_sd = nn.Dropout2d(0.0)
        self.emb_c5 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2)
        self.emb_c5_bn = bn2d(256)
        self.emb_c5_sd = nn.Dropout2d(0.0)
        self.emb_c6 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.emb_c6_bn = bn2d(512)
        self.emb_c6_sd = nn.Dropout2d(0.0)
        self.emb_c7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.emb_c7_bn = bn2d(512)
        self.emb_c7_sd = nn.Dropout2d(0.0)
        self.maps_c1 = nn.Conv2d(512 + 256, 256, kernel_size=5, padding=2)
        self.maps_c1_bn = bn2d(256)
        self.maps_c2 = nn.Conv2d(256 + 128, 256, kernel_size=5, padding=(0, 2))
        self.maps_c2_bn = bn2d(256)
        self.maps_c3 = nn.Conv2d(256 + 3, 8 + 3 + self.nb_previous_images +
            1 + 1, kernel_size=5, padding=2)
        atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
        ma_size = 9 + 9 + 9 + 9
        flipped_size = self.nb_previous_images
        self.vec_fc1 = nn.Linear(512 * 3 * 5, atts_size + ma_size +
            flipped_size, bias=False)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def downscale(self, img):
        return ia.imresize_single_image(img, (train.MODEL_HEIGHT, train.
            MODEL_WIDTH), interpolation='cubic')

    def downscale_prev(self, img):
        return ia.imresize_single_image(img, (train.MODEL_PREV_HEIGHT,
            train.MODEL_PREV_WIDTH), interpolation='cubic')

    def embed_state(self, previous_states, state, volatile=False,
        requires_grad=True, gpu=-1):
        prev_scrs = [self.downscale_prev(s.screenshot_rs) for s in
            previous_states]
        prev_scrs_y = [cv2.cvtColor(scr, cv2.COLOR_RGB2GRAY) for scr in
            prev_scrs]
        inputs = np.array(self.downscale(state.screenshot_rs), dtype=np.float32
            )
        inputs = inputs / 255.0
        inputs = inputs.transpose((2, 0, 1))
        inputs = inputs[np.newaxis, ...]
        inputs = to_cuda(to_variable(inputs, volatile=volatile,
            requires_grad=requires_grad), gpu)
        inputs_prev = np.dstack(prev_scrs_y)
        inputs_prev = inputs_prev.astype(np.float32) / 255.0
        inputs_prev = inputs_prev.transpose((2, 0, 1))
        inputs_prev = inputs_prev[np.newaxis, ...]
        inputs_prev = to_cuda(to_variable(inputs_prev, volatile=volatile,
            requires_grad=requires_grad), gpu)
        return self.embed(inputs, inputs_prev)

    def embed(self, inputs, inputs_prev):
        return self.forward(inputs, inputs_prev, only_embed=True)

    def forward(self, inputs, inputs_prev, only_embed=False):

        def act(x):
            return F.relu(x, inplace=True)

        def lrelu(x, negative_slope=0.2):
            return F.leaky_relu(x, negative_slope=negative_slope, inplace=True)

        def up(x, f=2):
            m = nn.UpsamplingNearest2d(scale_factor=f)
            return m(x)

        def maxp(x):
            return F.max_pool2d(x, 2)
        B = inputs.size(0)
        pos_x = np.tile(np.linspace(0, 1, 40).astype(np.float32).reshape(1,
            1, 40), (B, 1, 23, 1))
        pos_x = np.concatenate([pos_x, np.fliplr(pos_x)], axis=1)
        pos_y = np.tile(np.linspace(0, 1, 23).astype(np.float32).reshape(1,
            23, 1), (B, 1, 1, 40))
        pos_y = np.concatenate([pos_y, np.flipud(pos_y)], axis=1)
        pos_x = to_cuda(to_variable(pos_x, volatile=inputs.volatile,
            requires_grad=inputs.requires_grad), Config.GPU)
        pos_y = to_cuda(to_variable(pos_y, volatile=inputs.volatile,
            requires_grad=inputs.requires_grad), Config.GPU)
        x_emb0_curr = inputs
        x_emb1_curr = lrelu(self.emb_c1_sd_curr(self.emb_c1_bn_curr(self.
            emb_c1_curr(x_emb0_curr))))
        x_emb2_curr = lrelu(self.emb_c2_sd_curr(self.emb_c2_bn_curr(self.
            emb_c2_curr(x_emb1_curr))))
        x_emb2_curr = F.pad(x_emb2_curr, (0, 0, 1, 0))
        x_emb2_curr_pool = maxp(x_emb2_curr)
        x_emb3_curr = lrelu(self.emb_c3_sd_curr(self.emb_c3_bn_curr(self.
            emb_c3_curr(x_emb2_curr_pool))))
        x_emb0_prev = inputs_prev
        x_emb1_prev = lrelu(self.emb_c1_sd_prev(self.emb_c1_bn_prev(self.
            emb_c1_prev(x_emb0_prev))))
        x_emb1_prev = F.pad(x_emb1_prev, (0, 0, 1, 0))
        x_emb1_prev = maxp(x_emb1_prev)
        x_emb2_prev = lrelu(self.emb_c2_sd_prev(self.emb_c2_bn_prev(self.
            emb_c2_prev(x_emb1_prev))))
        x_emb3 = torch.cat([x_emb3_curr, x_emb2_prev, pos_x, pos_y], 1)
        x_emb3 = F.pad(x_emb3, (0, 0, 1, 0))
        x_emb4 = lrelu(self.emb_c4_sd(self.emb_c4_bn(self.emb_c4(x_emb3))))
        x_emb5 = lrelu(self.emb_c5_sd(self.emb_c5_bn(self.emb_c5(x_emb4))))
        x_emb6 = lrelu(self.emb_c6_sd(self.emb_c6_bn(self.emb_c6(x_emb5))))
        x_emb7 = lrelu(self.emb_c7_sd(self.emb_c7_bn(self.emb_c7(x_emb6))))
        x_emb = x_emb7
        if only_embed:
            return x_emb
        else:
            x_maps = x_emb
            x_maps = up(x_maps, 4)
            x_maps = lrelu(self.maps_c1_bn(self.maps_c1(torch.cat([x_maps,
                x_emb4], 1))))
            x_maps = up(x_maps, 4)
            x_maps = lrelu(self.maps_c2_bn(self.maps_c2(torch.cat([x_maps,
                F.pad(x_emb2_curr, (0, 0, 1, 1))], 1))))
            x_maps = F.pad(x_maps, (0, 0, 1, 0))
            x_maps = up(x_maps)
            x_maps = F.sigmoid(self.maps_c3(torch.cat([x_maps, inputs], 1)))
            ae_size = 3 + self.nb_previous_images
            x_grids = x_maps[:, 0:8, (...)]
            x_ae = x_maps[:, 8:8 + ae_size, (...)]
            x_flow = x_maps[:, 8 + ae_size:8 + ae_size + 1, (...)]
            x_canny = x_maps[:, 8 + ae_size + 1:8 + ae_size + 2, (...)]
            x_vec = x_emb
            x_vec = x_vec.view(-1, 512 * 3 * 5)
            x_vec = F.dropout(x_vec, p=0.5, training=self.training)
            x_vec = F.sigmoid(self.vec_fc1(x_vec))
            atts_size = 10 + 7 + 3 + 5 + 8 + 4 + 4 + 4 + 3
            ma_size = 9 + 9 + 9 + 9
            x_atts = x_vec[:, 0:atts_size]
            x_ma = x_vec[:, atts_size:atts_size + ma_size]
            x_flipped = x_vec[:, atts_size + ma_size:]
            return (x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped,
                x_emb)

    def predict_grids(self, inputs, inputs_prev):
        x_ae, x_grids, x_atts, x_ma, x_flow, x_canny, x_flipped, x_emb = (self
            .forward(inputs, inputs_prev))
        return x_grids


ANGLE_BIN_SIZE = 5


GPU = 0


class SteeringWheelTrackerCNNModel(nn.Module):

    def __init__(self):
        super(SteeringWheelTrackerCNNModel, self).__init__()
        self.c1 = nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=1)
        self.fc1 = nn.Linear(32 * (32 // 4) * (64 // 4), 16)
        self.fc2 = nn.Linear(16, 360 // ANGLE_BIN_SIZE)

    def forward(self, inputs, softmax=False):
        x = inputs
        x = F.relu(self.c1(x))
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 32 * (32 // 4) * (64 // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if softmax:
            x = F.softmax(x)
        return x

    def forward_image(self, subimg, softmax=False, volatile=False,
        requires_grad=True, gpu=GPU):
        subimg = np.float32([subimg / 255]).transpose((0, 3, 1, 2))
        subimg = to_cuda(to_variable(subimg, volatile=volatile,
            requires_grad=requires_grad), GPU)
        return self.forward(subimg, softmax=softmax)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_aleju_self_driving_truck(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(AEDecoder(*[], **{}), [torch.rand([512, 512])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DirectRewardPredictor(*[], **{'nb_bins': 4}), [torch.rand([512, 512]), 0], {})

    @_fails_compile()
    def test_002(self):
        self._check(SteeringWheelTrackerCNNModel(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

