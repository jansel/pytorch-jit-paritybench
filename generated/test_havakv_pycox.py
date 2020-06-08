import sys
_module = sys.modules[__name__]
del sys
pycox = _module
datasets = _module
_dataset_loader = _module
from_deepsurv = _module
from_kkbox = _module
from_rdatasets = _module
from_simulations = _module
evaluation = _module
admin = _module
concordance = _module
eval_surv = _module
ipcw = _module
metrics = _module
models = _module
base = _module
bce_surv = _module
cox = _module
cox_cc = _module
cox_time = _module
data = _module
deephit = _module
interpolation = _module
logistic_hazard = _module
loss = _module
mtlr = _module
pc_hazard = _module
pmf = _module
utils = _module
preprocessing = _module
discretization = _module
feature_transforms = _module
label_transforms = _module
simulations = _module
discrete_logit_hazard = _module
relative_risk = _module
setup = _module
test_admin = _module
test_bce_surv = _module
test_cox = _module
test_cox_cc = _module
test_cox_time = _module
test_deephit = _module
test_interpolation = _module
test_logistic_hazard = _module
test_loss = _module
test_models_utils = _module
test_mtlr = _module
test_pc_hazard = _module
test_pmf = _module
utils_model_testing = _module
test_utils = _module

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


import torch


import warnings


import numpy as np


from torch import nn


from typing import Tuple


from torch import Tensor


import torch.nn.functional as F


class MLPVanillaCoxTime(nn.Module):
    """A version of torchtuples.practical.MLPVanilla that works for CoxTime.
    The difference is that it takes `time` as an additional input and removes the output bias and
    output activation.
    """

    def __init__(self, in_features, num_nodes, batch_norm=True, dropout=
        None, activation=nn.ReLU, w_init_=lambda w: nn.init.kaiming_normal_
        (w, nonlinearity='relu')):
        super().__init__()
        in_features += 1
        out_features = 1
        output_activation = None
        output_bias = False
        self.net = tt.practical.MLPVanilla(in_features, num_nodes,
            out_features, batch_norm, dropout, activation,
            output_activation, output_bias, w_init_)

    def forward(self, input, time):
        input = torch.cat([input, time], dim=1)
        return self.net(input)


class MixedInputMLPCoxTime(nn.Module):
    """A version of torchtuples.practical.MixedInputMLP that works for CoxTime.
    The difference is that it takes `time` as an additional input and removes the output bias and
    output activation.
    """

    def __init__(self, in_features, num_embeddings, embedding_dims,
        num_nodes, batch_norm=True, dropout=None, activation=nn.ReLU,
        dropout_embedding=0.0, w_init_=lambda w: nn.init.kaiming_normal_(w,
        nonlinearity='relu')):
        super().__init__()
        in_features += 1
        out_features = 1
        output_activation = None
        output_bias = False
        self.net = tt.practical.MixedInputMLP(in_features, num_embeddings,
            embedding_dims, num_nodes, out_features, batch_norm, dropout,
            activation, dropout_embedding, output_activation, output_bias,
            w_init_)

    def forward(self, input_numeric, input_categoric, time):
        input_numeric = torch.cat([input_numeric, time], dim=1)
        return self.net(input_numeric, input_categoric)


class _Loss(torch.nn.Module):
    """Generic loss function.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """

    def __init__(self, reduction: str='mean') ->None:
        super().__init__()
        self.reduction = reduction


def cox_cc_loss(g_case: Tensor, g_control: Tensor, shrink: float=0.0, clamp:
    Tuple[float, float]=(-3e+38, 80.0)) ->Tensor:
    """Torch loss function for the Cox case-control models.
    For only one control, see `cox_cc_loss_single_ctrl` instead.
    
    Arguments:
        g_case {torch.Tensor} -- Result of net(input_case)
        g_control {torch.Tensor} -- Results of [net(input_ctrl1), net(input_ctrl2), ...]
    
    Keyword Arguments:
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    
    Returns:
        [type] -- [description]
    """
    control_sum = 0.0
    shrink_control = 0.0
    if g_case.shape != g_control[0].shape:
        raise ValueError(
            f'Need `g_case` and `g_control[0]` to have same shape. Got {g_case.shape}'
             + f' and {g_control[0].shape}')
    for ctr in g_control:
        shrink_control += ctr.abs().mean()
        ctr = ctr - g_case
        ctr = torch.clamp(ctr, *clamp)
        control_sum += torch.exp(ctr)
    loss = torch.log(1.0 + control_sum)
    shrink_zero = shrink * (g_case.abs().mean() + shrink_control) / len(
        g_control)
    return torch.mean(loss) + shrink_zero.abs()


def cox_cc_loss_single_ctrl(g_case: Tensor, g_control: Tensor, shrink:
    float=0.0) ->Tensor:
    """CoxCC and CoxTime loss, but with only a single control.
    """
    loss = F.softplus(g_control - g_case).mean()
    if shrink != 0:
        loss += shrink * (g_case.abs().mean() + g_control.abs().mean())
    return loss


def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float=1e-07
    ) ->Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(rac{h_i}{\\sum_{j \\in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


class CoxPHLossSorted(torch.nn.Module):
    """Loss for CoxPH.
    Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(rac{h_i}{\\sum_{j \\in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_h: Tensor, events: Tensor) ->Tensor:
        return cox_ph_loss_sorted(log_h, events)


def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps:
    float=1e-07) ->Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(rac{h_i}{\\sum_{j \\in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(rac{h_i}{\\sum_{j \\in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor
        ) ->Tensor:
        return cox_ph_loss(log_h, durations, events)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_havakv_pycox(_paritybench_base):
    pass

    def test_000(self):
        self._check(CoxPHLossSorted(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CoxPHLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
