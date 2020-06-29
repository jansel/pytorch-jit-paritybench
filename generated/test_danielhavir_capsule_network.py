import sys
_module = sys.modules[__name__]
del sys
capsules = _module
loss = _module
main = _module
model = _module
trainer = _module

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


import torch.nn as nn


import torch.nn.functional as F


from numpy import prod


import torch.optim as optim


from time import time


def squash(s, dim=-1):
    """
	"Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
	Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
	
	Args:
		s: 	Vector before activation
		dim:	Dimension along which to calculate the norm
	
	Returns:
		Squashed vector
	"""
    squared_norm = torch.sum(s ** 2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm
        ) + 1e-08)


class PrimaryCapsules(nn.Module):

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size=9,
        stride=2, padding=0):
        """
		Initialize the layer.

		Args:
			in_channels: 	Number of input channels.
			out_channels: 	Number of output channels.
			dim_caps:		Dimensionality, i.e. length, of the output capsule vector.
		
		"""
        super(PrimaryCapsules, self).__init__()
        self.dim_caps = dim_caps
        self._caps_channel = int(out_channels / dim_caps)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), self._caps_channel, out.size(2), out.
            size(3), self.dim_caps)
        out = out.view(out.size(0), -1, self.dim_caps)
        return squash(out)


class RoutingCapsules(nn.Module):

    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing,
        device: torch.device):
        """
		Initialize the layer.

		Args:
			in_dim: 		Dimensionality (i.e. length) of each capsule vector.
			in_caps: 		Number of input capsules if digits layer.
			num_caps: 		Number of capsules in the capsule layer
			dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
			num_routing:	Number of iterations during routing algorithm		
		"""
        super(RoutingCapsules, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps, in_caps,
            dim_caps, in_dim))

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = ' -> '
        res = self.__class__.__name__ + '('
        res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
        res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
        res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
        res = res + 'num_routing=' + str(self.num_routing) + ')'
        res = res + line + ')'
        return res

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        temp_u_hat = u_hat.detach()
        """
		Procedure 1: Routing algorithm
		"""
        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1)
        for route_iter in range(self.num_routing - 1):
            c = F.softmax(b, dim=1)
            s = (c * temp_u_hat).sum(dim=2)
            v = squash(s)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        v = squash(s)
        return v


class MarginLoss(nn.Module):

    def __init__(self, size_average=False, loss_lambda=0.5):
        """
		Margin loss for digit existence
		Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2
		
		Args:
			size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
			loss_lambda: parameter for down-weighting the loss for missing digits
		"""
        super(MarginLoss, self).__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, labels):
        L_k = labels * F.relu(self.m_plus - inputs) ** 2 + self.loss_lambda * (
            1 - labels) * F.relu(inputs - self.m_minus) ** 2
        L_k = L_k.sum(dim=1)
        if self.size_average:
            return L_k.mean()
        else:
            return L_k.sum()


class CapsuleLoss(nn.Module):

    def __init__(self, loss_lambda=0.5, recon_loss_scale=0.0005,
        size_average=False):
        """
		Combined margin loss and reconstruction loss. Margin loss see above.
		Sum squared error (SSE) was used as a reconstruction loss.
		
		Args:
			recon_loss_scale: 	param for scaling down the the reconstruction loss
			size_average:		if True, reconstruction loss becomes MSE instead of SSE
		"""
        super(CapsuleLoss, self).__init__()
        self.size_average = size_average
        self.margin_loss = MarginLoss(size_average=size_average,
            loss_lambda=loss_lambda)
        self.reconstruction_loss = nn.MSELoss(size_average=size_average)
        self.recon_loss_scale = recon_loss_scale

    def forward(self, inputs, labels, images, reconstructions):
        margin_loss = self.margin_loss(inputs, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        caps_loss = margin_loss + self.recon_loss_scale * reconstruction_loss
        return caps_loss


class CapsuleNetwork(nn.Module):

    def __init__(self, img_shape, channels, primary_dim, num_classes,
        out_dim, num_routing, device: torch.device, kernel_size=9):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.device = device
        self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size, stride=
            1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.primary = caps.PrimaryCapsules(channels, channels, primary_dim,
            kernel_size)
        primary_caps = int(channels / primary_dim * (img_shape[1] - 2 * (
            kernel_size - 1)) * (img_shape[2] - 2 * (kernel_size - 1)) / 4)
        self.digits = caps.RoutingCapsules(primary_dim, primary_caps,
            num_classes, out_dim, num_routing, device=self.device)
        self.decoder = nn.Sequential(nn.Linear(out_dim * num_classes, 512),
            nn.ReLU(inplace=True), nn.Linear(512, 1024), nn.ReLU(inplace=
            True), nn.Linear(1024, int(prod(img_shape))), nn.Sigmoid())

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.primary(out)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_classes)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)
        reconstructions = self.decoder((out * y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)
        return preds, reconstructions


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_danielhavir_capsule_network(_paritybench_base):
    pass
    def test_000(self):
        self._check(CapsuleLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(MarginLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(PrimaryCapsules(*[], **{'in_channels': 4, 'out_channels': 4, 'dim_caps': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(RoutingCapsules(*[], **{'in_dim': 4, 'in_caps': 4, 'num_caps': 4, 'dim_caps': 4, 'num_routing': 4, 'device': 4}), [torch.rand([4, 4, 4])], {})

