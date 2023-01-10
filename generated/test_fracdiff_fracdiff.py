import sys
_module = sys.modules[__name__]
del sys
conf = _module
nky = _module
spx = _module
fracdiff = _module
fdiff = _module
sklearn = _module
fracdiffstat = _module
stat = _module
tol = _module
functional = _module
module = _module
tests = _module
issues = _module
test_32 = _module
test_fracdiff = _module
test_fracdiffstat = _module
test_sklearn = _module
test_stat = _module
test_tol = _module
test_fdiff = _module
test_howto = _module
test_torch = _module
test_torch_importerror = _module

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


from typing import Optional


import torch


from torch import Tensor


from torch.nn import Module


import numpy as np


from torch.testing import assert_close


class Fracdiff(Module):
    """A ``torch.nn.Module`` to compute fractional differentiation.

    Args:
        d (float): The order of differentiation.
        dim (int, default=-1): The dimension to differentiate.
            Currently, only the last dimension is supported.
        window (int, default=10): The window size for fractional differentiation.
        mode (str, default="same"): "same" or "valid".
            See :func:`fracdiff.fdiff` for details.

    Shape:
        - input: :math:`(N, *, L_{\\mathrm{in}})`, where where :math:`*` means any
          number of additional dimensions.
        - output: :math:`(N, *, L_{\\mathrm{out}})`, where :math:`L_{\\mathrm{out}}`
          is given by :math:`L_{\\mathrm{in}}` if `mode="same"` and
          :math:`L_{\\mathrm{in}} - \\mathrm{window} + 1` if `mode="valid"`.

    Examples:
        >>> from fracdiff.torch import Fracdiff
        >>> m = Fracdiff(0.5)
        >>> m
        Fracdiff(0.5, dim=-1, window=10, mode='same')
        >>> input = torch.arange(10).reshape(2, 5)
        >>> m(input)
        tensor([[0.0000, 1.0000, 1.5000, 1.8750, 2.1875],
                [5.0000, 3.5000, 3.3750, 3.4375, 3.5547]])
    """

    def __init__(self, d: float, dim: int=-1, window: int=10, mode: str='same') ->None:
        super().__init__()
        self.d = d
        self.dim = dim
        self.window = window
        self.mode = mode

    def extra_repr(self) ->str:
        params = str(self.d), f'dim={self.dim}', f'window={self.window}', f"mode='{self.mode}'"
        return ', '.join(params)

    def forward(self, input: Tensor, prepend: Optional[Tensor]=None, append: Optional[Tensor]=None) ->Tensor:
        """Apply fractional differentiation.

        Args:
            input (torch.Tensor): The input tensor.
            prepend (torch.Tensor, optional): The tensor to prepend
                to `input` along `self.dim` before computing the differentiation.
                Their dimensions must be equivalent to that of `input`,
                and their shapes must match `input`'s shape except on `dim`.
            append (torch.Tensor, optional): The tensor to append.

        Returns:
            torch.Tensor
        """
        return functional.fdiff(input, self.d, dim=self.dim, window=self.window, mode=self.mode, prepend=prepend, append=append)

