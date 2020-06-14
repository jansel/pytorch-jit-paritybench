import sys
_module = sys.modules[__name__]
del sys
run = _module
softsplat = _module

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


import numpy


import re


def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]
    while True:
        objMatch = re.search('(SIZE_)([0-4])(\\()([^\\)]*)(\\))', strKernel)
        if objMatch is None:
            break
        intArg = int(objMatch.group(2))
        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()
        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace(
            '}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for
            intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+',
            strIndex) + ')')
    while True:
        objMatch = re.search('(VALUE_)([0-4])(\\()([^\\)]+)(\\))', strKernel)
        if objMatch is None:
            break
        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')
        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = [('((' + strArgs[intArg + 1].replace('{', '(').replace(
            '}', ')').strip() + ')*' + str(intStrides[intArg]) + ')') for
            intArg in range(intArgs)]
        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' +
            str.join('+', strIndex) + ']')
    return strKernel


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sniklaus_softmax_splatting(_paritybench_base):
    pass
