import copy
import os
import re
import unittest
from functools import lru_cache

import torch
from torch.testing._internal.jit_utils import JitTestCase


class DummyBlock(torch.nn.ReLU):
    expansion = 1

    def __str__(self):
        return "_mock_layer()"

    __repr__ = __str__


@lru_cache(None)
def patch_functional():
    """
    Some projects import both `torch.functional` and `torch.nn.functional`
    as "F".  This hackily combines the two to make those projects work.
    """
    f1 = torch.functional
    f2 = torch.nn.functional
    names1 = {name for name in dir(f1) if re.match(r"^[a-z]", name)}
    names2 = {name for name in dir(f2) if re.match(r"^[a-z]", name)}
    for name in names1 - names2:
        setattr(f2, name, getattr(f1, name))
    for name in names2 - names1:
        setattr(f1, name, getattr(f2, name))


def _mock_layer(in_features=None, out_features=None, *args, **kwargs):
    if in_features and out_features:
        return torch.nn.Linear(in_features, out_features, **kwargs)
    return DummyBlock()


class _mock_config(dict):
    __getattr__ = dict.__getitem__


def _fails_compile():
    if os.environ.get('TEST_ALL'):
        return lambda x: x
    return unittest.skip("jit compile fails")


class _paritybench_base(JitTestCase):
    def _check(self, module, init_args, forward_args, compiles):
        args, kwargs = init_args()
        script = module(*args, **kwargs)

        try:
            script.eval()
        except:
            pass

        args, kwargs = forward_args()
        result1 = script(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        result2 = script(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        if os.environ.get('TEST_PY_ONLY'):
            return

        if os.environ.get('TEST_WORKING_ONLY') and not compiles:
            raise unittest.SkipTest("jit compile fails")

        jit_script = torch.jit.script(script)

        if os.environ.get('TEST_COMPILE_ONLY'):
            return

        result3 = jit_script(*args, **kwargs)

        if os.environ.get('TEST_RUN_ONLY'):
            return

        try:
            self.assertEqual(result1, result2)
        except AssertionError:
            return  # output is not deterministic
        self.assertEqual(result2, result3)
