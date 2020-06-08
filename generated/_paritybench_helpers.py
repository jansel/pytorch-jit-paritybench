
import torch, unittest, copy, os
from torch.testing._internal.jit_utils import JitTestCase


def _mock_layer(in_features=None, out_features=None, bias=True):
    if in_features and out_features:
        return torch.nn.Linear(in_features, out_features, bias)
    return torch.nn.ReLU()


class _mock_config(dict):
    __getattr__ = dict.__getitem__


def _fails_compile():
    if os.environ.get('TEST_ALL'):
        return lambda x: x
    return unittest.skip("jit compile fails")


class _paritybench_base(JitTestCase):
    def _check(self, script, args, kwargs):
        try:
            script.eval()
        except:
            pass
        result1 = script(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        result2 = script(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        if os.environ.get('TEST_PY_ONLY'):
            return
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
