import os
import re
import unittest
from functools import lru_cache

import torch
import torch._dynamo
import torch._inductor
from torch.testing._internal.jit_utils import JitTestCase
from torch._dynamo.testing import same

from paritybench.utils import INDUCTOR_TOL, get_cosine_and_fp64_outputs, \
    patch_torch_manual_seed, reset_rng_state, wrap_args, wrap_kwargs

# Remove randomness
torch._inductor.config.fallback_random = True
# Remove randomeness when torch manual seed is called
patch_torch_manual_seed()


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

        device = torch.device("cuda")
        try:
            script.eval()
            script.to(device)
        except:
            pass

        args, kwargs = forward_args()
        args = wrap_args(args, device)
        kwargs = wrap_kwargs(kwargs, device)

        cosine, fp64_outputs = get_cosine_and_fp64_outputs(script, args)

        reset_rng_state()
        result1 = script(*args, **kwargs)
        reset_rng_state()
        result2 = script(*args, **kwargs)
        if os.environ.get('TEST_PY_ONLY'):
            return

        if os.environ.get('TEST_WORKING_ONLY') and not compiles:
            raise unittest.SkipTest("jit compile fails")

        reset_rng_state()
        if not os.environ.get('TEST_TORCHSCRIPT'):  # test dynamo by default
            torch._dynamo.reset()
            compiled_model = torch._dynamo.optimize("inductor")(script)
        else:
            compiled_model = torch.jit.script(script)

        if os.environ.get('TEST_COMPILE_ONLY'):
            return

        result3 = compiled_model(*args, **kwargs)

        if os.environ.get('TEST_RUN_ONLY'):
            return

        try:
            self.assertEqual(result1, result2)
        except AssertionError:
            return  # output is not deterministic
        self.assertTrue(
            same(
                result2,
                result3,
                fp64_ref=fp64_outputs,
                cos_similarity=cosine,
                tol=INDUCTOR_TOL,
            )
        )
