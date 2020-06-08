import sys
_module = sys.modules[__name__]
del sys
bert_model = _module
flask_example = _module
flask_multigpu_example = _module
future_example = _module
redis_streamer_gunicorn = _module
redis_worker_example = _module
app = _module
model = _module
service_streamer = _module
managed_model = _module
setup = _module
test_service_streamer = _module
vision_case = _module

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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ShannonAI_service_streamer(_paritybench_base):
    pass
