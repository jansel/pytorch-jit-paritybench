import sys
_module = sys.modules[__name__]
del sys
main = _module
model = _module
model_keras = _module
model_pytorch = _module
model_tensorflow = _module

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


from torch.nn import Module


from torch.nn import LSTM


from torch.nn import Linear


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import numpy as np


class Net(Module):
    """
    pytorch预测模型，包括LSTM时序预测层和Linear回归输出层
    可以根据自己的情况增加模型结构
    """

    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.
            hidden_size, num_layers=config.lstm_layers, batch_first=True,
            dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=
            config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hichenway_stock_predict_with_LSTM(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Net(*[], **{'config': _mock_config(input_size=4, hidden_size=4, lstm_layers=1, dropout_rate=0.5, output_size=4)}), [torch.rand([4, 4, 4])], {})

