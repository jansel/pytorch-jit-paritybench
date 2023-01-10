import sys
_module = sys.modules[__name__]
del sys
jury = _module
cli = _module
collator = _module
core = _module
definitions = _module
metrics = _module
_core = _module
auto = _module
auxiliary = _module
base = _module
utils = _module
accuracy = _module
accuracy_for_language_generation = _module
accuracy_for_sequence_classification = _module
bartscore = _module
bartscore_for_language_generation = _module
bertscore = _module
bertscore_for_language_generation = _module
bleu = _module
bleu_for_language_generation = _module
bleurt = _module
bleurt_for_language_generation = _module
cer = _module
cer_for_language_generation = _module
chrf = _module
chrf_for_language_generation = _module
comet = _module
comet_for_language_generation = _module
f1 = _module
f1_for_language_generation = _module
f1_for_sequence_classification = _module
meteor = _module
meteor_for_language_generation = _module
precision = _module
precision_for_language_generation = _module
precision_for_sequence_classification = _module
prism = _module
prism_for_language_generation = _module
recall = _module
recall_for_language_generation = _module
recall_for_sequence_classification = _module
rouge = _module
rouge_for_language_generation = _module
sacrebleu = _module
sacrebleu_for_language_generation = _module
seqeval = _module
seqeval_for_sequence_labeling = _module
squad = _module
squad_for_language_generation = _module
ter = _module
ter_for_language_generation = _module
wer = _module
wer_for_language_generation = _module
tokenizer = _module
common = _module
io = _module
nlp = _module
setup = _module
tests = _module
conftest = _module
test_accuracy = _module
test_bartscore = _module
test_bertscore = _module
test_bleu = _module
test_bleurt = _module
test_cer = _module
test_chrf = _module
test_comet = _module
test_custom_bleu = _module
test_f1 = _module
test_meteor = _module
test_precision = _module
test_prism = _module
test_recall = _module
test_rouge = _module
test_sacrebleu = _module
test_seqeval = _module
test_squad = _module
test_ter = _module
test_wer = _module
test_cli = _module
test_import = _module
test_jury = _module
test_utils = _module
run_code_style = _module
run_tests = _module
custom_bleu = _module
custom_bleu_for_language_generation = _module

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

