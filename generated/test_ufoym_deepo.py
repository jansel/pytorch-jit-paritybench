import sys
_module = sys.modules[__name__]
del sys
generator = _module
core = _module
composer = _module
generate = _module
modules = _module
__module__ = _module
boost = _module
caffe = _module
chainer = _module
cntk = _module
darknet = _module
jupyter = _module
jupyterlab = _module
keras = _module
lasagne = _module
mxnet = _module
onnx = _module
opencv = _module
python = _module
pytorch = _module
sonnet = _module
tensorflow = _module
theano = _module
tools = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


def parametrized(dec):

    def layer(*args, **kwargs):

        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


@parametrized
def source(module, _source):
    module.source = _source
    return module


@source('apt')
class Tools(Module):

    def __repr__(self):
        return ''

    def build(self):
        return """
            DEBIAN_FRONTEND=noninteractive $APT_INSTALL \\
                build-essential \\
                apt-utils \\
                ca-certificates \\
                wget \\
                git \\
                vim \\
                libssl-dev \\
                curl \\
                unzip \\
                unrar \\
                && \\

            $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \\
            cd ~/cmake && \\
            ./bootstrap && \\
            make -j"$(nproc)" install && \\
            """


@parametrized
def dependency(module, *_deps):
    module.deps = _deps
    return module


@parametrized
def version(module, _ver):
    module.version = _ver
    return module


@dependency(Tools)
@version('3.6')
@source('apt')
class Python(Module):

    def __init__(self, manager, **args):
        super(self.__class__, self).__init__(manager, **args)
        if self.version not in ('2.7', '3.6'):
            raise NotImplementedError('unsupported python version')

    def build(self):
        return ("""
            DEBIAN_FRONTEND=noninteractive $APT_INSTALL \\
                software-properties-common \\
                && \\
            add-apt-repository ppa:deadsnakes/ppa && \\
            apt-get update && \\
            DEBIAN_FRONTEND=noninteractive $APT_INSTALL \\
                python%s \\
                python%s-dev \\
                python3-distutils%s \\
                && \\
            wget -O ~/get-pip.py \\
                https://bootstrap.pypa.io/get-pip.py && \\
            python%s ~/get-pip.py && \\
            ln -s /usr/bin/python%s /usr/local/bin/python3 && \\
            ln -s /usr/bin/python%s /usr/local/bin/python && \\
            $PIP_INSTALL \\
                setuptools \\
                && \\
            """ % (self.version, self.version, '-extra' if self.composer.ubuntu_ver.startswith('18.') else '', self.version, self.version, self.version) if self.version.startswith('3') else """
            DEBIAN_FRONTEND=noninteractive $APT_INSTALL \\
                python-pip \\
                python-dev \\
                && \\
            $PIP_INSTALL \\
                setuptools \\
                pip \\
                && \\
            """).rstrip() + """
            $PIP_INSTALL \\
                numpy \\
                scipy \\
                pandas \\
                cloudpickle \\
                scikit-image>=0.14.2 \\
                scikit-learn \\
                matplotlib \\
                Cython \\
                tqdm \\
                && \\
        """


@dependency(Python)
@source('pip')
class Pytorch(Module):

    def build(self):
        cuver = 'cpu' if self.composer.cuda_ver is None else 'cu%d' % (float(self.composer.cuda_ver) * 10)
        return """
            $PIP_INSTALL \\
                future \\
                numpy \\
                protobuf \\
                enum34 \\
                pyyaml \\
                typing \\
                && \\
            $PIP_INSTALL \\
                --pre torch torchvision -f \\
                https://download.pytorch.org/whl/nightly/%s/torch_nightly.html \\
                && \\
        """ % cuver


@dependency(Tools)
@source('git')
class Torch(Module):

    def build(self):
        return """
            DEBIAN_FRONTEND=noninteractive $APT_INSTALL \\
                sudo \\
                && \\

            $GIT_CLONE https://github.com/nagadomi/distro.git ~/torch --recursive && \\
            cd ~/torch && \\
            bash install-deps && \\
            sed -i 's/${THIS_DIR}\\/install/\\/usr\\/local/g' ./install.sh && \\
            ./install.sh && \\
        """

