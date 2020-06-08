import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
shapenet = _module
jit = _module
abstract_network = _module
feature_extractors = _module
homogeneous_shape_layer = _module
homogeneous_transform_layer = _module
shape_layer = _module
shape_network = _module
layer = _module
homogeneous_shape_layer = _module
homogeneous_transform_layer = _module
shape_layer = _module
networks = _module
abstract_network = _module
feature_extractors = _module
single_shape = _module
shape_network = _module
utils = _module
scripts = _module
export_to_jit = _module
predict_from_net = _module
prepare_datasets = _module
train_single_shapenet = _module
load_config_file = _module
misc = _module
tests = _module
test_jit_equality = _module
test_homogeneous_shape_layer = _module
test_homogeneous_transform_layer = _module
test_feature_extractor = _module
test_single_shape_network = _module
test_config = _module

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


from abc import abstractmethod


import logging


from torch.utils.cpp_extension import load as load_cpp


import numpy as np


class AbstractShapeNetwork(torch.jit.ScriptModule):
    """
    Abstract JIT Network

    """

    def __init__(self, **kwargs):
        super().__init__(optimize=True)

    @staticmethod
    def norm_type_to_class(norm_type):
        norm_dict = {'instance': torch.nn.InstanceNorm2d, 'batch': torch.nn
            .BatchNorm2d}
        norm_class = norm_dict.get(norm_type, None)
        return norm_class


class AbstractFeatureExtractor(torch.jit.ScriptModule):
    """
    Abstract Feature Extractor Class all further feature extractors
    should be derived from

    """

    def __init__(self, in_channels, out_params, norm_class, p_dropout=0):
        """

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_params : int
            number of outputs
        norm_class : Any
            Class implementing a normalization
        p_dropout : float
            dropout probability

        """
        super().__init__()
        self.model = self._build_model(in_channels, out_params, norm_class,
            p_dropout)

    @torch.jit.script_method
    def forward(self, input_batch):
        """
        Feed batch through network

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            batch to feed through network

        Returns
        -------
        :class:`torch.Tensor`
            extracted features

        """
        return self.model(input_batch)

    @staticmethod
    @abstractmethod
    def _build_model(in_channels, out_features, norm_class, p_dropout):
        """
        Build the actual model structure

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_features : int
            number of outputs
        norm_class : Any
            class implementing a normalization
        p_dropout : float
            dropout probability

        Returns
        -------
        :class:`torch.jit.ScriptModule`
            ensembled model

        """
        raise NotImplementedError


class Conv2dRelu(torch.jit.ScriptModule):
    """
    Block holding one Conv2d and one ReLU layer
    
    """

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        **args :
            positional arguments (passed to Conv2d)
        **kwargs : dict
            keyword arguments (passed to Conv2d)

        """
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    @torch.jit.script_method
    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch : class:`torch.Tensor`
            input batch

        Returns
        -------
        class:`torch.Tensor`
            result

        """
        return self._relu(self._conv(input_batch))


class _HomogeneousTransformationLayerPy(torch.jit.ScriptModule):
    """
    Module to perform homogeneous transformations in 2D and 3D
    (Implemented in Python)

    """
    __constants__ = ['_n_dims']

    def __init__(self, n_dims):
        """

        Parameters
        ----------
        n_dims: int
            number of dimensions
        """
        super().__init__()
        homogen_trafo = torch.zeros(1, n_dims + 1, n_dims + 1)
        homogen_trafo[:, (-1), :-1] = 0.0
        homogen_trafo[:, (-1), (-1)] = 1.0
        self.register_buffer('_trafo_matrix', homogen_trafo)
        self._n_dims = n_dims

    @torch.jit.script_method
    def forward(self, shapes: torch.Tensor, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix and applies it to the
        shape tensor

        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes to transform
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            the transformed shapes in cartesian coordinates

        """
        assert shapes.size(-1
            ) == self._n_dims, 'Layer for other dimensionality specified'
        trafo_matrix = self._ensemble_trafo(rotation_params,
            translation_params, scale_params)
        homogen_shapes = torch.cat([shapes, torch.ones([shapes.size(0),
            shapes.size(1), 1], dtype=shapes.dtype, device=shapes.device)],
            dim=-1)
        transformed_shapes = torch.bmm(homogen_shapes, trafo_matrix.permute
            (0, 2, 1))
        return transformed_shapes[:, :, :-1]

    @torch.jit.script_method
    def _ensemble_trafo(self, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the transformation matrix in 2D and 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            transformation matrix

        """
        rotation_params = rotation_params.view(rotation_params.size()[:2])
        translation_params = translation_params.view(translation_params.
            size()[:2])
        scale_params = scale_params.view(scale_params.size()[:2])
        if self._n_dims == 2:
            trafo = self._ensemble_2d_matrix(rotation_params,
                translation_params, scale_params)
        else:
            trafo = self._ensemble_3d_matrix(rotation_params,
                translation_params, scale_params)
        return trafo

    @torch.jit.script_method
    def _ensemble_2d_matrix(self, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 2D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one parameter)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (two parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            2D transformation matrix

        """
        homogen_trafo = getattr(self, '_trafo_matrix').repeat(scale_params.
            size(0), 1, 1).clone()
        homogen_trafo[:, (0), (0)] = (scale_params * rotation_params.cos())[:,
            (0)].clone()
        homogen_trafo[:, (0), (1)] = (scale_params * rotation_params.sin())[:,
            (0)].clone()
        homogen_trafo[:, (1), (0)] = (-scale_params * rotation_params.sin())[:,
            (0)].clone()
        homogen_trafo[:, (1), (1)] = (scale_params * rotation_params.cos())[:,
            (0)].clone()
        homogen_trafo[:, :-1, (-1)] = translation_params.clone()
        return homogen_trafo

    @torch.jit.script_method
    def _ensemble_3d_matrix(self, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (three parameters)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (three parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            3D transformation matrix

        """
        homogen_trafo = getattr(self, '_trafo_matrix').repeat(scale_params.
            size(0), 1, 1).clone()
        roll = rotation_params[:, (2)].unsqueeze(-1)
        pitch = rotation_params[:, (1)].unsqueeze(-1)
        yaw = rotation_params[:, (0)].unsqueeze(-1)
        homogen_trafo[:, (0), (0)] = (scale_params * (pitch.cos() * roll.cos())
            )[:, (0)].clone()
        homogen_trafo[:, (0), (1)] = (scale_params * (pitch.cos() * roll.sin())
            )[:, (0)].clone()
        homogen_trafo[:, (0), (2)] = (scale_params * -pitch.sin())[:, (0)
            ].clone()
        homogen_trafo[:, (1), (0)] = (scale_params * (yaw.sin() * pitch.sin
            () * roll.cos() - yaw.cos() * roll.sin()))[:, (0)].clone()
        homogen_trafo[:, (1), (1)] = (scale_params * (yaw.sin() * pitch.sin
            () * roll.sin() + yaw.cos() * roll.cos()))[:, (0)].clone()
        homogen_trafo[:, (1), (2)] = (scale_params * (yaw.sin() * pitch.cos())
            )[:, (0)].clone()
        homogen_trafo[:, (2), (0)] = (scale_params * (yaw.cos() * pitch.sin
            () * roll.cos() + yaw.sin() * roll.sin()))[:, (0)].clone()
        homogen_trafo[:, (2), (1)] = (scale_params * (yaw.cos() * pitch.sin
            () * roll.sin() - yaw.sin() * roll.cos()))[:, (0)].clone()
        homogen_trafo[:, (2), (2)] = (scale_params * (yaw.cos() * pitch.cos())
            )[:, (0)].clone()
        homogen_trafo[:, :-1, (-1)] = translation_params.clone()
        return homogen_trafo


class HomogeneousTransformationLayer(torch.jit.ScriptModule):
    """
    Wrapper Class to Wrap the Python and C++ API into a combined python API

    """

    def __init__(self, n_dims: int, use_cpp=False):
        """

        Parameters
        ----------
        n_dims : int
            number of dimensions
        use_cpp : bool
            whether or not to use C++ implementation

        Raises
        ------
        AssertionError
            if ``use_cpp`` is True, currently only the python version is
            supported

        """
        assert use_cpp == False, 'Currently only the python version is supported'
        super().__init__()
        self._n_params = {}
        if n_dims == 2:
            self._n_params['scale'] = 1
            self._n_params['rotation'] = 1
            self._n_params['translation'] = 2
        elif n_dims == 3:
            self._n_params['scale'] = 3
            self._n_params['rotation'] = 3
            self._n_params['translation'] = 3
        self._layer = _HomogeneousTransformationLayerPy(n_dims)
        total_params = 0
        for key, val in self._n_params.items():
            self.register_buffer('_indices_%s_params' % key, torch.arange(
                total_params, total_params + val))
            total_params += val

    @torch.jit.script_method
    def forward(self, shapes: torch.Tensor, params: torch.Tensor):
        """
        Selects individual parameters from ``params`` and forwards them through 
        the actual layer implementation
        
        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes to transform
        params : :class:`torch.Tensor`
            parameters specifying the affine transformation
        
        Returns
        -------
        Returns
        -------
        :class:`torch.Tensor`
            the transformed shapes in cartesian coordinates

        """
        rotation_params = params.index_select(dim=1, index=getattr(self,
            '_indices_rotation_params'))
        scale_params = params.index_select(dim=1, index=getattr(self,
            '_indices_scale_params'))
        translation_params = params.index_select(dim=1, index=getattr(self,
            '_indices_translation_params'))
        return self._layer(shapes, rotation_params, translation_params,
            scale_params)

    @property
    def num_params(self):
        num_params = 0
        for key, val in self._n_params.items():
            num_params += val
        return num_params


class _ShapeLayerPy(torch.jit.ScriptModule):
    """
    Python Implementation of Shape Layer

    """

    def __init__(self, shapes):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()
        self.register_buffer('_shape_mean', torch.from_numpy(shapes[0]).
            float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(_shape).float().unsqueeze(0))
        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer('_shape_components', component_tensor)

    @torch.jit.script_method
    def forward(self, shape_params: torch.Tensor):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            shape parameters

        Returns
        -------
        :class:`torch.Tensor`
            ensembled shape

        """
        shapes = getattr(self, '_shape_mean').clone()
        shapes = shapes.expand(shape_params.size(0), shapes.size(1), shapes
            .size(2))
        components = getattr(self, '_shape_components')
        components = components.expand(shape_params.size(0), components.
            size(1), components.size(2), components.size(3))
        weighted_components = components.mul(shape_params.expand_as(components)
            )
        shapes = shapes.add(weighted_components.sum(dim=1))
        return shapes

    @property
    def num_params(self):
        return getattr(self, '_shape_components').size(1)


class ShapeLayer(torch.jit.ScriptModule):

    def __init__(self, shapes, use_cpp=False):
        """

        Parameters
        ----------
        shapes : np.ndarray
            the shape components needed by the actual shape layer implementation
        use_cpp : bool
            whether to use cpp implementation or not
            (Currently only the python version is supported)

        """
        super().__init__()
        self._layer = _ShapeLayerPy(shapes)
        assert not use_cpp, 'Currently only the Python Version is supported'

    @torch.jit.script_method
    def forward(self, shape_params: torch.Tensor):
        return self._layer(shape_params)

    @property
    def num_params(self):
        return self._layer.num_params


class HomogeneousShapeLayer(torch.nn.Module):
    """
    Module to Perform a Shape Prediction
    (including a global homogeneous transformation)

    """

    def __init__(self, shapes, n_dims, use_cpp=False):
        """

        Parameters
        ----------
        shapes : np.ndarray
            shapes to construct a :class:`ShapeLayer`
        n_dims : int
            number of shape dimensions
        use_cpp : bool
            whether or not to use (experimental) C++ Implementation

        See Also
        --------
        :class:`ShapeLayer`
        :class:`HomogeneousTransformationLayer`

        """
        super().__init__()
        self._shape_layer = ShapeLayer(shapes, use_cpp)
        self._homogen_trafo = HomogeneousTransformationLayer(n_dims, use_cpp)
        self.register_buffer('_indices_shape_params', torch.arange(self.
            _shape_layer.num_params))
        self.register_buffer('_indices_homogen_params', torch.arange(self.
            _shape_layer.num_params, self.num_params))

    def forward(self, params: torch.Tensor):
        """
        Performs the actual prediction

        Parameters
        ----------
        params : :class:`torch.Tensor`
            input parameters

        Returns
        -------
        :class:`torch.Tensor`
            predicted shape

        """
        shape_params = params.index_select(dim=1, index=getattr(self,
            '_indices_shape_params'))
        transformation_params = params.index_select(dim=1, index=getattr(
            self, '_indices_homogen_params'))
        shapes = self._shape_layer(shape_params)
        transformed_shapes = self._homogen_trafo(shapes, transformation_params)
        return transformed_shapes

    @property
    def num_params(self):
        """
        Property to access these layer's number of parameters

        Returns
        -------
        int
            number of parameters

        """
        return self._shape_layer.num_params + self._homogen_trafo.num_params


class HomogeneousTransformationLayer(torch.nn.Module):
    """
    Wrapper Class to Wrap the Python and C++ API into a combined python API

    """

    def __init__(self, n_dims: int, use_cpp=False):
        """

        Parameters
        ----------
        n_dims : int
            number of dimensions
        use_cpp : bool
            whether or not to use C++ implementation

        """
        super().__init__()
        self._n_params = {}
        if n_dims == 2:
            self._n_params['scale'] = 1
            self._n_params['rotation'] = 1
            self._n_params['translation'] = 2
        elif n_dims == 3:
            self._n_params['scale'] = 3
            self._n_params['rotation'] = 3
            self._n_params['translation'] = 3
        if use_cpp:
            self._layer = _HomogeneousTransformationLayerCpp(n_dims)
        else:
            self._layer = _HomogeneousTransformationLayerPy(n_dims)
        total_params = 0
        for key, val in self._n_params.items():
            self.register_buffer('_indices_%s_params' % key, torch.arange(
                total_params, total_params + val))
            total_params += val

    def forward(self, shapes: torch.Tensor, params: torch.Tensor):
        """
        Actual prediction

        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes before applied global transformation
        params : :class:`torch.Tensor`
            parameters specifying the global transformation

        Returns
        -------
        :class:`torch.Tensor`
            Transformed shapes

        """
        rotation_params = params.index_select(dim=1, index=getattr(self,
            '_indices_rotation_params'))
        scale_params = params.index_select(dim=1, index=getattr(self,
            '_indices_scale_params'))
        translation_params = params.index_select(dim=1, index=getattr(self,
            '_indices_translation_params'))
        return self._layer(shapes, rotation_params, translation_params,
            scale_params)

    @property
    def num_params(self):
        num_params = 0
        for key, val in self._n_params.items():
            num_params += val
        return num_params


class _HomogeneousTransformationLayerCpp(torch.nn.Module):
    """
    Module to perform homogeneous transformations in 2D and 3D
    (Implemented in C++)

    """

    def __init__(self, n_dims, verbose=True):
        """

        Parameters
        ----------
        n_dims : int
            number of dimensions
        verbose : float
            if True: verbosity during C++ loading

        """
        super().__init__()
        homogen_trafo = torch.zeros(1, n_dims + 1, n_dims + 1)
        homogen_trafo[:, (-1), :-1] = 0.0
        homogen_trafo[:, (-1), (-1)] = 1.0
        self.register_buffer('_trafo_matrix', homogen_trafo)
        self._n_dims = n_dims
        self._func = load_cpp('homogeneous_transform_function', sources=[os
            .path.join(os.path.split(__file__)[0],
            'homogeneous_transform_layer.cpp')], verbose=verbose)

    def forward(self, shapes: torch.Tensor, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix and applies it to the
        shape tensor

        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes to transform
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            the transformed shapes in cartesian coordinates

        """
        transformed_shapes = self._func.forward(shapes, getattr(self,
            '_trafo_matrix'), rotation_params, translation_params, scale_params
            )
        return transformed_shapes


class _HomogeneousTransformationLayerPy(torch.nn.Module):
    """
    Module to perform homogeneous transformations in 2D and 3D
    (Implemented in Python)

    """

    def __init__(self, n_dims):
        """

        Parameters
        ----------
        n_dims : int
            number of dimensions

        """
        super().__init__()
        homogen_trafo = torch.zeros(1, n_dims + 1, n_dims + 1)
        homogen_trafo[:, (-1), :-1] = 0.0
        homogen_trafo[:, (-1), (-1)] = 1.0
        self.register_buffer('_trafo_matrix', homogen_trafo)
        self._n_dims = n_dims

    def forward(self, shapes: torch.Tensor, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix and applies it to the
        shape tensor

        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes to transform
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            the transformed shapes in cartesian coordinates

        """
        assert shapes.size(-1
            ) == self._n_dims, 'Layer for other dimensionality specified'
        trafo_matrix = self._ensemble_trafo(rotation_params,
            translation_params, scale_params)
        homogen_shapes = torch.cat([shapes, shapes.new_ones(*shapes.size()[
            :-1], 1)], dim=-1)
        transformed_shapes = torch.bmm(homogen_shapes, trafo_matrix.permute
            (0, 2, 1))
        transformed_shapes = transformed_shapes[(...), :-1]
        return transformed_shapes

    def _ensemble_trafo(self, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the transformation matrix in 2D and 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            transformation matrix

        """
        rotation_params = rotation_params.view(rotation_params.size()[:2])
        translation_params = translation_params.view(translation_params.
            size()[:2])
        scale_params = scale_params.view(scale_params.size()[:2])
        if self._n_dims == 2:
            return self._ensemble_2d_matrix(rotation_params,
                translation_params, scale_params)
        elif self._n_dims == 3:
            return self._ensemble_3d_matrix(rotation_params,
                translation_params, scale_params)
        else:
            raise NotImplementedError(
                'Implementation for n_dims = %d not available' % self._n_dims)

    def _ensemble_2d_matrix(self, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 2D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one parameter)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (two parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            2D transformation matrix

        """
        homogen_trafo = getattr(self, '_trafo_matrix').repeat(scale_params.
            size(0), 1, 1).clone()
        homogen_trafo[:, (0), (0)] = (scale_params * rotation_params.cos())[:,
            (0)].clone()
        homogen_trafo[:, (0), (1)] = (scale_params * rotation_params.sin())[:,
            (0)].clone()
        homogen_trafo[:, (1), (0)] = (-scale_params * rotation_params.sin())[:,
            (0)].clone()
        homogen_trafo[:, (1), (1)] = (scale_params * rotation_params.cos())[:,
            (0)].clone()
        homogen_trafo[:, :-1, (-1)] = translation_params.clone()
        return homogen_trafo

    def _ensemble_3d_matrix(self, rotation_params: torch.Tensor,
        translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (three parameters)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (three parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            3D transformation matrix

        """
        homogen_trafo = getattr(self, '_trafo_matrix').repeat(scale_params.
            size(0), 1, 1).clone()
        roll = rotation_params[:, (2)].unsqueeze(-1)
        pitch = rotation_params[:, (1)].unsqueeze(-1)
        yaw = rotation_params[:, (0)].unsqueeze(-1)
        homogen_trafo[:, (0), (0)] = (scale_params * (pitch.cos() * roll.cos())
            )[:, (0)].clone()
        homogen_trafo[:, (0), (1)] = (scale_params * (pitch.cos() * roll.sin())
            )[:, (0)].clone()
        homogen_trafo[:, (0), (2)] = (scale_params * -pitch.sin())[:, (0)
            ].clone()
        homogen_trafo[:, (1), (0)] = (scale_params * (yaw.sin() * pitch.sin
            () * roll.cos() - yaw.cos() * roll.sin()))[:, (0)].clone()
        homogen_trafo[:, (1), (1)] = (scale_params * (yaw.sin() * pitch.sin
            () * roll.sin() + yaw.cos() * roll.cos()))[:, (0)].clone()
        homogen_trafo[:, (1), (2)] = (scale_params * (yaw.sin() * pitch.cos())
            )[:, (0)].clone()
        homogen_trafo[:, (2), (0)] = (scale_params * (yaw.cos() * pitch.sin
            () * roll.cos() + yaw.sin() * roll.sin()))[:, (0)].clone()
        homogen_trafo[:, (2), (1)] = (scale_params * (yaw.cos() * pitch.sin
            () * roll.sin() - yaw.sin() * roll.cos()))[:, (0)].clone()
        homogen_trafo[:, (2), (2)] = (scale_params * (yaw.cos() * pitch.cos())
            )[:, (0)].clone()
        homogen_trafo[:, :-1, (-1)] = translation_params.clone()
        return homogen_trafo


class ShapeLayer(torch.nn.Module):
    """
    Wrapper to compine Python and C++ Implementation under Single API

    """

    def __init__(self, shapes, use_cpp=False):
        """

        Parameters
        ----------
        shapes : np.ndarray
            the actual shape components
        use_cpp : bool
            whether or not to use the (experimental) C++ Implementation
        """
        super().__init__()
        if use_cpp:
            self._layer = _ShapeLayerCpp(shapes)
        else:
            self._layer = _ShapeLayerPy(shapes)

    def forward(self, shape_params: torch.Tensor):
        """
        Forwards parameters to Python or C++ Implementation

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            parameters for shape ensembling

        Returns
        -------
        :class:`torch.Tensor`
            Ensempled Shape

        """
        return self._layer(shape_params)

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return self._layer.num_params


class _ShapeLayerPy(torch.nn.Module):
    """
    Python Implementation of Shape Layer

    """

    def __init__(self, shapes):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()
        self.register_buffer('_shape_mean', torch.from_numpy(shapes[0]).
            float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(_shape).float().unsqueeze(0))
        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer('_shape_components', component_tensor)

    def forward(self, shape_params: torch.Tensor):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            shape parameters

        Returns
        -------
        :class:`torch.Tensor`
            ensembled shape

        """
        shapes = getattr(self, '_shape_mean').clone()
        shapes = shapes.expand(shape_params.size(0), *shapes.size()[1:])
        components = getattr(self, '_shape_components')
        components = components.expand(shape_params.size(0), *components.
            size()[1:])
        weighted_components = components.mul(shape_params.expand_as(components)
            )
        shapes = shapes.add(weighted_components.sum(dim=1))
        return shapes

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return getattr(self, '_shape_components').size(1)


class _ShapeLayerCpp(torch.nn.Module):
    """
    C++ Implementation of Shape Layer

    """

    def __init__(self, shapes, verbose=True):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()
        self.register_buffer('_shape_mean', torch.from_numpy(shapes[0]).
            float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(_shape).float().unsqueeze(0))
        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer('_shape_components', component_tensor)
        self._func = load_cpp('shape_function', sources=[os.path.join(os.
            path.split(__file__)[0], 'shape_layer.cpp')], verbose=verbose)

    def forward(self, shape_params: torch.Tensor):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : :class:`torch.Tensor`
            shape parameters

        Returns
        -------
        :class:`torch.Tensor`
            ensembled shape
        """
        shapes = self._func.forward(shape_params, getattr(self,
            '_shape_mean'), getattr(self, '_shape_components'))
        return shapes

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return getattr(self, '_shape_components').size(1)


class AbstractFeatureExtractor(torch.nn.Module):
    """
    Abstract Feature Extractor Class all further feature extracotrs
    should be derived from

    """

    def __init__(self, in_channels, out_params, norm_class, p_dropout=0):
        """

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_params : int
            number of outputs
        norm_class : Any
            Class implementing a normalization
        p_dropout : float
            dropout probability

        """
        super().__init__()
        self.model = self._build_model(in_channels, out_params, norm_class,
            p_dropout)

    def forward(self, input_batch):
        """
        Feed batch through network

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            batch to feed through network

        Returns
        -------
        :class:`torch.Tensor`
            exracted features

        """
        return self.model(input_batch)

    @staticmethod
    @abstractmethod
    def _build_model(in_channels, out_features, norm_class, p_dropout):
        """
        Build the actual model structure

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_features : int
            number of outputs
        norm_class : Any
            class implementing a normalization
        p_dropout : float
            dropout probability

        Returns
        -------
        :class:`torch.nn.Module`
            ensembled model
        """
        raise NotImplementedError


class Conv2dRelu(torch.nn.Module):
    """
    Block holding one Conv2d and one ReLU layer
    """

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        *args :
            positional arguments (passed to Conv2d)
        **kwargs :
            keyword arguments (passed to Conv2d)

        """
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
            result
        """
        return self._relu(self._conv(input_batch))


class CustomGroupNorm(torch.nn.Module):
    """
    Custom Group Norm which adds n_groups=2 as default parameter
    """

    def __init__(self, n_features, n_groups=2):
        """

        Parameters
        ----------
        n_features : int
            number of input features
        n_groups : int
            number of normalization groups
        """
        super().__init__()
        self.norm = torch.nn.GroupNorm(n_groups, n_features)

    def forward(self, x):
        """
        Forward batch through network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            batch to forward

        Returns
        -------
        :class:`torch.Tensor`
            normalized results

        """
        return self.norm(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_justusschock_shapenet(_paritybench_base):
    pass

    def test_000(self):
        self._check(Conv2dRelu(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CustomGroupNorm(*[], **{'n_features': 4}), [torch.rand([4, 4, 4, 4])], {})
