"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from ..dbs import images
from .. import layers, ops, colors


def mnist(dirs=None, to_tensor=True, onehot=False, nclass=None):
    def _mnist(**kwargs):
        [xtrain, ytrain], [xvalid, yvalid] = images.mnist.load(dirs,
                                                               to_tensor,
                                                               onehot,
                                                               nclass)
        input_shape = list(xtrain.shape)
        input_shape[0] = None
        label_shape = [None] + list(ytrain.shape[1:])
        if nclass is not None:
            normed_axis = ops.helper.normalize_axes(label_shape)
            if ops.core.axis < 0:
                normed_axis += 1
            label_shape.insert(normed_axis, nclass)
        return (input_shape, label_shape), ([xtrain, ytrain], [xvalid, yvalid])
    return _mnist


def cifar(dirs, to_tensor=True, onehot=False, nclass=None, coarse=True):
    def _cifar(**kwargs):
        [xtrain, ytrain], [xvalid, yvalid] = images.cifar.load(dirs,
                                                               to_tensor,
                                                               onehot,
                                                               nclass,
                                                               coarse)
        input_shape = list(xtrain.shape)
        input_shape[0] = None
        label_shape = [None] + list(ytrain.shape[1:])

        if nclass is not None:
            normed_axis = ops.helper.normalize_axes(label_shape)
            if ops.core.axis < 0:
                normed_axis += 1
            label_shape.insert(normed_axis, nclass)
        return (input_shape, label_shape), ([xtrain, ytrain], [xvalid, yvalid])
    return _cifar
