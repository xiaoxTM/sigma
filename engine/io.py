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

def imageio(fun):
    """ fun : callable
              returns generator, iterations
    """
    def reader(**kwargs):
        (input_shape, label_shape), (train, valid) = fun()
        print('input shape: {}\nlabel shape: {}'
              .format(colors.red(input_shape),
                      colors.red(label_shape)))
        valid_gen, valid_iters = None, None
        if isinstance(train, str):
            generator, _, iterations = images.generator(train, **kwargs)
            if valid is not None:
                kwargs['shuffle'] = False
                valid_gen, _, valid_iters = images.generator(valid, **kwargs)
        elif isinstance(train, (list, tuple)):
            generator, _, iterations = images.make_generator(train[0],
                                                             train[1],
                                                             **kwargs)
            if valid is not None:
                kwargs['shuffle'] = False
                valid_gen, _, valid_iters = images.make_generator(valid[0],
                                                                  valid[1],
                                                                  **kwargs)
        inputs = layers.base.input_spec(input_shape,
                                        dtype=ops.core.float32,
                                        name='inputs')
        labels = None
        if label_shape is not None:
            labels =layers.base.label_spec(label_shape,
                                           dtype=ops.core.float32,
                                           name='labels')
        return (inputs, labels), \
               (generator(inputs, labels), iterations), \
               (valid_gen(inputs, labels), valid_iters)
    return reader


def mnist(dirs=None, to_tensor=True, onehot=False, nclass=None):
    @imageio
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
    @imageio
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
