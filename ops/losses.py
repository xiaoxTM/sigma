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

from . import helper, core

def loss(fun):
    def _loss(axis,
              from_logits=True,
              onehot=True,
              reuse=False,
              name=None,
              scope=None,
              *args):
        ops_scope, _, _ = helper.assign_scope(name,
                                                 scope,
                                                 fun.__name__,
                                                 reuse)
        return fun(axis, from_logits, onehot, reuse, name, ops_scope, *args)
    return _loss


@loss
def binary_cross_entropy(axis,
                         from_logits=True,
                         onehot=True,
                         reuse=False,
                         name=None,
                         scope=None):
    def _binary_cross_entropy(x, labels):
        with scope:
            if not onehot:
                depth = core.shape(x)[axis]
                labels = core.one_hot(labels, depth)
            if not from_logits:
                # source code borrowed from:
                #     @keras.backend.tensorflow_backend.py
                x = core.clip(x, statis.epsilon, 1- statis.epsilon)
                x = core.log(x / (1-x))
            return core.mean(
                core.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=x),
                axis=axis)
    return _binary_cross_entropy


@loss
def categorical_cross_entropy(axis,
                              from_logits=True,
                              onehot=True,
                              reuse=False,
                              name=None,
                              scope=None):
    def _categorical_cross_entropy(x, labels):
        with scope:
            if not onehot:
                depth = core.shape(x)[axis]
                labels = core.one_hot(labels, depth)
            if from_logits:
                return core.mean(
                    core.softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=x),
                    axis=axis)
            else:
                # source code borrowed from:
                #     @keras.backend.tensorflow_backend.py
                x /= core.sum(x,
                              len(x.get_shape())-1,
                              True)
                x = core.clip(x, statis.epsilon, 1-statis.epsilon)
            return -core.sum(label * core.log(x),
                             len(output.get_shape())-1)
    return _categorical_cross_entropy


@loss
def mean_square_error(axis,
                      from_logits=True,
                      onehot=True,
                      reuse=False,
                      name=None,
                      scope=None):
    def _mean_square_error(x, labels):
        with scope:
            if not onehot:
                depth = core.shape(x)[axis]
                labels = core.one_hot(labels, depth)
            return core.mean(core.square(x - labels), axis=axis)
    return _mean_square_error


@loss
def mean_absolute_error(axis,
                        from_logits=True,
                        onehot=True,
                        reuse=False,
                        name=None,
                        scope=None):
    def _mean_sabsolute_error(x, labels):
        with scope:
            if not onehot:
                depth = core.shape(x)[axis]
                labels = core.one_hot(labels, depth)
            return core.mean(core.abs(x - labels), axis=axis)
    return _mean_absolute_error


@loss
def winner_takes_all(axis,
                     from_logits=True,
                     onehot=True,
                     reuse=False,
                     name=None,
                     scope=None):
    def _winner_takes_all(x, labels):
        with scope:
            shape = core.shape(x)
            pred = core.argmax(x, axis=axis)
            if not onehot:
                labels = core.one_hot(labels, shape[axis])
            loss_tensor = core.where(pred==labels,
                                   core.zeros_like(labels),
                                   core.ones_like(labels))
            loss_matrix = core.reshape(loss_tensor, (shape[0], -1))
            return core.mean(loss_matrix, axis=axis)
    return _winner_takes_all


@loss
def margin_loss(axis,
                from_logits=True,
                onehot=True,
                reuse=False,
                name=None,
                scope=None,
                positive_margin=0.9,
                negative_margin=0.1,
                downweight=0.5):
    if axis is None:
        axis = core.axis
    def _margin_loss(x, labels):
        with scope:
            if not onehot:
                depth = core.shape(x)[axis]
                labels = core.one_hot(labels, depth)
            labels = core.cast(labels, core.float32)
            # L_k = T_k * max(0, m+ - |x|)^2 + (1-T_k) * max(0.0, |x| - m-)^2
            ploss = core.sum(labels * core.max(0.0, positive_margin - x)**2,
                             axis=axis)
            nloss = core.sum((1-labels) * core.max(0.0, x - negative_margin)**2,
                             axis=axis)
            return core.mean(ploss + downweight * nloss)
    return _margin_loss


def get(l, **kwargs):
    """ get loss from None | string | callable function
    """
    if l is None:
        raise TypeError('no loss specified to get loss function')
    elif isinstance(l, str):
        return eval('{}(**kwargs)'.format(l))
    elif helper.is_tensor(l) or callable(l):
        return l
    else:
        raise ValueError('cannot get loss `{}` with type {}'
                         .format(l, type(l)))
