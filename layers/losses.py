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

from .. import ops
from . import core

""" termology
    ----------
        logits : unnormalized output from network
                 logits generally is the input to softmax
"""

@core.layer
def binary_cross_entropy(inputs,
                         axis=None,
                         from_logits=False,
                         onehot=True,
                         epsilon=ops.core.epsilon,
                         reuse=False,
                         name=None,
                         scope=None):
    """ binary cross entropy
        Attributes
        ==========
            inputs : list / dict
                     must be [network_output, labels]
            axis : int / None
                   axis indicates number of class
            from_logits : bool
                          whether network_output is logits or probabilities
            onehot : bool
                     whether labels is in onehot format or not
    """
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.binary_cross_entropy(axis,
                                           from_logits,
                                           onehot,
                                           reuse,
                                           name,
                                           scope,
                                           epsilon)(inputs, labels)


@core.layer
def categorical_cross_entropy(inputs,
                              axis=None,
                              from_logits=False,
                              onehot=True,
                              epsilon=ops.core.epsilon,
                              reuse=False,
                              name=None,
                              scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.categorical_cross_entropy(axis,
                                                from_logits,
                                                onehot,
                                                reuse,
                                                name,
                                                scope,
                                                epsilon)(inputs, labels)


@core.layer
def mean_square_error(inputs,
                      axis=None,
                      from_logits=True,
                      onehot=True,
                      reuse=False,
                      name=None,
                      scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.mean_square_error(axis,
                                        from_logits,
                                        onehot,
                                        reuse,
                                        name,
                                        scope)(inputs, labels)


@core.layer
def mean_absolute_error(inputs,
                        axis=None,
                        from_logits=False,
                        onehot=True,
                        reuse=False,
                        name=None,
                        scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.mean_absolute_error(axis,
                                          from_logits,
                                          onehot,
                                          reuse,
                                          name,
                                          scope)(inputs, labels)


@core.layer
def winner_takes_all(inputs,
                     axis=None,
                     from_logits=False,
                     onehot=True,
                     reuse=False,
                     name=None,
                     scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.winner_takes_all(axis,
                                       from_logits,
                                       onehot,
                                       reuse,
                                       name,
                                       scope)(inputs, labels)


@core.layer
def margin_loss(inputs,
                axis=None,
                positive_margin=0.9,
                negative_margin=0.1,
                downweight=0.5,
                from_logits=True,
                onehot=True,
                reuse=False,
                name=None,
                scope=None):
    """ margin loss for capsule networks
        NOTE margin_loss cannot be used for normal CNN
        because margin loss treat inputs as
        `vector in vector out` tensor
    """
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.margin_loss(axis,
                                  from_logits,
                                  onehot,
                                  reuse,
                                  name,
                                  scope,
                                  positive_margin,
                                  negative_margin,
                                  downweight)(inputs, labels)


@core.layer
def chamfer_loss(inputs,
                 axis=None,
                 dtype=ops.core.float64,
                 metric=None,
                 alpha=0.5,
                 belta=0.5,
                 from_logits=True,
                 onehot=True,
                 reuse=False,
                 name=None,
                 scope=None):
    """ chamfer distance as loss function
        chamfer distance of two sets of points is the
        MEAN MIN distance of each points in each set
        that is:
            mean(min(each point in setA, setB)) + mean(min(setA, each point in setB))
    """
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.chamfer_loss(axis,
                                   from_logits,
                                   onehot,
                                   reuse,
                                   name,
                                   scope,
                                   dtype,
                                   metric,
                                   alpha,
                                   belta)(inputs, labels)


# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
mae = mean_absolute_error
wta = winner_takes_all


def get(l, inputs, labels, **kwargs):
    if isinstance(l, str):
        return eval('{}([inputs, labels], **kwargs)'.format(l))
    elif core.helper.is_tensor(l) or callable(l):
        return l
    else:
        raise ValueError('cannot get activates `{}` with type {}'
                         .format(l, type(l)))
