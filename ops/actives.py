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

from .. import colors
from . import helper, core

def squash(axis=-2,
           epsilon=core.epsilon,
           safe=True,
           aslayer=False,
           reuse=False,
           name=None,
           scope=None):
    """ squash function to range input into [0, 1) as activation
        used for capsule networks. see :
        @https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf

        input (x) should with shape of
        [batch-size, dim, channels] for fully_connected capsule
        or
        [batch-size, neurons, dims, channels] for conv1d
        or
        [batch-size, rows, cols, dims, channels] for conv2d
        capsule layer
        where `ncaps` denotes the number of capsules
        and `cap_dim` denotes the dimension of each capsule
    """
    def _squash(x):
        with helper.maybe_layer(aslayer, name, scope, 'squash', reuse):
            norm, squared_norm = core.norm(x,
                                           axis,
                                           safe=safe,
                                           epsilon=epsilon,
                                           return_squared=True,
                                           name=name,
                                           keepdims=True)
            return (x / norm) * (squared_norm / (1 + squared_norm))
    return _squash


def crelu(aslayer=False,
          reuse=False,
          name=None,
          scope=None):
    def _crelu(x):
        with helper.maybe_layer(aslayer, name, scope, 'crelu', reuse):
            return core.crelu(x)
    return _crelu


""" relu activates
    calculates:
        max(x, 0)
"""
def relu(aslayer=False,
         reuse=False,
         name=None,
         scope=None):
    def _relu(x):
        with helper.maybe_layer(aslayer, name, scope, 'relu', reuse):
            return core.relu(x)
    return _relu


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
def relu6(aslayer=False,
          reuse=False,
          name=None,
          scope=None):
    def _relu6(x):
        with helper.maybe_layer(aslayer, name, scope, 'relu6', reuse):
            return core.relu6(x)
    return _relu6


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
def elu(aslayer=False,
        reuse=False,
        name=None,
        scope=None):
    def _elu(x):
        with helper.maybe_layer(aslayer, name, scope, 'elu', reuse):
            return core.elu(x)
    return _elu


""" elu activates
    calculates:
        scale * alpha * (exp(x) -1)
"""
def selu(alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946,
         aslayer=False,
         reuse=False,
         name=None,
         scope=None):
    def _selu(x):
        with helper.maybe_layer(aslayer, name, scope, 'selu', reuse):
            return scale * alpha * (core.exp(x) - 1)
    return _selu


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
def leaky_relu(alpha=0.2,
               aslayer=False,
               reuse=False,
               name=None,
               scope=None):
    def _leaky_relu(x):
        with helper.maybe_layer(aslayer, name, scope, 'leaky_relu', reuse):
            return core.leaky_relu(x, alpha)
    return _leaky_relu


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
def softmax(axis=-1,
            aslayer=False,
            reuse=False,
            name=None,
            scope=None):
    def _softmax(x):
        with helper.maybe_layer(aslayer, name, scope, 'softmax', reuse):
            return core.softmax(x, axis)
    return _softmax


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
def softplus(aslayer=False,
             reuse=False,
             name=None,
             scope=None):
    def _softplus(x):
        with helper.maybe_layer(aslayer, name, scope, 'softplus', reuse):
            return core.softplus(x)
    return _softplus


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
def softsign(aslayer=False,
             reuse=False,
             name=None,
             scope=None):
    def _softsign(x):
        with helper.maybe_layer(aslayer, name, scope, 'softsign', reuse):
            return core.softsign(x)
    return _softsign


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
def sigmoid(aslayer=False,
            reuse=False,
            name=None,
            scope=None):
    def _sigmoid(x):
        with helper.maybe_layer(aslayer, name, scope, 'sigmoid', reuse):
            return core.sigmoid(x)
    return _sigmoid


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
def tanh(aslayer=False,
         reuse=False,
         name=None,
         scope=None):
    def _tanh(x):
        with helper.maybe_layer(aslayer, name, scope, 'tanh', reuse):
            return core.tanh(x)
    return _tanh


""" linear activates
    calculates:
        x
"""
def linear(aslayer=False,
           reuse=False,
           name=None,
           scope=None):
    def _linear(x):
        with helper.maybe_layer(aslayer, name, scope, 'linear', reuse):
            return x
    return _linear


def get(act, **kwargs):
    if act is None:
        return linear(**kwargs)
    if isinstance(act, str):
        if act not in ['relu', 'crelu', 'relu6', 'elu', 'selu',
                       'leaky_relu', 'softmax', 'sigmoid', 'squash',
                       'softplus', 'softsign', 'tanh', 'linear']:
            raise ValueError('activation function `{}` not support.'
                            .format(colors.red(act)))
        return eval('{}(**kwargs)'.format(act))
    elif callable(act):
        return act
    else:
        raise ValueError('cannot get activates `{}` with type {}'
                        .format(act, type(act)))
