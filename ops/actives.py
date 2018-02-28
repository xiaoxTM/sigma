from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .. import colors
from . import helper, core

def squash(epsilon=core.epsilon, reuse=False, name=None, scope=None):
    """ squash function to range input into [0, 1) as activation
        used for capsule networks. see :
        @https://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf

        input (x) should with shape of
        [batch-size, ncaps, caps_dim] for fully_connected capsule layer
        or
        [batch-size, units, ncaps, caps_dim] for conv1d
        or
        [batch-size, rows, cols, ncaps, caps_dim] for conv2d
        capsule layer
        where `ncaps` denotes the number of capsules
        and `cap_dim` denotes the dimension of each capsule
    """
    ops_scope, name = helper.assign_scope(name, scope, 'squash', reuse)
    def _squash(x):
        with ops_scope:
            squared_norm = core.sum(core.square(x), axis=core.axis, keepdims=True)
            # if squared_norm contains 0 element
            # add \epsilon to avoid dividing 0
            norm = core.cond(core.all(squared_norm),
                             lambda : core.sqrt(squared_norm),
                             lambda : core.sqrt(squared_norm + epsilon)
                            )
            return (x / norm) * (squared_norm / (1 + squared_norm))
    return _squash


def crelu(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'crelu', reuse)
    def _crelu(x):
        with ops_scope:
            return core.crelu(x, name)
    return _crelu


""" relu activates
    calculates:
        max(x, 0)
"""
def relu(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'relu', reuse)
    def _relu(x):
        with ops_scope:
            return core.relu(x, name)
    return _relu


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
def relu6(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'relu6', reuse)
    def _relu6(x):
        with ops_scope:
            return core.relu6(x, name)
    return _relu6


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
def elu(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'elu', reuse)
    def _elu(x):
        with ops_scope:
            return core.elu(x, name)
    return _elu


""" elu activates
    calculates:
        scale * alpha * (exp(x) -1)
"""
def selu(alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946,
         reuse=False,
         name=None,
         scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'selu', reuse)
    def _selu(x):
        with ops_scope:
            return scale * alpha * (core.exp(x) - 1)
    return _selu


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
def leaky_relu(alpha=0.2, reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'leaky_relu', reuse)
    if tf.__version__ >= '1.4.0':
        run = core.leaky_relu
    else:
        def _run(x, alpha, name):
            return core.maximum(x, x * alpha, name)
        run = _run
    def _leaky_relu(x):
        with ops_scope:
            return core.leaky_relu(x, alpha, name)
    return _leaky_relu


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
def softmax(dim=-1, reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'softmax', reuse)
    def _softmax(x):
        with ops_scope:
            return core.softmax(x, dim, name)
    return _softmax


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
def softplus(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'softplus', reuse)
    def _softplus(x):
        with ops_scope:
            return core.softplus(x, name)
    return _softplus


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
def softsign(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'softsign', reuse)
    def _softsign(x):
        with ops_scope:
            return core.softsign(x, name)
    return _softsign


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
def sigmoid(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'sigmoid', reuse)
    def _sigmoid(x):
        with ops_scope:
            return core.sigmoid(x, name)
    return _sigmoid


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
def tanh(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'tanh', reuse)
    def _tanh(x):
        with ops_scope:
            return core.tanh(x, name)
    return _tanh


""" linear activates
    calculates:
        x
"""
def linear(reuse=False, name=None, scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'linear', reuse)
    def _linear(x):
        with ops_scope:
            return x
    return _linear


def get(act, **kwargs):
    if act is None:
        return linear(**kwargs)
    if isinstance(act, str):
        if act not in ['relu', 'crelu', 'relu6', 'elu', 'selu',
                       'leaky_relu', 'softmax', 'sigmoid',
                       'softplus', 'softsign', 'tanh', 'linear']:
            raise ValueError('activation function {}`{}`{} not support.'
                            .format(colors.fg.red, act, colors.reset))
        return eval('{}(**kwargs)'.format(act))
    elif callable(act):
        return act
    else:
        raise ValueError('cannot get activates `{}` with type {}'
                        .format(act, type(act)))
