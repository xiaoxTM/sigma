import tensorflow as tf
from .. import colors
from . import helper

def crelu(name=None):
    if name is None:
        name = helper.dispatch_name('crelu')
    def _crelu(x):
        return tf.nn.crelu(x, name)
    return _crelu


""" relu activates
    calculates:
        max(x, 0)
"""
def relu(name=None):
    if name is None:
        name = helper.dispatch_name('relu')
    def _relu(x):
        return tf.nn.relu(x, name)
    return _relu


""" relu6 activates
    calculates:
        min(max(x, 0), 6)
"""
def relu6(name=None):
    if name is None:
        name = helper.dispatch_name('relu6')
    def _relu6(x):
        return tf.nn.relu6(x, name)
    return _relu6


""" elu activates
    calculates:
        exp(x) -1 if x < 0 else x
"""
def elu(name=None):
    if name is None:
        name = helper.dispatch_name('elu')
    def _elu(x):
        return tf.nn.elu(x, name)
    return _elu



""" elu activates
    calculates:
        scale * alpha * (exp(x) -1)
"""
def selu(alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946, name=None):
    if name is None:
        name = helper.dispatch_name('selu')
    def _selu(x):
        return scale * alpha * (tf.exp(x) - 1)
    return _selu


""" leaky_relu activates
    calculates:
        max(x, alpha*x)
"""
def leaky_relu(alpha=0.2, name=None):
    if name is None:
        name = helper.dispatch_name('leaky_relu')
    if tf.__version__ >= '1.4.0':
        run = tf.nn.leaky_relu
    else:
        def _run(x, alpha, name):
            return tf.maximum(x, x * alpha, name)
        run = _run
    def _leaky_relu(x):
        return run(x, alpha, name)
    return _leaky_relu


""" softmax activates
    calculates:
        exp(x) / reduce_sum(exp(x), dim)
"""
def softmax(dim=-1, name=None):
    if name is None:
        name = helper.dispatch_name('softmax')
    def _softmax(x):
        return tf.nn.softmax(x, dim, name)
    return _softmax


""" softplus activates
    calculates:
        log(exp(x) + 1)
"""
def softplus(name=None):
    if name is None:
        name = helper.dispatch_name('softplus')
    def _softplus(x):
        return tf.nn.softplus(x, name)
    return _softplus


""" softsign activates
    calculates:
        x / (abs(x) + 1)
"""
def softsign(name=None):
    if name is None:
        name = helper.dispatch_name('softsign')
    def _softsign(x):
        return tf.nn.softsign(x, name)
    return _softsign


""" sigmoid activates
    calculates:
        1 / (1 + exp(-x))
"""
def sigmoid(name=None):
    if name is None:
        name = helper.dispatch_name('sigmoid')
    def _sigmoid(x):
        return tf.nn.sigmoid(x, name)
    return _sigmoid


""" hyperbolic tangent activates
    calculates:
        (exp(x) -1) / (exp(x) + 1)
"""
def tanh(name=None):
    if name is None:
        name = helper.dispatch_name('tanh')
    def _tanh(x):
        return tf.nn.tanh(x, name)
    return _tanh


""" linear activates
    calculates:
        x
"""
def linear():
    def _linear(x):
        return x
    return _linear


def get(act):
    if act is None:
        return linear()
    if isinstance(act, str):
        if act not in ['relu', 'crelu', 'relu6', 'elu', 'selu',
                       'leaky_relu', 'softmax', 'sigmoid',
                       'softplus', 'softsign', 'tanh', 'linear']:
            raise ValueError('activation function {}`{}`{} not support.'
                            .format(colors.fg.red, act, colors.reset))
        return eval('{}()'.format(act))
    elif callable(act):
        return act
    else:
        raise ValueError('cannot get activates `{}` with type {}'
                        .format(act, type(act)))
