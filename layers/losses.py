from ..ops import losses, helper
from ..ops import regularizers
from . import layers

def _loss(fun, logits, labels, typename, reuse, name):
    x = fun(logits, labels)
    helper.print_layer([logits, labels], x, typename, reuse, name)
    return x


def _regularize(fun, inputs, typename, reuse, name):
    x = fun(inputs)
    helper.print_layer(inputs, x, typename, reuse, name)
    return x


@layers
def binary_cross_entropy(logits, labels, axis=None,
                         onehot=True, reuse=False, name=None):
    fun = losses.binary_cross_entropy(axis, onehot, name)
    return _loss(fun, logits, labels, 'binary_cross_entropy',
                 reuse, name)


@layers
def categorical_cross_entropy(logits, labels, axis=None,
                              onehot=True, reuse=False, name=None):
    fun = losses.categorical_cross_entropy(axis, onehot, name)
    return _loss(fun, logits, labels, 'categorical_cross_entropy',
                 reuse, name)


@layers
def mean_square_error(logits, labels, axis=None,
                      onehot=True, reuse=False, name=None):
    fun = losses.mean_square_error(axis, onehot, name)
    return _loss(fun, logits, labels, 'mean_square_error',
                 reuse, name)


@layers
def winner_takes_all(logits, labels, axis=None,
                     onehot=True, reuse=False, name=None):
    fun = losses.winner_takes_all(axis, onehot, name)
    return _loss(fun, logits, labels, 'winner_takes_all',
                 reuse, name)


@layers
def total_variation_regularize(inputs, reuse=False, name=None):
    fun = regularizers.total_variation_regularizer(inputs.get_shape().as_list())
    return _regularize(fun, inputs, 'total_variation_regularize',
                       reuse, layers.graph, name)

# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
wta = winner_takes_all
tvr = total_variation_regularize
