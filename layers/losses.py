from ..ops import losses, helper, regularizers


def _loss(fun, logits, labels, typename, reuse, name):
    x = fun(logits, labels)
    # helper.print_layer([logits, labels], x, typename, reuse, name)
    return x


def _regularize(fun, inputs, typename, reuse, name):
    x = fun(inputs)
    # helper.print_layer(inputs, x, typename, reuse, name)
    return x


def binary_cross_entropy(logits, labels, axis=None,
                         onehot=True, reuse=False, name=None):
    fun = losses.binary_cross_entropy(axis, onehot, name)
    return _loss(fun, logits, labels, 'binary_cross_entropy',
                 reuse, name)


def categorical_cross_entropy(logits, labels, axis=None,
                              onehot=True, reuse=False, name=None):
    fun = losses.categorical_cross_entropy(axis, onehot, name)
    return _loss(fun, logits, labels, 'categorical_cross_entropy',
                 reuse, name)


def mean_square_error(logits, labels, axis=None,
                      onehot=True, reuse=False, name=None):
    fun = losses.mean_square_error(axis, onehot, name)
    return _loss(fun, logits, labels, 'mean_square_error',
                 reuse, name)


def winner_takes_all(logits, labels, axis=None,
                     onehot=True, reuse=False, name=None):
    fun = losses.winner_takes_all(axis, onehot, name)
    return _loss(fun, logits, labels, 'winner_takes_all',
                 reuse, name)


def total_variation_regularize(inputs, reuse=False, name=None):
    fun = regularizers.total_variation_regularizer(inputs.get_shape().as_list())
    return _regularize(fun, inputs, 'total_variation_regularize',
                       reuse, name)

# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
wta = winner_takes_all
tvr = total_variation_regularize
