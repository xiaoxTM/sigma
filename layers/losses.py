from ..ops import losses, helper, regularizers, core
from .core import layer

""" termology
    ----------
        logits : unnormalized output from network
                 logits generally is the input to softmax
"""


@layer
def binary_cross_entropy(inputs, labels,
                         axis=None,
                         logits=True,
                         onehot=True,
                         reuse=False,
                         name=None,
                         scope=None):
    return losses.binary_cross_entropy(axis,
                                       logits,
                                       onehot,
                                       reuse,
                                       name,
                                       scope)(inputs, labels)


@layer
def categorical_cross_entropy(inputs, labels,
                              axis=None,
                              logits=True,
                              onehot=True,
                              reuse=False,
                              name=None,
                              scope=None):
    return losses.categorical_cross_entropy(axis,
                                            logits,
                                            onehot,
                                            reuse,
                                            name,
                                            scope)(inputs, labels)


@layer
def mean_square_error(inputs, labels,
                      axis=None,
                      logits=True,
                      onehot=True,
                      reuse=False,
                      name=None,
                      scope=None):
    return losses.mean_square_error(axis,
                                    logits,
                                    onehot,
                                    reuse,
                                    name,
                                    scope)(inputs, labels)


@layer
def mean_absolute_error(inputs, labels,
                        axis=None,
                        logits=True,
                        onehot=True,
                        reuse=False,
                        name=None,
                        scope=None):
    return losses.mean_absolute_error(axis,
                                      logits,
                                      onehot,
                                      reuse,
                                      name,
                                      scope)(inputs, labels)


@layer
def winner_takes_all(inputs, labels,
                     axis=None,
                     logits=True,
                     onehot=True,
                     reuse=False,
                     name=None,
                     scope=None):
    return losses.winner_takes_all(axis,
                                   logits,
                                   onehot,
                                   reuse,
                                   name,
                                   scope)(inputs, labels)


@layer
def total_variation_regularize(inputs,
                               reuse=False,
                               name=None,
                               scope=None):
    shape = core.shape(inputs)
    return regularizers.total_variation_regularizer(shape,
                                                    reuse,
                                                    name,
                                                    scope)(inputs)

# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
mae = mean_absolute_error
wta = winner_takes_all
tvr = total_variation_regularize
