from .. import ops
from . import core

""" termology
    ----------
        logits : unnormalized output from network
                 logits generally is the input to softmax
"""


@core.layer
def binary_cross_entropy(inputs, labels,
                         axis=None,
                         logits=True,
                         onehot=True,
                         reuse=False,
                         name=None,
                         scope=None):
    return ops.losses.binary_cross_entropy(axis,
                                           logits,
                                           onehot,
                                           reuse,
                                           name,
                                           scope)(inputs, labels)


@core.layer
def categorical_cross_entropy(inputs, labels,
                              axis=None,
                              logits=True,
                              onehot=True,
                              reuse=False,
                              name=None,
                              scope=None):
    return ops.losses.categorical_cross_entropy(axis,
                                                logits,
                                                onehot,
                                                reuse,
                                                name,
                                                scope)(inputs, labels)


@core.layer
def mean_square_error(inputs, labels,
                      axis=None,
                      logits=True,
                      onehot=True,
                      reuse=False,
                      name=None,
                      scope=None):
    return ops.losses.mean_square_error(axis,
                                        logits,
                                        onehot,
                                        reuse,
                                        name,
                                        scope)(inputs, labels)


@core.layer
def mean_absolute_error(inputs, labels,
                        axis=None,
                        logits=True,
                        onehot=True,
                        reuse=False,
                        name=None,
                        scope=None):
    return ops.losses.mean_absolute_error(axis,
                                          logits,
                                          onehot,
                                          reuse,
                                          name,
                                          scope)(inputs, labels)


@core.layer
def winner_takes_all(inputs, labels,
                     axis=None,
                     logits=True,
                     onehot=True,
                     reuse=False,
                     name=None,
                     scope=None):
    return ops.losses.winner_takes_all(axis,
                                       logits,
                                       onehot,
                                       reuse,
                                       name,
                                       scope)(inputs, labels)

@core.layer
def margin_loss(inputs, labels,
                axis=None,
                positive_margin=0.9,
                negative_margin=0.1,
                downweighting=0.5,
                logits=True,
                onehot=True,
                reuse=False,
                name=None,
                scope=None):
    """ margin loss for capsule networks
        NOTE margin_loss cannot be used for normal CNN
        because margin loss treat inputs as
        `vector in vector out` tensor 
    """
    return ops.losses.margin_loss(axis,
                                  logits,
                                  onehot,
                                  reuse,
                                  name,
                                  scope,
                                  positive_margin,
                                  negative_margin,
                                  downweighting)(inputs, labels)


# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
mae = mean_absolute_error
wta = winner_takes_all


def get(act, *args, **kwargs):
    if isinstance(act, str):
        return eval('{}(*args, **kwargs)'.format(act))
    elif callable(act):
        return act
    else:
        raise ValueError('cannot get activates `{}` with type {}'
                         .format(initializer, type(initializer)))