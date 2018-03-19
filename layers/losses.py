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
                         from_logits=True,
                         onehot=True,
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
                                           scope)(inputs, labels)


@core.layer
def categorical_cross_entropy(inputs,
                              axis=None,
                              from_logits=True,
                              onehot=True,
                              reuse=False,
                              name=None,
                              scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.losses.categorical_cross_entropy(axis,
                                                from_logits,
                                                onehot,
                                                reuse,
                                                name,
                                                scope)(inputs, labels)


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
                        from_logits=True,
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
                     from_logits=True,
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
                downweighting=0.5,
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
                                  downweighting)(inputs, labels)


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
