import tensorflow as tf
from . import helper, core


def loss(fun):
    def _loss(axis,
              from_logits=True,
              onehot=True,
              reuse=False,
              name=None,
              scope=None,
              *args):
        ops_scope, _, name = helper.assign_scope(name,
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
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                        logits=x,
                                                        name=name),
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
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=x,
                                                            name=name),
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
            loss_tensor = tf.where(pred==labels,
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
                downweighting=0.5):
    def _margin_loss(x, labels):
        with scope:
            if not onehot:
                depth = core.shape(x)[axis]
                labels = core.one_hot(labels, depth)
            pmask = core.cast(core.less(x, positive_margin), core.float32)
            ploss = pmask * core.pow(positive_margin-x, 2)
            ploss = core.sum(core.cast(labels, core.float32) * ploss,
                             axis=core.axis)
            nmask = core.cast(core.less(negative_margin, x), core.float32)
            nloss = nmask * core.pow(negative_margin-x, 2)
            nloss = core.sum(core.cast(1-labels, core.float32) * nloss,
                             axis=core.axis)
            return core.mean(ploss + downweighting * nloss)
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
