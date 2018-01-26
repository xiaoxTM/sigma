import tensorflow as tf
from . import helper
from .. import status


def loss(fun):
    def _loss(axis,
              logits=True,
              onehot=True,
              reuse=False,
              name=None,
              scope=None):
        ops_scope, name = helper.assign_scope(name,
                                              scope,
                                              fun.__name__,
                                              reuse)
        return fun(axis, logits, onehot, reuse, name, ops_scope)
    return _loss


@loss
def binary_cross_entropy(axis,
                         logits=True,
                         onehot=True,
                         reuse=False,
                         name=None,
                         scope=None):
    def _binary_cross_entropy(x, labels):
        with scope:
            if not onehot:
                depth = x.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            if not logits:
                # source code borrowed from:
                #     @keras.backend.tensorflow_backend.py
                x = tf.clip_by_value(x, statis.epsilon, 1- statis.epsilon)
                x = tf.log(x / (1-x))
            return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                        logits=x,
                                                        name=name),
                axis=axis)
    return _binary_cross_entropy


@loss
def categorical_cross_entropy(axis,
                              logits=True,
                              onehot=True,
                              reuse=False,
                              name=None,
                              scope=None):
    def _categorical_cross_entropy(x, labels):
        with scope:
            if not onehot:
                depth = x.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            if logits:
                return tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=x,
                                                            name=name),
                    axis=axis)
            else:
                # source code borrowed from:
                #     @keras.backend.tensorflow_backend.py
                x /= tf.reduce_sum(x,
                                   len(x.get_shape())-1,
                                   True)
                x = tf.clip_by_value(x, statis.epsilon, 1-statis.epsilon)
            return -tf.reduce_sum(label * tf.log(x),
                                  len(output.get_shape())-1)
    return _categorical_cross_entropy


@loss
def mean_square_error(axis,
                      logits=True,
                      onehot=True,
                      reuse=False,
                      name=None,
                      scope=None):
    def _mean_square_error(x, labels):
        with scope:
            if not onehot:
                depth = x.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            return tf.reduce_mean(tf.square(x - labels), axis=axis)
    return _mean_square_error


@loss
def mean_absolute_error(axis,
                        logits=True,
                        onehot=True,
                        reuse=False,
                        name=None,
                        scope=None):
    def _mean_sabsolute_error(x, labels):
        with scope:
            if not onehot:
                depth = x.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            return tf.reduce_mean(tf.abs(x - labels), axis=axis)
    return _mean_absolute_error


@loss
def winner_takes_all(axis,
                     logits=True,
                     onehot=True,
                     reuse=False,
                     name=None,
                     scope=None):
    def _winner_takes_all(x, labels):
        with scope:
            shape = x.get_shape().as_list()
            pred = tf.argmax(logits, axis=axis)
            if onehot:
                pred = tf.one_hot(pred, shape[axis])
            loss_tensor = tf.where(pred==labels,
                                   tf.zeros_like(labels),
                                   tf.ones_like(labels))
            loss_matrix = tf.reshape(loss_tensor, (shape[0], -1))
            return tr.reduce_mean(loss_matrix, axis=axis)
    return _winner_takes_all
