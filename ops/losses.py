import tensorflow as tf
from . import helper

def loss(func):
    def _loss(axis, onehot=True, name=None):
        return func(axis, onehot, name)
    return _loss

@loss
def binary_cross_entropy(axis, onehot=True, name=None):
    if name is None:
        name = helper.dispatch_name('binary_cross_entropy')
    scope = tf.name_scope(name)
    def _binary_cross_entropy(logits, labels):
        with scope:
            if not onehot:
                depth = logits.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            return tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                      logits=logits,
                                                      name=name),
              axis=axis)
    return _binary_cross_entropy

@loss
def categorical_cross_entropy(axis, onehot=True, name=None):
    if name is None:
        name = helper.dispatch_name('categorical_cross_entropy')
    scope = tf.name_scope(name)
    def _categorical_cross_entropy(logits, labels):
        with scope:
            if not onehot:
                depth = logits.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            return tf.reduce_mean(
              tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                      logits=logits,
                                                      name=name),
              axis=axis)
    return _categorical_cross_entropy

@loss
def mean_square_error(axis, onehot=True, name=None):
    if name is None:
        name = helper.dispatch_name('mean_square_error')
    scope = tf.name_scope(name)
    def _mean_square_error(logits, labels):
        with scope:
            if not onehot:
                depth = logits.get_shape().as_list()[axis]
                labels = tf.one_hot(labels, depth)
            return tf.reduce_mean(tf.square(logits - labels), axis=axis)
    return _mean_square_error

@loss
def winner_takes_all(axis, onehot=True, name=None):
    if name is None:
        name = helper.dispatch_name('winner_takes_all')
    scope = tf.name_scope(name)
    def _winner_takes_all(logits, labels):
        with scope:
            shape = logits.get_shape().as_list()
            pred = tf.argmax(logits, axis=axis)
            if onehot:
                pred = tf.one_hot(pred, shape[axis])
            loss_tensor = tf.where(pred==labels,
                                   tf.zeros_like(labels),
                                   tf.ones_like(labels))
            loss_matrix = tf.reshape(loss_tensor, (shape[0], -1))
            return tr.reduce_mean(loss_matrix, axis=axis)
    return _winner_takes_all
