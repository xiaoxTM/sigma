import tensorflow as tf
from . import helper

def binary_cross_entropy(axis, name=None):
    if name is None:
        name = helper.dispatch_name('bce')
    scope = tf.name_scope(name)
    def _binary_cross_entropy(logits, labels):
        with scope:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                          logits=logits,
                                                                          name=name),
                                  axis=axis)
    return _binary_cross_entropy


def categorical_cross_entropy(axis, name=None):
    if name is None:
        name = helper.dispatch_name('cce')
    scope = tf.name_scope(name)
    def _categorical_cross_entropy(logits, labels):
        with scope:
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                          logits=logits,
                                                                          name=name),
                                  axis=axis)
    return _categorical_cross_entropy


def mean_square_error(axis, name=None):
    if name is None:
        name = helper.dispatch_name('mse')
    scope = tf.name_scope(name)
    def _mean_square_error(logits, labels):
        with scope:
            return tf.reduce_mean(tf.square(logits - labels), axis=axis)
    return _mean_square_error
