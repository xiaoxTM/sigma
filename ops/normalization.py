from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from . import mm, helper, actives, core
from .. import status

def instance_norm(input_shape,
                  offset_initializer='zeros',
                  scale_initializer='ones',
                  offset_regularizer=None,
                  scale_regularizer=None,
                  epsilon=core.epsilon,
                  act=None,
                  trainable=True,
                  collections=None,
                  reuse=False,
                  name=None,
                  scope=None):
    ops_scope, name = helper.assign_scope(name,
                                          scope,
                                          'instance_norm',
                                          reuse)
    input_len = len(input_shape)
    axis = helper.normalize_axes(input_shape)
    neurons = input_shape[axis]
    axes = list(range(input_len))
    del axes[axis]
    del axes[0]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('offset',
                           neurons,
                           core.float32,
                           offset_initializer,
                           offset_regularizer,
                           trainable,
                           collections,
                           reuse,
                           name,
                           scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('scale',
                          neurons,
                          core.float32,
                          scale_initializer,
                          scale_regularizer,
                          trainable,
                          collections,
                          reuse,
                          name,
                          scope)
    act = actives.get(act)
    def _instance_norm(x):
        with ops_scope:
            mean, variance = tf.nn.moments(x, axes, keep_dims=True)
            normalized = (x - mean) / core.sqrt(variance + epsilon)
            if scale is not None:
                normalized = scale * normalized
            if offset is not None:
                normalized = normalized + offset
            return  act(normalized)
    return _instance_norm


layer_norm = instance_norm

""" code inspired by (borrowed from):
        https://github.com/tensorflow/magenta/blob/master
              /magenta/models/image_stylization/ops.py
"""
def conditional_instance_norm(input_shape,
                              bank_size,
                              offset_initializer='zeros',
                              scale_initializer='ones',
                              offset_regularizer=None,
                              scale_regularizer=None,
                              epsilon=core.epsilon,
                              act=None,
                              trainable=True,
                              collections=None,
                              reuse=False,
                              name=None,
                              scope=None):
    ops_scope, name = helper.assign_scope(name,
                                          scope,
                                          'conditional_instance_norm',
                                          reuse)
    input_len = len(input_shape)
    axis = helper.normalize_axes(input_shape)
    neurons = [bank_size, input_shape[axis]]
    axes = list(range(input_len))
    del axes[axis]
    del axes[0]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('offset',
                           neurons,
                           core.float32,
                           offset_initializer,
                           offset_regularizer,
                           trainable,
                           collections,
                           reuse,
                           name,
                           scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('scale',
                          neurons,
                          core.float32,
                          scale_initializer,
                          scale_regularizer,
                          trainable,
                          collections,
                          reuse,
                          name,
                          scope)
    act = actives.get(act)
    def _condition_on(labels):
        select_scale = tf.gather(scale, labels)
        select_offset = tf.gather(offset, labels)
        select_scale = tf.expand_dims(tf.expand_dims(select_scale, 1), 1)
        select_offset = tf.expand_dims(tf.expand_dims(select_offset, 1), 1)
        return select_scale, select_offset
    
    def _conditional_instance_norm(x, labels):
        with ops_scope:
            mean, variance = tf.nn.moments(x, axes, keep_dims=True)
            normalized = (x - mean) / core.sqrt(variance + epsilon)
            select_scale, select_offset = _condition_on(labels)
            if select_scale is not None:
                normalized = select_scale * normalized
            if select_offset is not None:
                normalized = normalized + select_offset
            return  act(normalized)
    return _conditional_instance_norm


def batch_norm(input_shape,
               momentum=0.99,
               offset_initializer='zeros',
               scale_initializer='ones',
               offset_regularizer=None,
               scale_regularizer=None,
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               epsilon=core.epsilon,
               act=None,
               trainable=True,
               fused=False,
               collections=None,
               reuse=False,
               name=None,
               scope=None):
    """ batch normalization layer
        Attributes
        ==========
        input_shape : list / tuple
                      input tensor shape
        momentum : float | None
                   momentum to update moving mean and variance
                   if None, moving mean and variance
                   will not be updated
        offset_initializer : string / callable function | None | bool
                             initializer to initialize offset
                             if False, offset will be ignored
                             (output will not be centered)
        offset_regularizer : string
                             penalty for offset
        scale_initializer : string / callable function | None | bool
                            initializer to initialize scale
                            if False, scale will be ignored
        scale_regularizer : string
                            penalty for scale
        moving_mean_initializer : string / callable function | None
                                  initializer for moving_mean
        moving_variance_initializer : string / callable function | None
                                      initializer for moving_variance
        is_training : bool
        fused : bool
                use fused_batch_normal if true
    """

    ops_scope, name = helper.assign_scope(name,
                                          scope,
                                          'batch_norm',
                                          reuse)
    axis = list(range(len(input_shape)-1))
    if fused:
        axis = [0 ,1, 2]
    # if not isinstance(axis, (list, tuple)):
    #     axis = [axis]
    neurons = input_shape[core.axis]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('offset',
                           neurons,
                           core.float32,
                           offset_initializer,
                           offset_regularizer,
                           trainable,
                           collections,
                           reuse,
                           name,
                           scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('scale',
                          neurons,
                          core.float32,
                          scale_initializer,
                          scale_regularizer,
                          trainable,
                          collections,
                          reuse,
                          name,
                          scope)

    moving_mean = None
    moving_mean = mm.malloc('moving-mean',
                            neurons,
                            core.float32,
                            moving_mean_initializer,
                            None,
                            trainable,
                            collections,
                            reuse,
                            name,
                            scope)

    moving_variance = None
    moving_variance = mm.malloc('moving-variance',
                                neurons,
                                core.float32,
                                moving_variance_initializer,
                                None,
                                trainable,
                                collections,
                                reuse,
                                name,
                                scope)
    act = actives.get(act)
    # print('input shape:', input_shape)
    # print('axis:', axis)
    def _train(x):
        if fused:
            # tf.nn.fused_batch_norm(x, scale, offset, mean=None, variance=None,
            #                        epsionl=0.001, is_training=True, name=None)
            # x must be 4-d tensor
            # mean / variance used for inference
            x_shape = core.shape(x)
            for _ in range(4 - x.get_shape().ndims):
                x = tf.expand_dims(x, 1)
            # print('x shape:', x.get_shape().as_list())
            x, mean, variance = tf.nn.fused_batch_norm(x, scale,
                                                       offset, epsilon=epsilon)
            x = core.reshape(x, x_shape)
        else:
            mean, variance = tf.nn.moments(x, axis, keep_dims=True)
            # tf.nn.batch_normalize(x, mean, variance, offset,
            #                       scale, variance_epsilon, name)
            x = tf.nn.batch_normalization(x, mean, variance,
                                          offset, scale, epsilon)
            mean = tf.squeeze(mean)
            variance = tf.squeeze(variance)
        if momentum is not None:
            # print('mm:', moving_mean.get_shape().as_list())
            # print('mn', mean.get_shape().as_list())
            moving_mean.assign(moving_mean * momentum + mean * (1 - momentum))
            moving_variance.assign(
                       moving_variance * momentum + variance * (1 - momentum))
        return act(x)

    def _infer(x):
        if fused:
            x_shape = core.shape(x)
            for _ in range(4 - x.get_shape().ndims):
                x = tf.expand_dims(x, 1)
            x, mean, variance = tf.nn.fused_batch_norm(x, scale,
                                         offset, moving_mean, moving_variance,
                                         epsilon, is_training=False)
            x = core.reshape(x, x_shape)
        else:
            mean, variance = tf.nn.moments(x, axis, keep_dims=True)
            x = tf.nn.batch_normalization(x, moving_mean, moving_variance,
                                          offset, scale, epsilon)
        return act(x)

    def _batch_norm(x):
        with ops_scope:
            x = tf.cond(tf.cast(status.is_training, tf.bool),
                        lambda: _train(x),
                        lambda: _infer(x))
            # if update moving_mean and moving_variance
            # print('x shape:', x.get_shape().as_list())
            return x
    return _batch_norm


def dropout(pkeep,
            noise_shape=None,
            seed=None,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, name = helper.assign_scope(name,
                                          scope,
                                          'dropout',
                                          reuse)
    def _dropout(x):
        with ops_scope:
            return tf.nn.dropout(x, pkeep, noise_shape, seed, name)
    return _dropout
