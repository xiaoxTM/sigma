import tensorflow as tf
from . import mm
from . import helper
from .. import status

def batch_norm(input_shape,
               momentum=0.99,
               offset_initializer=None,
               scale_initializer=None,
               offset_regularizer=None,
               scale_regularizer=None,
               moving_mean_initializer=None,
               moving_variance_initializer=None,
               epsilon=1e-5,
               fused=False,
               reuse=False,
               collections=None,
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

    if name is  None:
        if fused:
            name = helper.dispatch_name('fused_batch_norm')
        else:
            name = helper.dispatch_name('batch_norm')

    if scope is None:
        scope = name

    axis = list(range(len(input_shape)-1))
    if fused:
        axis = [0 ,1, 2]
    # if not isinstance(axis, (list, tuple)):
    #     axis = [axis]
    neurons = input_shape[-1]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('{}-offset'.format(name), neurons, tf.float32, offset_initializer,
                           offset_regularizer, trainable=True,
                           collections=collections, reuse=reuse,
                           scope=scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('{}-scale'.format(name), neurons, tf.float32,
                          scale_initializer,
                          scale_regularizer, trainable=True,
                          collections=collections, reuse=reuse,
                          scope=scope)

    moving_mean = None
    moving_mean = mm.malloc('{}-moving-mean'.format(name),
                            neurons, tf.float32,
                            moving_mean_initializer,
                            trainable=True,
                            collections=collections,
                            reuse=reuse, scope=scope)

    moving_variance = None
    moving_variance = mm.malloc('{}-moving-variance'.format(name),
                                neurons, tf.float32,
                                moving_variance_initializer,
                                trainable=True,
                                collections=collections,
                                reuse=reuse, scope=scope)
    # print('input shape:', input_shape)
    # print('axis:', axis)
    def _train(x):
        if fused:
            # tf.nn.fused_batch_norm(x, scale, offset, mean=None, variance=None,
            #                        epsionl=0.001, is_training=True, name=None)
            # x must be 4-d tensor
            # mean / variance used for inference
            x_shape = x.get_shape().as_list()
            for _ in range(4 - x.get_shape().ndims):
                x = tf.expand_dims(x, 1)
            # print('x shape:', x.get_shape().as_list())
            x, mean, variance = tf.nn.fused_batch_norm(x, scale, offset, epsilon=epsilon)
            x = tf.reshape(x, x_shape)
        else:
            mean, variance = tf.nn.moments(x, axis, keep_dims=True)
            # tf.nn.batch_normalize(x, mean, variance, offset,
            #                       scale, variance_epsilon, name)
            x = tf.nn.batch_normalization(x, mean, variance, offset, scale, epsilon)
            mean = tf.squeeze(mean)
            variance = tf.squeeze(variance)
        if momentum is not None:
            # print('mm:', moving_mean.get_shape().as_list())
            # print('mn', mean.get_shape().as_list())
            moving_mean.assign(moving_mean * momentum + mean * (1 - momentum))
            moving_variance.assign(moving_variance * momentum + variance * (1 - momentum))
        return x

    def _infer(x):
        if fused:
            x_shape = x.get_shape().as_list()
            for _ in range(4 - x.get_shape().ndims):
                x = tf.expand_dims(x, 1)
            x, mean, variance = tf.nn.fused_batch_norm(x, scale, offset, moving_mean,
                                         moving_variance, epsilon, is_training=False)
            x = tf.reshape(x, x_shape)
        else:
            mean, variance = tf.nn.moments(x, axis, keep_dims=True)
            x = tf.nn.batch_normalization(x, moving_mean, moving_variance,
                                          offset, scale, epsilon)
        return x

    def _batch_norm(x):
        x = tf.cond(tf.cast(status.is_training, tf.bool),
                    lambda: _train(x),
                    lambda: _infer(x))
        # if update moving_mean and moving_variance
        # print('x shape:', x.get_shape().as_list())
        return x
    return _batch_norm


def layer_norm(input_shape, scale=True, epsilon=1e-5,
               reuse=False, name=None, scope=None):
    if name is None:
        name = helper.dispatch_name('layer_norm')
    if scope is None:
        scope = tf.name_scope(name)
    axis = len(input_shape) - 1
    neurons = [input_shape[0]]
    offset = mm.malloc('{}-offset'.format(name), neurons, tf.float32,
                       initializer=tf.zeros_initializer,
                       trainable=True, reuse=reuse,
                       scope=scope)
    if scale is True:
        scale = mm.malloc('{}-scale'.format(name),
                          [neurons], dtype=tf.float32,
                          initializer=tf.ones_initializer,
                          trainable=True, reuse=reuse,
                          scope=scope)

    def _layer_norm(x):
        mean, var = tf.nn.moments(x, [axis], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + epsilon)
        if scale:
            x = scale * x + offset
        else:
            x = x + offset
        return x

    return _layer_norm


def dropout(pkeep, noise_shape=None, seed=None, name=None):
    if name is None:
        name = helper.dispatch_name('dropout')
    scope = tf.name_scope(name)
    def _dropout(x):
        with scope:
            return tf.nn.dropout(x, pkeep, noise_shape, seed, name)
    return _dropout
