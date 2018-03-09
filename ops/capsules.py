from .. import colors
from . import core, actives, helper, mm
import tensorflow as tf

""" code borrowed from:
    https://github.com/Sarasra/models/blob/master/research/capsules/models/layers/layers.py
"""

def _leaky_routing(logits):
    """ leaky routing
        This enables active capsules to be routed to the added
        parent capsule if they are not good fit for any of the
        parent capsules

        Attributes
        ==========
        logits : tf.Tensor
                 output tensor of one layer with shape of:
                 [batch-size, ncaps, caps_dim]
                 for fully connected
                 or
                 [batch-size, rows, cols, ncaps, caps_dim]
                 for conv2d
                 NOTE that, logits is not `activated` by
                 `softmax` or `sigmoid`
    """
    shape = core.shape(logits)
    nouts = shape[core.axis]
    shape[core.axis] = 1
    leak = core.zeros(shape)
    # add an extra dimension to routes
    leaky_logits = core.concat([leak, logits], axis=core.axis)
    # routes to capsules who has max probability
    leaky_routs = core.softmax(leaky_logits, axis=core.axis)
    # remove added dimension
    return core.split(leaky_routs, [1, nouts], axis=core.axis)[1]


def _agreement_routing(prediction_vectors, logits, iterations,
                       leaky=False):
    """ calculate v_j by dynamic routing
        Attributes
        ==========
        prediction_vectors : tf.Tensor
                             predictions from previous layers
                             denotes u_{j|i}_hat in paper
                             with the shape of
                             [batch-size, incaps, outcaps, outcapdim]
                             for fully_connected
                             [batch-size, neurons, incaps, outcaps, outcapdim]
                             for conv1d
                             [batch-size, nrows, ncols, incaps, outcaps, outcapdim]
                             for conv2d
        logits : tf.Tensor
                 with the shape of
                 [1, incaps, outcaps, 1]
                 for fully_connected
                 [1, 1, incaps, outcaps, 1]
                 for conv1d
                 [1, 1, 1, incaps, outcaps, 1]
                 for conv2d
        iterations : int
                     iteration times to adjust c_{i, j}.
                     donates r in paper
        leaky : boolean
                whether leaky_routing or not

        Returns
        ==========
        Activated tensor of the output layer. That is v_j
        with shape of
        [batch-size, incaps, outcaps, outcapdim]
        for fully_connected
        [batch-size, neurons, incaps, outcaps, outcapdim]
        for conv1d
        [batch-size, nrows, ncols, incaps, outcaps, outcapdim]
        for conv2d
    """
    shape = core.shape(prediction_vectors)
    shape.pop(-3)

    act = actives.squash()

    def _update(i, logits, activations):
        """ dynamic routing to update coefficiences (c_{i, j})
        """
        #print('logits shape:', core.shape(logits))
        if leaky:
            coefficients = _leaky_routing(logits)
        else:
            coefficients = core.softmax(logits, axis=-1)
        #tf.summary.histogram('coefficients-{}'.format(i), coefficients)
        preactivate = coefficients * prediction_vectors
        #print('preactivate:', core.shape(preactivate))
        preactivate = core.sum(preactivate, axis=-3, keepdims=True)
        #tf.summary.histogram('preactivate-{}'.format(i), preactivate)
        #print('summed preactivate:', core.shape(preactivate))
        activation = act(preactivate)
        #tf.summary.histogram('activation-{}'.format(i), activation)
        #print('activation:', core.shape(activation))
        activations = activations.write(i, activation)
        # sum up along outcapdim dimension
        distance = core.sum(prediction_vectors * activation,
                            axis=core.axis,
                            keepdims=True)
        #tf.summary.histogram('distance-{}'.format(i), distance)
        logits += distance
        #tf.summary.histogram('logits-{}'.format(i), logits)
        return (i+1, logits, activations)

    activations = core.TensorArray(dtype=core.float32,
                                   size=iterations,
                                   clear_after_read=False)
    idx = core.constant(0, dtype=core.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < iterations,
        _update,
        loop_vars=[idx, logits, activations])
    #   [batch-size, nrows, ncols, 1, outcaps, outcapdim]
    # =>[batch-size, nrows, ncols, outcaps, outcapdim]
    return core.reshape(activations.read(iterations-1), shape)


def conv(convop, weight_shape, logits_shape, iterations,
         leaky=False,
         weight_initializer='glorot_uniform',
         weight_regularizer=None,
         logits_initializer='zeros',
         logits_regularizer=None,
         act=None,
         trainable=True,
         dtype=core.float32,
         collections=None,
         summarize=True,
         reuse=False,
         name=None,
         scope=None):
    """ coefficient_shape : [1, 1, 1, incaps, outcaps, 1]
                             for conv2d operation
    """
    ops_scope, _, name = helper.assign_scope(name,
                                          scope,
                                          'caps_'+convop.__name__[1:],
                                          reuse)
    act = actives.get(act)
    weight = mm.malloc('weight',
                       name,
                       weight_shape,
                       dtype,
                       weight_initializer,
                       weight_regularizer,
                       trainable,
                       collections,
                       reuse,
                       scope)
    if summarize and not reuse:
        tf.summary.histogram(weight.name, weight)
    logits = mm.malloc('logits',
                       name,
                       logits_shape,
                       dtype,
                       logits_initializer,
                       logits_regularizer,
                       False,
                       collections,
                       reuse,
                       scope)
    if summarize and not reuse:
        tf.summary.histogram(logits.name, logits)
    def _conv(x):
        with ops_scope:
            #  [batch-size, nrows, ncols, incaps, incapdim] x [incapdim, outcaps, outcapdim]
            #=>[batch-size, ncols, nrows, incaps, outcaps, outcapdim]
            x = convop(x)
            x = core.tensordot(x, weight, axes=[[-1], [0]])
            x = _agreement_routing(x, logits, iterations, leaky)
            return x
    return _conv


def fully_connected(input_shape, nouts, caps_dims,
                    iterations=2,
                    leaky=False,
                    weight_initializer='glorot_uniform',
                    weight_regularizer=None,
                    logits_initializer='zeros', # no bias
                    logits_regularizer=None,
                    act=None,
                    trainable=True,
                    dtype=core.float32,
                    collections=None,
                    summarize=True,
                    reuse=False,
                    name=None,
                    scope=None):
    """ fully_connected layer for capsule networks
        Attributes
        ==========
        input_shape : list / tuple
                      input tensor shape. Should in form of:
                      [batch-size, incaps, incapdim]
        nouts : int
                output number of capsules
        caps_dims : int
                    output capsule dimension
    """
    if len(input_shape) != 3:
        raise ValueError('capsule fully_connected require input shape {}[batch-size,'
                         ' incaps, incapdim]{}, given {}{}{}'
                         .format(colors.fg.green, colors.reset,
                                 colors.fg.red, input_shape, colors.reset))
    output_shape = [input_shape[0], nouts, caps_dims]
    incaps, incapdim = input_shape[-2:]
    logits_shape = [input_shape[0], incaps, nouts, 1]
    weight_shape = [incapdim, nouts, caps_dims]
    def _fully_connected(x):
        return x
    return conv(_fully_connected,
                weight_shape,
                logits_shape,
                iterations,
                leaky,
                weight_initializer,
                weight_regularizer,
                logits_initializer,
                logits_regularizer,
                act,
                trainable,
                dtype,
                collections,
                summarize,
                reuse,
                name,
                scope), output_shape


def conv2d(input_shape, nouts, caps_dims, kshape,
           iterations=2,
           leaky=False,
           stride=1,
           padding='valid',
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           logits_initializer='zeros',
           logits_regularizer=None,
           act=None,
           trainable=True,
           dtype=core.float32,
           collections=None,
           summarize=True,
           reuse=False,
           name=None,
           scope=None):
    """ primary capsule convolutional
        Attributes
        ==========
        input_shape : list / tuple
                      should have form of:
                      [batch-size, rows, cols, incaps, incaps_dim]
                      where `rows/cols` denotes the row/col of matrix
                      `incaps_dim` denotes the vector size of each capsule
                      (as depth channels)
                      `incaps` means the number of capsules
        nouts : int
                number of output capsules
        caps_dims : int
                    output capsule vector size (aka. outcapdim)
        kshape : int / list / tuple
                 kernel shape for convolving operation
    """
    if len(input_shape) != 5:
        raise ValueError('capsule conv2d require input shape {}[batch-size, '
                         'rows, cols, incaps, incapdim]{}, given {}{}{}'
                         .format(colors.fg.green, colors.reset, colors.fg.red,
                                 input_shape, colors.reset))
    kshape = helper.norm_input_2d(kshape)
    stride = helper.norm_input_2d(stride)

    input_nshape = input_shape[:]
    #  [batch-size, rows, cols, incaps, incapdim]
    #=>[batch-size, rows, cols, incpas * incapdim]
    input_nshape[core.axis] *= input_nshape[-2]
    input_nshape.pop(-2)
    # output shape may be not right
    output_shape = helper.get_output_shape(input_nshape, nouts * caps_dims,
                                           kshape, stride, padding)
    output_shape = output_shape[:-1] + [nouts, caps_dims]
    # depthwise_conv2d kernel shape:
    # [krow, kcol, incaps * incapdim, 1]
    kernel_shape = kshape[1:-1] +[input_shape[-2] * input_shape[core.axis], 1]
    kernel = mm.malloc('kernel',
                       name,
                       kernel_shape,
                       dtype,
                       weight_initializer,
                       weight_regularizer,
                       trainable,
                       collections,
                       reuse,
                       scope)
    if summarize and not reuse:
        tf.summary.histogram(kernel.name, kernel)

    incaps, incapdim = input_shape[-2:]
    #  for data_format = 'NCHW',
    #  coefficient_shape = [batch-size, rows, cols, incaps, nouts, 1]
    logits_shape = output_shape[:3] + [incaps, nouts, 1]
    weight_shape = [incapdim, nouts, caps_dims]

    def _conv2d(x):
        x = core.reshape(x, input_nshape)
        x = core.depthwise_conv2d(x, kernel, stride, padding.upper())
        xshape = input_shape[:]
        xshape[1:3] = core.shape(x)[1:3]
        #   [batch-size, nrows, ncols, incaps * incapdim]
        # =>[batch-size, nrows, ncols, incaps, incapdim]
        return core.reshape(x, xshape)
    return conv(_conv2d,
                weight_shape,
                logits_shape,
                iterations,
                leaky,
                weight_initializer,
                weight_regularizer,
                logits_initializer,
                logits_regularizer,
                act,
                trainable,
                dtype,
                collections,
                summarize,
                reuse,
                name,
                scope), output_shape
