from .. import colors
from . import core, actives, helper, mm
import tensorflow as tf

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


def _agreement_routing(prediction_vectors,
                       logits_shape,
                       iterations,
                       bias,
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
    activations = core.TensorArray(dtype=core.float32,
                                   size=iterations,
                                   clear_after_read=False)
    logits = tf.fill(logits_shape, 0.0)
    act = actives.squash()
    idx = core.constant(0, dtype=core.int32)
    capsule_axis = helper.normalize_axes(core.shape(logits), -3)

    def _update(i, logits, activations):
        """ dynamic routing to update coefficiences (c_{i, j})
        """
        if leaky:
            coefficients = _leaky_routing(logits)
        else:
            coefficients = core.softmax(logits, axis=capsule_axis)
        preactivate = coefficients * prediction_vectors
        preactivate = core.sum(preactivate, axis=-3, keepdims=True)
        if bias:
            preactivate += bias
        activation = act(preactivate)
        activations = activations.write(i, activation)
        # sum up along outcapdim dimension
        distance = core.sum(prediction_vectors * activation,
                            axis=core.axis,
                            keepdims=True)
        logits += distance
        return (i+1, logits, activations)

    _, logits, activations = tf.while_loop(
        lambda idx, logits, activations: idx < iterations,
        _update,
        loop_vars=[idx, logits, activations],
        swap_memory=True)
    #   [batch-size, nrows, ncols, 1, outcaps, outcapdim]
    # =>[batch-size, nrows, ncols, outcaps, outcapdim]
    return core.reshape(activations.read(iterations-1), shape)


def conv(convop, bias_shape, logits_shape, iterations,
         leaky=False,
         bias_initializer='zeros',
         bias_regularizer=None,
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
                                          'caps'+convop.__name__,
                                          reuse)
    act = actives.get(act)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('bias',
                         name,
                         bias_shape,
                         dtype,
                         bias_initializer,
                         bias_regularizer,
                         trainable,
                         collections,
                         reuse,
                         scope)
        if summarize and not reuse:
            tf.summary.histogram(bias.name, bias)
    else:
        bias = False
    def _conv(x):
        with ops_scope:
            #  [batch-size, nrows, ncols, incaps, incapdim] x [incapdim, outcaps, outcapdim]
            #=>[batch-size, nrows, ncols, incaps, outcaps, outcapdim]
            # equal to depthwise convolutional for conv2d
            x = convop(x)
            # x shape:
            # for fully-connected:
            #     [batch-size, incaps, outcaps, caps_dims]
            # for 1d:
            #     [batch-size, neurons, incaps, outcaps, caps_dims]
            # for 2d:
            #     [batch-size, nrows, ncols, incaps, outcaps, caps_dims]
            with tf.name_scope('agreement_routing'):
                x = _agreement_routing(x, logits_shape, iterations, bias, leaky)
            return x
    return _conv


def dot(input_shape, nouts, caps_dims,
        iterations=2,
        leaky=False,
        weight_initializer='glorot_uniform',
        weight_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
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
    weight_shape = [incaps, incapdim, nouts * caps_dims]
    bias_shape = [nouts, caps_dims]
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
    def _dot(x):
        # x shape:
        #    [batch-size, incaps, incapdims]
        # weight shape:
        #    [incaps, incapdim, nouts * caps_dims]

        # expand x to [batch-size, incaps, incapdims, 1]
        # and tile to [batch-size, incaps, incapdims, nouts * caps_dims]
        x = core.tile(core.expand_dims(x, -1), [1, 1, 1, nouts * caps_dims])
        # resulting [batch-size, incaps, incapdims, nouts * caps_dims]
        # then sum along with incapdims to get [batch-size, incaps, nouts * caps_dims]
        # then reshape to [batch-size, incaps, nouts, caps_dims]
        return core.reshape(core.sum(x * weight, 2), [-1, incaps, nouts, caps_dims])
    return conv(_dot,
                bias_shape,
                logits_shape,
                iterations,
                leaky,
                bias_initializer,
                bias_regularizer,
                act,
                trainable,
                dtype,
                collections,
                summarize,
                reuse,
                name,
                scope), output_shape


def conv2d(input_shape, nouts, caps_dims, kshape,
           iterations=3,
           leaky=False,
           stride=1,
           padding='valid',
           fastmode=False,
           weight_initializer='glorot_uniform',
           weight_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
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
    incaps, incapdim = input_shape[-2:]
    logits_shape = output_shape[:3] + [incaps, nouts, 1]
    bias_shape = [nouts, caps_dims]
    # kernel shape:
    # [krow, kcol, incaps, incapdims, outcaps * outcapdims]
    if fastmode:
        # if run in fast mode, apply depthwise_conv2d
        kernel_shape = kshape[1:-1] + [input_shape[-2] * input_shape[core.axis], 1]
        weight_shape = [incaps, incapdim, nouts * caps_dims]
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
    else:
        # else run in slow mode, apply capsulewise_conv2d
        # that is, for each `capsule version of feature map`
        # apply conv2d
        kernel_shape = kshape[1:-1] + input_shape[-2:] + [nouts * caps_dims]
    #print('kernel shape in conv2d:', kernel_shape)
    kernels = mm.malloc('kernel',
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
        tf.summary.histogram(kernels.name, kernels)

    def _body(idx, x, array):
        # kernels shape : [krow, kcol, incaps, incapdims, outcaps * outcapdims]
        # kernel shape  : [krow, kcol, incapdims, outcaps * outcapdims]
        kernel = tf.gather(kernels, idx, axis=-3)
        # x shape    : [batch-size, rows, cols, incaps, incapdim]
        # subx shape : [batch-size, rows, cols, incapdim]
        subx = tf.gather(x, idx, axis=-2)
        conv2d_output = core.conv2d(subx,
                                    kernel,
                                    stride,
                                    padding.upper())
        array = array.write(idx, conv2d_output)
        return [idx + 1, x, array]

    def _capsulewise_conv2d(x):
        """ capsule wise convolution in 2d
            that is, convolve along `incaps` dimension
        """
        # x shape:
        #     [batch-size, rows, cols, incaps, incapdim]
        iterations = input_shape[-2] # <- incaps
        idx = core.constant(0, core.int32)
        array = core.TensorArray(dtype=core.float32,
                                 size=iterations,
                                 clear_after_read=False)
        _, x, array = tf.while_loop(
            lambda idx, x, array : idx < iterations,
            _body,
            loop_vars = [idx, x, array],
            parallel_iterations=iterations,
        )
        # array should have the shape of:
        # incaps * [batch-size, nrows, ncols, outcaps * outcapdims]
        # stack to
        # [incaps, batch-size, nrows, ncols, outcaps * outcapdims]
        array = array.stack()
        #print('array after stack:', array)
        # then reshape to
        # [incaps, batch-size, nrows, ncols, outcaps, outcapdims]
        newshape = [iterations] + core.shape(array)[1:-1] + [nouts, caps_dims]
        #print('new shape:', newshape)
        array = core.reshape(array, newshape)
        # then transpose to
        # [batch-size, nrows, ncols, incaps, outcaps, caps_dims]
        array = tf.transpose(array, (1, 2, 3, 0, 4, 5))
        return array

    def _depthwise_conv2d(x):
        # x shape :
        #    [batch-size, rows, cols, incaps, incapdims]
        # to shape
        #    [batch-size, rows, cols, incaps * incapdims]
        x = core.reshape(x, input_nshape)
        # x shape :
        #    [batch-size, rows, cols, incaps * incapdims]
        x = core.depthwise_conv2d(x, kernels, stride, padding)
        xshape = input_shape[:]
        xshape[1:3] = core.shape(x)[1:3]
        # x shape :
        #    [batch-size, nrows, ncols, incaps, incapdims]
        x = core.reshape(x, xshape)
        # x to shape :
        #    [batch-size, nrows, ncols, incaps, incapdims, nouts * caps_dims]
        x = core.tile(core.expand_dims(x, -1),
                      [1, 1, 1, 1, 1, nouts * caps_dims])
        # [batch-size, nrows, ncols, incaps, incapdims, nouts * caps_dims]
        # then sum along with incapdims to get
        # [batch-size, nrows, ncols, incaps, nouts * caps_dims]
        # then reshape to
        # [batch-size, nrows, ncols, incaps, nouts, caps_dims]
        return core.reshape(core.sum(x * weight, 4),
                            xshape[:3] + [incaps, nouts, caps_dims])

    if fastmode:
        _conv2d = _depthwise_conv2d
    else:
        _conv2d = _capsulewise_conv2d

    return conv(_conv2d,
                bias_shape,
                logits_shape,
                iterations,
                leaky,
                bias_initializer,
                bias_regularizer,
                act,
                trainable,
                dtype,
                collections,
                summarize,
                reuse,
                name,
                scope), output_shape
