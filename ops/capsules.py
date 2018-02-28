
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


def _update_routing(prediction_vectors, biases, iterations,
                    leaky=False):
    """ calculate v_j by dynamic routing
        Attributes
        ==========
        prediction_vectors : tf.Tensor
                             predictions from previous layers
                             denotes u_{j|i}_hat in paper
        biases : tf.Tensor
                 b_{i, j} to adjust coupling coefficients (c_{i, j})
                 donates b_{i, j} in paper
        iterations : int
                     iteration times to adjust c_{i, j}.
                     donates r in paper
        leaky : boolean
                whether leaky_routing or not

        Returns
        ==========
        Activated tensor of the output layer. That is v_j
    """
    pass


def fully_connected(input_shape, nouts, caps_dim,
                    weight_initializer='glorot_uniform',
                    weight_regularizer=None,
                    bias_initializer=False, # no bias
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
                      [batch-size, num_of_capsules, capsule_dimensions]
        nouts : int
                output number of capsules
        caps_dims : int
                    output capsule dimension
    """
    ops_scope, name = helper.assign_scope(name,
                                          scope,
                                          'caps_fully_connected',
                                          reuse)
    wshape = input_shape[1:] + [nouts * caps_dim]
    weight = mm.malloc('weight',
                        wshape,
                        dtype,
                        weight_initializer,
                        weight_regularizer,
                        trainable,
                        collections,
                        reuse,
                        name,
                        scope)
    if summarize and not reuse:
        tf.summary.histogram(weight.name, weight)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = mm.malloc('bias',
                         (nouts, caps_dims),
                         dtype,
                         bias_initializer,
                         bias_regularizer,
                         trainable,
                         collections,
                         reuse,
                         name,
                         scope)
        if summarize and not reuse:
            tf.summary.histogram(bias.name, bias)
    else:
        bias = 0
    def _fully_connected(x):
        with ops_scope:
            tiled = tf.tile(tf.expand_dims(x, -1),
                            [1] * len(input_shape) + [nouts * caps_dims])
            prediction_vectors = core.sum(tiled * weight, axis=core.axis)

    return _fully_connected


def conv2d(input_shape, nouts, caps_dims, kshape,
           stride=1,
           padding='valid',
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
    pass
