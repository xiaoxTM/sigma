from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from . import initializers
from . import regularizers

def malloc(name, shape,
           dtype=None,
           initializer=None,
           regularizer=None,
           trainable=True,
           collections=None,
           reuse=False,
           layer=None,
           scope=None):
    if name is None:
        raise ValueError('`name` not given.')
    if scope is None and layer is None:
        raise ValueError('`layer` and `scope` can not be both None')
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    _scope = scope
    if _scope is None:
        _scope = layer
    elif layer is not None:
        _scope = '{}/{}'.format(_scope, layer)
    # print('malloc scope, name, reuse:', _scope, name, reuse)
    with tf.variable_scope(_scope, reuse=reuse):
        variable = tf.get_variable(name, shape, dtype, initializer,
                                   regularizer, trainable, collections)
    if scope is not None and not reuse:
        tf.add_to_collection(scope, variable)
    return variable
