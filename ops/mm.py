import tensorflow as tf
from . import initializers
from . import regularizers

def malloc(name, shape, dtype=None, initializer=None,
           regularizer=None, trainable=True,
           collections=None, reuse=False, scope=None):
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    if scope is None:
        scope = name
    with tf.variable_scope(scope, reuse=reuse):
        variable = tf.get_variable(name, shape, dtype, initializer,
                                   regularizer, trainable, collections)
    if not reuse:
        tf.add_to_collection(scope, variable)
    return variable
