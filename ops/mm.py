import tensorflow as tf
from . import initializers
from . import regularizers

def malloc(name,
           layername,
           shape,
           dtype=None,
           initializer=None,
           regularizer=None,
           trainable=True,
           collections=None,
           reuse=False,
           scope=None):
    if name is None or layername is None:
        raise ValueError('`name` or `layername` not given.')
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    if scope is None:
        scope = layername
    else:
        scope = '{}/{}'.format(scope, layername)
    variable_type = 'trainable'
    if not trainable:
        variable_type = 'non-trainable'
    scope = '{}/variables/{}'.format(scope, variable_type)
    with tf.variable_scope(scope, reuse=reuse):
        variable = tf.get_variable(name, shape, dtype, initializer,
                                   regularizer, trainable, collections)
    if scope is not None and not reuse:
        tf.add_to_collection(scope, variable)
    return variable
