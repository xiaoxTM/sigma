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
           collections=None, # default is tf.GraphKeys.GLOBAL_VARIABLES
           reuse=False,
           scope=None):
    if name is None or layername is None:
        raise ValueError('`name` or `layername` not given.')
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    add_to_collect = True
    if scope is None:
        scope = layername
        add_to_collect = False
    else:
        scope = '{}/{}'.format(scope, layername)
    variable_type = 'trainable'
    if not trainable:
        variable_type = 'non-trainable'
    scope = '{}/variables/{}'.format(scope, variable_type)
    with tf.variable_scope(scope, reuse=reuse):
        variable = tf.get_variable(name, shape, dtype, initializer,
                                   regularizer, trainable, collections)
    if add_to_collect and not reuse:
        tf.add_to_collection(scope, variable)
    return variable


def local_variable(name,
                   layername,
                   shape,
                   dtype=None,
                   initializers=None,
                   regularizer=None,
                   trainable=True,
                   reuse=False,
                   scope=None):
    return malloc(name,
                  layername,
                  shape,
                  dtype,
                  initializer,
                  regularizer,
                  trainable,
                  tf.GraphKeys.LOCAL_VARIABLEs,
                  reuse,
                  scope)

def global_variable(name,
                   layername,
                   shape,
                   dtype=None,
                   initializers=None,
                   regularizer=None,
                   trainable=True,
                   reuse=False,
                   scope=None):
    return malloc(name,
                  layername,
                  shape,
                  dtype,
                  initializer,
                  regularizer,
                  trainable,
                  tf.GraphKeys.GLOBAL_VARIABLEs,
                  reuse,
                  scope)
