import tensorflow as tf

def placeholder(dtype, shape=None, name=None):
    return tf.placeholder(dtype, shape, name)


def shape(x):
    return x.get_shape().as_list()


def tshape(x, name=None, out_type=tf.int32):
    return tf.shape(x, name, out_type)


def reshape(x, output_shape,
            name=None):
    return tf.reshape(x, output_shape, name)


def argmax(x,
           axis=None,
           dtype='int64',
           name=None):
    return tf.argmax(x, axis, dtype, name)


def argmin(x,
           axis=None,
           dtype='int64',
           name=None):
    return tf.argmin(x, axis, dtype, name)


def mean(x,
         axis=None,
         keepdims=None,
         name=None):
    return tf.reduce_mean(x, axis, keepdims, name)


def max(x,
        axis=None,
        keepdims=None,
        name=None):
    return tf.reduce_max(x, axis, keepdims, name)


def min(x,
        axis=None,
        keepdims=None,
        name=None):
    return tf.reduce_min(x, axis, keepdims, name)


def minimum(x, y, name=None):
    return tf.minimum(x, y, name)


def maximum(x, y, name=None):
    return tf.maximum(x, y, name)


def prod(x,
         axis=None,
         keepdims=None,
         name=None):
    return tf.reduce_prod(x, axis, keepdims, name)


def multiply(x, y, name=None):
    return tf.multiply(x, y, name)


def exp(x, name=None):
    return tf.exp(x, name)


def pow(x, y, name=None):
    return tf.pow(x, y, name)


def log(x, name=None):
    return tf.log(x, name)

def sum(x,
        axis=None,
        keepdims=None,
        name=None):
    return tf.reduce_sum(x, axis, keepdims, name)


def concat(inputs, axis, name='concat'):
    return tf.concat(inputs, axis, name)


def add(inputs, name=None):
    return tf.add_n(inputs, name)


def square(x, name=None):
    return tf.square(x, name)


def sqrt(x, name=None):
    return tf.sqrt(x, name)


""" return reciprocal of square root of input element-wise
    that is :
        y = 1 / root(inputs)
"""
def rsqrt(x, name=None):
    return tf.rsqrt(x, name)


def abs(x, name=None):
    return tf.abs(x, name)


def neq(x, y, name=None):
    tf.not_equal(x, y, name)


def eq(x, y, name=None):
    return tf.equal(x, y, name)


def greater(x, y, name=None):
    return tf.greater(x, y, name)


def geq(x, y, name=None):
    return tf.greater_equal(x, y, name)


def less(x, y, name=None):
    return tf.less(x, y, name)


def leq(x, y, name=None):
    return tf.less_equal(x, y, name)


def clip(x, minv, maxv, name=None):
    return tf.clip_by_value(x, minv, maxv, name)


#===========================================================
"""
"""

def zeros(shape, dtype=tf.float32, name=None):
    return tf.zeros(shape, dtype, name)


def zeros_like(x, dtype=None, name=None, optimize=True):
    return tf.zeros_like(x, dtype, name, optimize)


def ones(shape, dtype=tf.float32, name=None):
    return tf.ones(shape, dtype, name)


def ones_like(x, dtype=tf.float32, name=None, optimize=True):
    return tf.ones_like(x, dtype, name, optimize)


def one_hot(indices, depth,
            pos_value=None,
            neg_value=None,
            axis=None,
            dtype=None,
            name=None):
    return tf.one_hot(indices, depth, pos_value, neg_value, axis, dtype, name)


#===========================================================
""" activations
"""

def crelu(x, name=None):
    return tf.nn.crelu(x, name)


def relu(x, name=None):
    return tf.nn.relu(x, name)


def relu6(x, name=None):
    return tf.nn.relu(x, name)


def elu(x, name=None):
    return tf.nn.elu(x, name)


def selu(x,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946,
         name=None):
    return tf.nn.selu(x, alpha, scale, name)


def leaky_relu(x, alpha=0.2, name=None):
    if tf.__version__ >= '1.4.0':
        return tf.nn.leaky_relu(x, alpha, name)
    return tf.nn.maximum(x, x * alpha, name)


def softmax(x, axis=-1, name=None):
    return tf.nn.softmax(x, axis, name)


def softplus(x, name=None):
    return tf.nn.softplus(x, name)


def softsign(x, name=None):
    return  tf.nn.softplus(x, name)


def sigmoid(x, name=None):
    return tf.sigmoid(x, name)


def tanh(x, name=None):
    return tf.nn.tanh(x, name)
