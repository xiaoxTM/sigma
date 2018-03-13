# Copyright 2017 Renwu GAO All Rights Reserved.
#
# ==============================================================================

import tensorflow as tf

epsilon = 1e-5
data_format = 'NHWC'
axis = -1

# floating data type
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64
bfloat16 = tf.bfloat16

# integer data type
int8 = tf.int8
uint8 = tf.uint8
int16 = tf.int16
uint16 = tf.uint16
int32 = tf.int32
#uint32 = tf.uint32
int64 = tf.int64
#uint64 = tf.uint64

boolean = tf.bool
string = tf.string

# quantized integer
qint8 = tf.qint8
quint8 = tf.quint8
qint16 = tf.qint16
quint16 = tf.quint16
qint32 = tf.qint32

TensorArray = tf.TensorArray

def wrap(fun, *args, **kwargs):
    return eval('tf.{}(*args, **kwargs)'.format(fun))


#= workflow control =====================
def cond(condition,
         true_fun,
         false_fun,
         strict=False,
         name=None):
    return tf.cond(condition,
                   true_fun,
                   false_fun,
                   strict,
                   name)


def assign(x, y,
           validate_shape=None,
           use_locking=None,
           name=None):
    return tf.assign(x, y, validate_shape, use_locking, name)


def assign_add(x, y, use_locking=None, name=None):
    return tf.assign_add(x, y, use_locking, name)



def rank(x):
    return tf.rank(x)


def gather(x, indices, validate=None, name=None, axis=0):
    return tf.gather(x, indices, validate, name, axis)


def dtype(x):
    return x.dtype.base_dtype.name


def cast(x, dtype, name=None):
    return tf.cast(x, dtype, name)


def placeholder(dtype, shape=None, name=None):
    return tf.placeholder(dtype, shape, name)


def shape(x):
    return x.get_shape().as_list()


def tshape(x, name=None, out_type=int32):
    return tf.shape(x, name, out_type)


def reshape(x, output_shape, name=None):
    return tf.reshape(x, output_shape, name)


def argmax(x,
           axis=None,
           dtype=int64,
           name=None):
    return tf.argmax(x, axis=axis, output_type=dtype, name=name)


def argmin(x,
           axis=None,
           dtype=int64,
           name=None):
    return tf.argmin(x, axis=axis, output_type=dtype, name=name)


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


def split(x, num_or_size_splits,
          axis=0,
          num=None,
          name='split'):
    return tf.split(value, num_or_size_splits, axis, num, name)


def add(inputs, name=None):
    return tf.add_n(inputs, name)


def bias_add(x, bias, data_format=None, name=None):
    return tf.nn.bias_add(x, bias, data_format, name)


def square(x, name=None):
    return tf.square(x, name)


def sqrt(x, name=None):
    return tf.sqrt(x, name)


def norm(x,
         axis=None,
         keepdims=None,
         ord='euclidean',
         name=None):
    return tf.norm(x, ord, axis, keepdims, name)


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

def all(x, axis=None, keepdims=None, name=None):
    return tf.reduce_all(x, axis, keepdims, name)


def clip(x, minv, maxv, name=None):
    return tf.clip_by_value(x, minv, maxv, name)


#===========================================================
"""
"""

def zeros(shape, dtype=float32, name=None):
    return tf.zeros(shape, dtype, name)


def zeros_like(x, dtype=None, name=None, optimize=True):
    return tf.zeros_like(x, dtype, name, optimize)


def ones(shape, dtype=float32, name=None):
    return tf.ones(shape, dtype, name)


def ones_like(x, dtype=float32, name=None, optimize=True):
    return tf.ones_like(x, dtype, name, optimize)


def one_hot(indices, depth,
            pos_value=None,
            neg_value=None,
            axis=None,
            dtype=None,
            name=None):
    return tf.one_hot(indices, depth, pos_value, neg_value, axis, dtype, name)


def constant(value,
             dtype=None,
             shape=None,
             name='const',
             verify_shape=False):
    return tf.constant(value, dtype, shape, name, verify_shape)


def fill(shape, values, name=None):
    return tf.fill(shape, values, name)


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


def random_normal(*args, **kwargs):
    return tf.random_normal(*args, **kwargs)


def random_uniform(*args, **kwargs):
    return tf.random_uniform(*args, **kwargs)


def truncated_normal(*args, **kwargs):
    return tf.truncated_normal(*args, **kwargs)


# convolutional operations
def embedding(*args, **kwargs):
    return tf.nn.embedding_lookup(*args, **kwargs)


def conv1d(*args, **kwargs):
    return tf.nn.conv1d(*args, **kwargs)


def conv2d(*args, **kwargs):
    return tf.nn.conv2d(*args, **kwargs)


def conv3d(*args, **kwargs):
    return tf.nn.conv3d(*args, **kwargs)


def deconv2d(*args, **kwargs):
    return tf.nn.conv2d_transpose(*args, **kwargs)


def sepconv2d(*args, **kwargs):
    return tf.nn.separable_conv2d(*args, **kwargs)


def depthwise_conv2d(x, kernels, strides, padding,
                     rate=None,
                     name=None):
    return tf.nn.depthwise_conv2d(x, kernels, strides,
                               padding, rate,
                               name, data_format)


def dot(*args, **kwargs):
    return tf.matmul(*args, **kwargs)


def tensordot(a, b, axes, name=None):
    return tf.tensordot(a, b, axes, name)


def expand_dims(x, axis=None, name=None):
    return tf.expand_dims(x, axis, name)
