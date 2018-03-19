# Copyright 2017 Renwu GAO All Rights Reserved.
#
# ==============================================================================
import inspect
import tensorflow as tf
from . import commons

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

Optimizer = tf.train.Optimizer

def padnorm(fun):
    def _padnorm(*args, **kwargs):
        signature = inspect.signature(fun)
        items = list(signature.parameters.items())
        # merge args into kwargs
        for idx, arg in enumerate(args):
            kwargs[items[idx][0]] = arg
        padding = kwargs.get('padding', None)
        if padding is not None:
            # ignore padding parameter if not exists
            kwargs['padding'] = padding.upper()
        return fun(**kwargs)
    return _padnorm


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


def gather_nd(x, indices, name=None):
    return tf.gather_nd(x, indices, name)


def meshgrid(*args, **kwargs):
    return tf.meshgrid(*args, **kwargs)


def stack(values, axis=0, name='stack'):
    return tf.stack(values, axis, name)


def unstack(x, num=None, axis=0, name='unstack'):
    return tf.unstack(x, num, axis, name)


def concat(inputs, axis, name='concat'):
    return tf.concat(inputs, axis, name)


def split(x, num_or_size_splits,
          axis=0,
          num=None,
          name='split'):
    return tf.split(x, num_or_size_splits, axis, num, name)


def range(start, limit=None, delta=1, dtype=None, name='range'):
    if limit is None:
        # treat `start` as `limit`
        return tf.range(start, delta=delta, dtype=dtype, name=name)
    else:
        return tf.range(start, limit=limit, delta=delta, dtype=dtype,
                        name=name)


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


def reshape(x, output_shape, smart=True, name=None):
    if smart:
        stats = commons.shape_statistics(output_shape)
        nones = stats['nones']
        if len(nones) == 1:
            if len(stats['-1']) == 0:
                output_shape[stats['nones'][0]] = -1
            else:
                raise ValueError('`output_shape` can not contains '
                                 'both `None` and `-1`. {}'
                                 .format(output_shape))
        elif len(nones) > 1:
            raise ValueError('`output_shape` can not contains multiple '
                             '`None`. {}'.format(output_shape))
        else:
            if len(stats['-1']) > 1:
                raise ValueError('`output_shape` can not contains multiple '
                               '`-1`. {}'.format(output_shape))
    return tf.reshape(x, output_shape, name)


def tile(x, multiples, name=None):
    return tf.tile(x, multiples, name)


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


def random_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  dtype=tf.float32,
                  seed=None,
                  name=None):
    return tf.random_normal(shape,
                            mean,
                            stddev,
                            dtype,
                            seed,
                            name)
def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=tf.float32,
                   seed=None,
                   name=None):
    return tf.random_uniform(shape,
                             minval,
                             maxval,
                             dtype,
                             seed,
                             name)
def truncated_normal(shape,
                     mean=0.0,
                     stddev=1.0,
                     dtype=tf.float32,
                     seed=None,
                     name=None):
    return tf.truncated_normal(shape,
                               mean,
                               stddev,
                               dtype,
                               seed,
                               name)


# convolutional operations
def embedding(params,
              ids,
              partition_strategy='mod',
              validate_indices=True,
              max_norm=None,
              name=None):
    return tf.nn.embedding_lookup(params,
                                  ids,
                                  partition_strategy,
                                  name,
                                  validate_indices,
                                  max_norm)
@padnorm
def conv1d(x, filters, stride, padding,
           use_cudnn_on_gpu=None,
           data_format=commons.data_format,
           name=None):
    return tf.nn.conv1d(x,
                        filters,
                        stride,
                        padding,
                        use_cudnn_on_gpu,
                        data_format,
                        name)
@padnorm
def conv2d(x, filters, strides, padding,
           use_cudnn_on_gpu=None,
           data_format=commons.data_format,
           dilations=[1,1,1,1],
           name=None):
    if tf.__version__ >= '1.5':
        return tf.nn.conv2d(x,
                            filters,
                            strides,
                            padding,
                            use_cudnn_on_gpu,
                            data_format,
                            dilations,
                            name)
    else:
        return tf.nn.conv2d(x,
                            filters,
                            strides,
                            padding,
                            use_cudnn_on_gpu,
                            data_format,
                            name)
@padnorm
def conv3d(x, filters, strides, padding,
           use_cudnn_on_gpu=None,
           data_format=commons.data_format,
           dilations=[1,1,1,1],
           name=None):
    if tf.__version__ >= '1.5':
        return tf.nn.conv3d(x,
                            filters,
                            strides,
                            padding,
                            use_cudnn_on_gpu,
                            data_format,
                            dilations,
                            name)
    else:
        return tf.nn.conv3d(x,
                    filters,
                    strides,
                    padding,
                    use_cudnn_on_gpu,
                    data_format,
                    name)
@padnorm
def deconv2d(x,
             filters,
             output_shape,
             strides,
             padding='SAME',
             data_format=commons.data_format,
             name=None):
    return tf.nn.conv2d_transpose(x,
                                  filters,
                                  output_shape,
                                  strides,
                                  padding,
                                  data_format,
                                  name)
@padnorm
def sepconv2d(x, depthwise_filter, pointwise_filter, strides, padding,
              rate=None,
              data_format=commons.data_format,
              name=None):
    return tf.nn.separable_conv2d(x,
                                  depthwise_filter,
                                  pointwise_filter,
                                  strides,
                                  padding,
                                  rate,
                                  name,
                                  data_format)
@padnorm
def depthwise_conv2d(x, kernels, strides, padding,
                     rate=None,
                     data_format=commons.data_format,
                     name=None):
    return tf.nn.depthwise_conv2d(x,
                                  kernels,
                                  strides,
                                  padding,
                                  rate,
                                  name,
                                  data_format)


def dot(*args, **kwargs):
    return tf.matmul(*args, **kwargs)


def tensordot(a, b, axes, name=None):
    return tf.tensordot(a, b, axes, name)


def expand_dims(x, axis=None, name=None):
    return tf.expand_dims(x, axis, name)
