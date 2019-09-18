"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import inspect
import h5py
import io
import pickle
import gzip
import os.path
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials import mnist
from tensorflow.python import debug as tf_debug
from tensorflow.python.training import moving_averages

from . import commons
from sigma import colors
from sigma.ops import helper

# backend version
version = tf.__version__

def version_convert(version):
    v = version.split('.')
    if len(v) == 3:
        major, mid, minor = v
    elif len(v) == 2:
        major, mid = v
        minor = '0'
    else:
        raise ValueError('`version` must be major.mid[.minor], given {}'
                         .format(version))
    major = int(major)
    mid = int(mid)
    minor = int(minor)
    return major * 100000 + mid * 100 + minor

def version_compare_great(ver1, ver2):
    v1 = version_convert(ver1)
    v2 = version_convert(ver2)
    return v1 > v2

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

#----- tensorflow GraphKeys -----#
""" tf.GraphKeys
  - [ ] ACTIVATIONS
  - [ ] ASSET_FILEPATHS
  - [ ] BIASES
  - [ ] CONCATENATED_VARIABLES
  - [ ] COND_CONTEXT
  - [ ] EVAL_STEP
  - [ ] GLOBAL_STEP
  - [x] GLOBAL_VARIABLES
  - [ ] INIT_OP
  - [ ] LOCAL_INIT_OP
  - [ ] LOCAL_RESOURCES
  - [x] LOCAL_VARIABLES
  - [x] LOSSES
  - [ ] METRIC_VARIABLES
  - [ ] MODEL_VARIABLES
  - [ ] MOVING_AVERAGE_VARIABLES
  - [ ] QUEUE_RUNNERS
  - [ ] READY_FOR_LOCAL_INIT_OP
  - [ ] READY_OP
  - [ ] REGULARIZATION_LOSSES
  - [ ] RESOURCES
  - [ ] SAVEABLE_OBJECTS
  - [ ] SAVERS
  - [ ] SUMMARIES
  - [ ] SUMMARY_OP
  - [ ] TABLE_INITIALIZERS
  - [ ] TRAINABLE_RESOURCE_VARIABLES
  - [x] TRAINABLE_VARIABLES
  - [ ] TRAIN_OP
  - [X] UPDATE_OPS
  - [x] VARIABLES
  - [x] WEIGHTS
  - [ ] WHILE_CONTEXT
"""
class Collections():
    @staticmethod
    @property
    def global_variables():
        return tf.GraphKeys.GLOBAL_VARIABLES

    @staticmethod
    @property
    def local_variables():
        return tf.GraphKeys.LOCAL_VARIABLES

    @staticmethod
    @property
    def losses():
        return tf.GraphKeys.LOSSES

    @staticmethod
    @property
    def trainable_variables():
        return tf.GraphKeys.TRAINABLE_VARIABLES

    @staticmethod
    @property
    def variables():
        return tf.GraphKeys.VARIABLES

    @staticmethod
    @property
    def weights():
        return tf.GraphKeys.WEIGHTS

    @staticmethod
    @property
    def update_ops():
        return tf.GraphKeys.UPDATE_OPS


def is_tensor(x):
    return tf.contrib.framework.is_tensor(x)


def to_tensor(x,
              dtype=None,
              name=None,
              preferred_dtype=None):
    return tf.convert_to_tensor(x, dtype, name, preferred_dtype)


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

def runtime_print(*inputs, **kwargs):
    return tf.print(*inputs, **kwargs)

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


def while_loop(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               back_prop=True,
               swap_memory=False,
               name=None,
               **kwargs):
    """ tensorflow while loop
        NOTE that for tensorflow > 1.4, it has more than 8 parameters
        e.g., 1.5 => 9 (+ maximum_iterations)
        which will be restored in kwargs
    """
    return tf.while_loop(cond,
                         body,
                         loop_vars,
                         shape_invariants,
                         parallel_iterations,
                         back_prop,
                         swap_memory,
                         name,
                         **kwargs)


def control_dependencies(inputs):
    return tf.control_dependencies(inputs)


def identity(x, name=None):
    return tf.identity(x, name)


def map_func(func, elems,
             dtype=None,
             parallel_iterations=None,
             back_prop=True,
             swap_memory=False,
             infer_shape=True,
             name=None):
    if parallel_iterations is None:
        parallel_iterations = 1
    return tf.map_fn(func,
                     elems,
                     dtype,
                     parallel_iterations,
                     back_prop,
                     swap_memory,
                     infer_shape,
                     name)

def stop_gradient(x, name):
    return tf.stop_gradient(x, name)


#----- tensorflow scop -----#
def name_scope(name, default_name=None, values=None):
    return tf.name_scope(name, default_name, values)


def variable_scope(name_or_scope,
                   default_name=None,
                   values=None,
                   initializer=None,
                   regularizer=None,
                   caching_device=None,
                   partitioner=None,
                   custom_getter=None,
                   reuse=None,
                   dtype=None,
                   use_resource=None,
                   **kwargs):
    """ create variable scope
        NOTE that for tensorflow > 1.3, it has more than 11 parameters
        e.g., 1.4 => 12 (+ constraint)
              1.5 => 13 (+ constraint, auxiliary_name_scope)
        which will be restored in kwargs
    """
    return tf.variable_scope(name_or_scope,
                             default_name,
                             values,
                             initializer,
                             regularizer,
                             caching_device,
                             partitioner,
                             custom_getter,
                             reuse,
                             dtype,
                             use_resource,
                             **kwargs)


def variables_initializer(var_list, name='init'):
    return tf.variables_initializer(var_list, name)


def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 **kwargs):
    """ create variable
        NOTE that for tensorflow > 1.3, it has more than 12 parameters
        e.g., 1.4 => 12 (+ constraint)
        which will be restored in kwargs
    """
    return tf.get_variable(name,
                           shape,
                           dtype,
                           initializer,
                           regularizer,
                           trainable,
                           collections,
                           caching_device,
                           partitioner,
                           validate_shape,
                           use_resource,
                           custom_getter,
                           **kwargs)


#----- tensorflow collections -----#
def add_to_collection(name, variable):
    return tf.add_to_collection(name, variable)


def get_collection(name, scope=None):
    return tf.get_collection(name, scope)


def trainable_parameters():
    trainable_variables_shapes = [v.get_shape() for v in tf.trainable_variables()]
    return np.sum([np.prod(s) for s in trainable_variables_shapes])


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


def transpose(x, perm, conjugate=False, name=None):
    return tf.transpose(x,
                        perm=perm,
                        conjugate=conjugate,
                        name=name)


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
    return tf.range(start, limit=limit, delta=delta, dtype=dtype,
                    name=name)


def dtype(x):
    return x.dtype.base_dtype.name


def cast(x, dtype, name=None):
    return tf.cast(x, dtype, name)


def placeholder(dtype, shape=None, name=None):
    return tf.placeholder(dtype, shape, name)


def shape(x, aslist=True):
    x = x.get_shape()
    if aslist:
        return x.as_list()
    return x


def tshape(x, name=None, out_type=int32):
    return tf.shape(x, name, out_type)


def reshape(x, output_shape, smart=True, name=None):
    """ if in smart mode
        it will automatically change `None` to `-1`
        if applicable
    """
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


#def flatten(x, name=None):
#    xshape = shape(x)
#    nshape = [xshape[0], -1]
#    return tf.reshape(x, nshape, name)


def tile(x, multiples, name=None):
    return tf.tile(x, multiples, name)


def expand_dims(x, axis=None, name=None):
    return tf.expand_dims(x, axis, name)


def squeeze(x, axis=None, name=None):
    return tf.squeeze(x, axis, name)


def argmax(x,
           axis=None,
           dtype=int64,
           name=None):
    if version_compare_great(tf.__version__,'1.4.0'):
        return tf.argmax(x, axis=axis, output_type=dtype, name=name)
    return tf.argmax(x, axis=axis, name=name)


def argmin(x,
           axis=None,
           dtype=int64,
           name=None):
    if version_comare(tf.__version__ ,'1.4.0'):
        return tf.argmin(x, axis=axis, output_type=dtype, name=name)
    return tf.argmin(x, axis=axis, name=name)


def mean(x,
         axis=None,
         keepdims=None,
         name=None):
    return tf.reduce_mean(x, axis, keepdims, name)


def max(x,
        y=None,
        axis=None,
        keepdims=None,
        name=None):
    if y is None:
        return tf.reduce_max(x, axis, keepdims, name)
    return tf.maximum(x, y, name)


def min(x,
        y=None,
        axis=None,
        keepdims=None,
        name=None):
    if y is None:
        return tf.reduce_min(x, axis, keepdims, name)
    return tf.minimum(x, y, name)


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
         axis,
         keepdims=False,
         ord='euclidean',
         epsilon=commons.epsilon,
         safe=True,
         return_squared=False,
         name=None):
    if not safe:
        normed = tf.norm(x, ord, axis, keepdims, name)
        if return_squared:
            normed = [normed, normed * normed]
    else:
        squared = tf.reduce_sum(x * x,
                                axis=axis,
                                keepdims=keepdims)
        normed = tf.sqrt(squared + epsilon)
        if return_squared:
            normed = [normed, squared]
    return normed


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


def any(x, axis=None, keepdims=None, name=None):
    return tf.reduce_any(x, axis, keepdims, name)


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
    if version_compare_great(tf.__version__, '1.4.0'):
        return tf.nn.leaky_relu(x, alpha, name)
    return tf.maximum(x, x * alpha, name)


def softmax(x, axis=-1, name=None):
    if not version_compare_great(tf.__version__, '1.5'):
        axis = helper.normalize_axes(shape(x), axis)
    return tf.nn.softmax(x, axis, name)


def softplus(x, name=None):
    return tf.nn.softplus(x, name)


def softsign(x, name=None):
    return  tf.nn.softplus(x, name)


def sigmoid(x, log=False, name=None):
    if log:
        return tf.log_sigmoid(x, name)
    return tf.sigmoid(x, name)


def tanh(x, name=None):
    return tf.nn.tanh(x, name)


def seed(x):
    tf.set_random_seed(x)

def group(*inputs, **kwargs):
    return tf.group(*inputs, **kwargs)


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
           data_format=commons.data_format[1],
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
           data_format=commons.data_format[2],
           dilations=[1,1,1,1],
           name=None):
    if version_compare_great(tf.__version__,'1.5.0'):
        return tf.nn.conv2d(x,
                            filters,
                            strides,
                            padding,
                            use_cudnn_on_gpu,
                            data_format,
                            dilations,
                            name)
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
           data_format=commons.data_format[3],
           dilations=[1,1,1,1],
           name=None):
    if version_compare_great(tf.__version__,'1.5.0'):
        return tf.nn.conv3d(x,
                            filters,
                            strides,
                            padding,
                            use_cudnn_on_gpu,
                            data_format,
                            dilations,
                            name)
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
             data_format=commons.data_format[2],
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
              data_format=commons.data_format[2],
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
                     data_format=commons.data_format[2],
                     name=None):
    return tf.nn.depthwise_conv2d(x,
                                  kernels,
                                  strides,
                                  padding,
                                  rate,
                                  name,
                                  data_format)


def matmul(a, b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
    return tf.matmul(a, b,
                     transpose_a,
                     transpose_b,
                     adjoint_a,
                     adjoint_b,
                     a_is_sparse,
                     b_is_sparse,
                     name)


def tensordot(a, b, axes, name=None):
    return tf.tensordot(a, b, axes, name)


#----- tensorflow losses -----#
def softmax_cross_entropy_with_logits(labels=None,
                                      logits=None,
                                      axis=commons.caxis,
                                      name=None):

    if version_compare_great(tf.__version__,'1.5.0'):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                          logits=logits,
                                                          dim=axis,
                                                          name=name)
    return tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,
                                                   labels=labels,
                                                   logits=logits,
                                                   dim=axis,
                                                   name=name)


def sigmoid_cross_entropy_with_logits(labels=None,
                                      logits=None,
                                      name=None):
    return tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None,
                                                   labels=labels,
                                                   logits=logits,
                                                   name=name)


#----- tensorflow summaries -----#
def summarize(name, tensor, mode='histogram', norm=True, reuse=False, **kwargs):
    if not reuse:
        if norm:
            name = helper.normalize_name(name)
        if mode == 'histogram':
            return tf.summary.histogram(name, tensor)
        elif mode == 'scalar':
            return tf.summary.scalar(name, tensor)
        elif mode == 'image':
            return tf.summary.image(name, tensor, **kwargs)
        else:
            raise ValueError('`{}` mode for summary not supported'
                             .format(mode))


#----- tensorflow log gradients -----#
def optimize_with_summarize(optimizer, loss, exclusion=None):
    """ summary gradients for given loss
        and return optimization operation for training

        Attributes
        ==========
            optimizer : tf.train.Optimizer or sub-class
            loss : tensor
                   loss to be optimized
            exclusion : list / tuple / None
                        list of strings which is the gradient name to
                        be excluded for summarization
        Returns
        ==========
            optimization operation for optimize loss and train networks
    """
    if exclusion is None:
        exclusion = []
    grads_and_vars = optimizer.compute_gradients(loss)
    for (grad, var) in grads_and_vars:
        if grad is not None and grad.name not in exclusion:
            tf.summary.histogram('{}-grads'.format(grad.name), grad)
    return optimizer.apply_gradients(grads_and_vars)


#----- tensorflow metrics -----#
def metrics_accuracy(labels,
                     predictions,
                     weights=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None):
    return tf.metrics.accuracy(labels,
                               predictions,
                               weights,
                               metrics_collections,
                               updates_collections,
                               name)


def metrics_auc(labels,
               predictions,
               weights=None,
               num_thresholds=200,
               metrics_collections=None,
               updates_collections=None,
               curve='ROC',
               name=None):
    return tf.metrics.auc(labels,
                          predictions,
                          weights,
                          num_thresholds,
                          metrics_collections,
                          updates_collections,
                          curve,
                          name)


def metrics_false_negatives(labels,
                            predictions,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):
    return tf.metrics.false_negatives(labels,
                                      predictions,
                                      weights,
                                      metrics_collections,
                                      updates_collections,
                                      name)


def metrics_false_negatives_at_threshold(labels,
                                         predictions,
                                         thresholds,
                                         weights=None,
                                         metrics_collections=None,
                                         updates_collections=None,
                                         name=None):
    return tf.metrics.false_negatives_at_threshold(labels,
                                                   predictions,
                                                   thresholds,
                                                   weights,
                                                   metrics_collections,
                                                   updates_collections,
                                                   name)


def metrics_false_positives(labels,
                            predictions,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):
    return tf.metrics.false_positives(labels,
                                      predictions,
                                      weights,
                                      metrics_collections,
                                      updates_collections,
                                      name)


def metrics_false_positives_at_threshold(labels,
                                         predictions,
                                         thresholds,
                                         weights=None,
                                         metrics_collections=None,
                                         updates_collections=None,
                                         name=None):
    return tf.metrics.false_positives_at_threshold(labels,
                                                   predictions,
                                                   thresholds,
                                                   weights,
                                                   metrics_collections,
                                                   updates_collections,
                                                   name)


def metrics_true_negatives(labels,
                           predictions,
                           weights=None,
                           metrics_collections=None,
                           updates_collections=None,
                           name=None):
    return tf.metrics.true_negatives(labels,
                                     predictions,
                                     weights,
                                     metrics_collections,
                                     updates_collections,
                                     name)


def metrics_true_negatives_at_threshold(labels,
                                        predictions,
                                        thresholds,
                                        weights=None,
                                        metrics_collections=None,
                                        updates_collections=None,
                                        name=None):
    return tf.metrics.true_negatives_at_threshold(labels,
                                                  predictions,
                                                  thresholds,
                                                  weights,
                                                  metrics_collections,
                                                  updates_collections,
                                                  name)


def metrics_true_positives(labels,
                           predictions,
                           weights=None,
                           metrics_collections=None,
                           updates_collections=None,
                           name=None):
    return tf.metrics.true_positives(labels,
                                     predictions,
                                     weights,
                                     metrics_collections,
                                     updates_collections,
                                     name)


def metrics_true_positives_at_threshold(labels,
                                        predictions,
                                        thresholds,
                                        weights=None,
                                        metrics_collections=None,
                                        updates_collections=None,
                                        name=None):
    return tf.metrics.true_positives_at_threshold(labels,
                                                  predictions,
                                                  thresholds,
                                                  weights,
                                                  metrics_collections,
                                                  updates_collections,
                                                  name)


def metrics_mean_iou(labels,
                     predictions,
                     num_classes,
                     weights=None,
                     metrics_collections=None,
                     updates_collections=None,
                     name=None):
    return tf.metrics.mean_iou(labels,
                               predictions,
                               num_classes,
                               weights,
                               metrics_collections,
                               updates_collections,
                               name)


def metrics_precision(labels,
                      predictions,
                      weights=None,
                      metrics_collections=None,
                      updates_collections=None,
                      name=None):
    return tf.metrics.precision(labels,
                                precision,
                                weights,
                                metrics_collections,
                                updates_collections,
                                name)


def metrics_recall(labels,
                   predictions,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   name=None):
    return tf.metrics.recall(labels,
                             predictions,
                             weights,
                             metrics_collections,
                             updates_collections,
                             name)


#----- tensorflow optimizer -----#
def AdadeltaOptimizer(learning_rate=0.001,
                      rho=0.95,
                      epsilon=1e-08,
                      use_locking=False,
                      name='Adadelta'):
    return tf.train.AdadeltaOptimizer(learning_rate,
                                      rho,
                                      epsilon,
                                      use_locking,
                                      name)

def AdagradOptimizer(learning_rate,
                     initial_accumulator_value=0.1,
                     use_locking=False,
                     name='Adagrad'):
    return tf.train.AdagradOptimizer(learning_rate,
                                     initial_accumulator_value,
                                     use_locking,
                                     name)


def AdagradDAOptimizer(learning_rate,
                       global_step,
                       initial_gradient_squared_accumulator_value=0.1,
                       l1_regularization_strength=0.0,
                       l2_regularization_strength=0.0,
                       use_locking=False,
                       name='AdagradDA'):
    return tf.train.AdagradDAOptimizer(learning_rate,
                                       global_step,
                                       initial_gradient_squared_accumulator_value,
                                       l1_regularization_strength,
                                       l2_regularization_strength,
                                       use_locking,
                                       name)

def AdamOptimizer(learning_rate=0.001,
                  beta1=0.9,
                  beta2=0.999,
                  epsilon=1e-08,
                  use_locking=False,
                  name='Adam'):
    return tf.train.AdamOptimizer(learning_rate,
                                  beta1,
                                  beta2,
                                  epsilon,
                                  use_locking,
                                  name)

def GradientDescentOptimizer(learning_rate,
                             use_locking=False,
                             name='GradientDescent'):
    return tf.train.GradientDescentOptimizer(learning_rate,
                                             use_locking,
                                             name)

def ProximalAdagradOptimizer(learning_rate=0.1,
                             initial_accumulator_value=0.1,
                             l1_regularization_strength=0.0,
                             l2_regularization_strength=0.0,
                             use_locking=False,
                             name='ProximalAdagrad'):
    return tf.train.ProximalAdagradOptimizer(learning_rate,
                                             initial_accumulator_value,
                                             l1_regularization_strength,
                                             l2_regularization_strength,
                                             use_locking,
                                             name)

def ProximalGradientDescentOptimizer(learning_rate,
                                     l1_regularization_strength=0.0,
                                     l2_regularization_strength=0.0,
                                     use_locking=False,
                                     name='ProximalGradientDescent'):
    return tf.train.ProximalGradientDescentOptimizer(learning_rate,
                                                     l1_regularization_strength,
                                                     l2_regularization_strength,
                                                     use_locking,
                                                     name)

def RMSPropOptimizer(learning_rate,
                     decay=0.9,
                     momentum=0.0,
                     epsilon=1e-10,
                     use_locking=False,
                     centered=False,
                     name='RMSProp'):
    return tf.train.RMSPropOptimizer(learning_rate,
                                     decay,
                                     momentum,
                                     epsilon,
                                     use_locking,
                                     centered,
                                     name)

def MomentumOptimizer(learning_rate,
                      momentum,
                      use_locking=False,
                      name='Momentum',
                      use_nesterov=False):
    return tf.train.MomentumOptimizer(learning_rate,
                                      momentum,
                                      use_locking,
                                      name,
                                      use_nesterov)

def FtrlOptimizer(learning_rate,
                  learning_rate_power=-0.5,
                  initial_accumulator_value=0.1,
                  l1_regularization_strength=0.0,
                  l2_regularization_strength=0.0,
                  use_locking=False,
                  name='Ftrl',
                  accum_name=None,
                  linear_name=None,
                  l2_shrinkage_regularization_strength=0.0):
    return tf.train.FtrlOptimizer(learning_rate,
                                  learning_rate_power,
                                  initial_accumulator_value,
                                  l1_regularization_strength,
                                  l2_regularization_strength,
                                  use_locking,
                                  name,
                                  accum_name,
                                  linear_name,
                                  l2_shrinkage_regularization_strength)


def get_optimizer(optimizer, **kwargs):
    if optimizer not in ['GradientDescentOptimizer',
                         'AdadeltaOptimizer',
                         'AdagradOptimizer',
                         'AdagradDAOptimizer',
                         'MomentumOptimizer',
                         'AdamOptimizer',
                         'FtrlOptimizer',
                         'ProximalGradientDescentOptimizer',
                         'ProximalAdagradOptimizer',
                         'RMSPropOptimizer']:
        raise NotImplementedError('optimizer `{}` not implemented'
                                  .format(optimizer))
    return eval('{}(**kwargs)'.format(optimizer))


#------ tensorflow pooling -----#
@padnorm
def avg_pool(x, ksize, strides, padding,
             data_format=commons.data_format[2],
             name=None):
    return tf.nn.avg_pool(x,
                          ksize,
                          strides,
                          padding,
                          data_format,
                          name)


@padnorm
def max_pool(x, ksize, strides, padding,
            data_format=commons.data_format[2],
            name=None):
    return tf.nn.max_pool(x,
                          ksize,
                          strides,
                          padding,
                          data_format,
                          name)


#----- tensorflow resize -----#
def resize_nearest_neighbor(images,
                            size,
                            align_corners=False,
                            name=None):
    return tf.image.resize_nearest_neighbor(images,
                                            size,
                                            align_corners,
                                            name)


def resize_bilinear(images,
                    size,
                    align_corners=False,
                    name=None):
    return tf.image.resize_bilinear(images,
                                    size,
                                    align_corners,
                                    name)


def resize_bicubic(images,
                   size,
                   align_corners=False,
                   name=None):
    return tf.image.resize_bicubic(images,
                                   size,
                                   align_corners,
                                   name)


def resize_area(images,
                size,
                align_corners=False,
                name=None):
    return tf.image.resize_area(images,
                                size,
                                align_corners,
                                name)


#------ tensorflow lx_loss -----#
def l2_loss(tensor, name=None):
    return tf.nn.l2_loss(tensor, name)


#----- tensorflow dropout
def dropout(x,
            keep_prob,
            noise_shape=None,
            seed=None,
            name=None):
    return tf.nn.dropout(x,
                         keep_prob,
                         noise_shape,
                         seed,
                         name)


#----- tensorflow moments -----#
def moments(x,
            axes,
            shift=None,
            keepdims=False,
            name=None):
    return tf.nn.moments(x,
                         axes,
                         shift,
                         name,
                         keepdims)


#----- tensorflow batch norm -----#
def fused_batch_norm(x,
                     scale,
                     offset,
                     mean=None,
                     variance=None,
                     epsilon=0.001,
                     data_format=commons.data_format[2],
                     is_training=True,
                     name=None):
    return tf.nn.fused_batch_norm(x,
                                  scale,
                                  offset,
                                  mean,
                                  variance,
                                  epsilon,
                                  data_format,
                                  is_training,
                                  name)


def batch_norm(x,
               mean,
               variance,
               offset,
               scale,
               variance_epsilon=commons.epsilon,
               name=None):
    return tf.nn.batch_normalization(x,
                                     mean,
                                     variance,
                                     offset,
                                     scale,
                                     variance_epsilon,
                                     name)


#----- tensorflow save model / weights -----#
def load_mnist(path, one_hot):
    return mnist.input_data.read_data_set(path, one_hot)


def load(session,
         checkpoint,
         saver=None,
         var_list=None,
         verbose=True):
    if saver is None:
        saver = tf.train.Saver(var_list)
    if not isinstance(saver, tf.train.Saver):
        raise TypeError('`{}saver{}` must be instance of {}tf.train.Saver{}. '
                        'given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.red(type(saver))))
    sessions = [tf.Session, tf_debug.LocalCLIDebugWrapperSession]
    if version_compare_great(tf.__version__, '1.12.0'):
        sessions.append(tf_debug.TensorBoardDebugWrapperSession)
    if not isinstance(session, tuple(sessions)):
        raise TypeError('`{}session{}` must be instance of {}tf.Session{}. '
                        'given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.fg.red(type(session))))
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint))
    if ckpt and ckpt.model_checkpoint_path:
        if verbose:
            print('{}load check point from {}{}{}'
                   .format(colors.fg.cyan, colors.fg.red,
                           ckpt.model_checkpoint_path, colors.reset)
                 )
        saver.restore(session, ckpt.model_checkpoint_path)
    elif verbose:
        if ckpt is None:
            print('{}no checkpoint to restore from{}'
                  .format(colors.fg.green, colors.reset))
        else:
            print('{}restoring from checkpoint: {} {}ignored{}'
                  .format(colors.fg.blue,
                          colors.green(checkpoint),
                          colors.fg.red,
                          colors.reset))
    return session, saver


def save(session,
         checkpoint,
         saver=None,
         verbose=True,
         **kwargs):
    if saver is None:
        saver = tf.train.Saver()
    if not isinstance(saver, tf.train.Saver):
        raise TypeError('`{}saver{}` must be instance of {}tf.train.Saver{}. '
                        'given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.red(type(saver))))
    sessions = [tf.Session, tf_debug.LocalCLIDebugWrapperSession]
    if version_compare_great(tf.__version__, '1.12.0'):
        sessions.append(tf_debug.TensorBoardDebugWrapperSession)
    if not isinstance(session, tuple(sessions)):
        raise TypeError('`{}session{}` must be instance of {}tf.Session{}. '
                        'given {}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.red(type(session))))
    if verbose:
        print('{}saving check point to {}{}{}'
               .format(colors.fg.cyan, colors.fg.red,
                       checkpoint, colors.reset))
    saver.save(session, checkpoint, **kwargs)
    return session, saver


def import_weights(filename, session,
                   graph=None,
                   collections=[Collections.global_variables],
                   verbose=True):
    if graph is None:
        graph = session.graph
    if collections is None:
        collections = graph.get_all_collection_keys()
    elif isinstance(collections, str):
        collections = [collections]

    if not isinstance(collections, (list, tuple, np.ndarray)):
        raise TypeError('collections must be list/tuple. given {}'
                        .format(colors.red(type(collections))))
    with h5py.File(filename, 'r') as f:
        if verbose:
            print('importing weights from {}'
                  .format(colors.green(filename)))
        imported_weights = {}
        for collection in collections:
            weight_group = f[collection]
            weight_names = commons.decode(weight_group.attrs['weight_names'])
            params = graph.get_collection(collection)
            for param in params:
                if not imported_weights.get(param.name, False):
                    if param.name in weight_names:
                        value = np.asarray(weight_group[param.name])
                        op = tf.assign(param, value)
                        session.run(op)
                    elif verbose:
                        print('parameter {} not found.'
                              .format(colors.red(param.name)))
                    imported_weights[param.name] = True
    return graph, session


def export_weights(filename, session,
                   graph=None,
                   collections=[Collections.global_variables],
                   verbose=True):
    with h5py.File(filename, mode='w') as f:
        if verbose:
            print('exporting weights to {}{}{}'
                  .format(colors.green(filename)))
        if graph is None:
            graph = session.graph
        f.attrs['sigma_version'] = sigma.__version__.encode('utf-8')
        f.attrs['data_format'] = ' '.join(map(str,commons.data_format)).encode('utf-8')
        if collections is None:
            collections = graph.get_all_collection_keys()
        elif isinstance(collections, str):
            collections = [collections]
        if not isinstance(collections, (list, tuple, np.ndarray)):
            raise TypeError('`collections` must be list/tuple or np.ndarray')
        exported_weights = {}
        for collection in collections:
            weight_group = f.create_group(collection)
            params = graph.get_collection(collection)
            names = []
            for param in params:
                if not exported_weights.get(param.name, False):
                    val = session.run(param)
                    pset = weight_group.create_dataset(str(param.name),
                                                       val.shape,
                                                       dtype=val.dtype)
                    names.append(param.name)
                    if not val.shape:
                        pset[()] = val
                    else:
                        pset[:] = val
                    exported_weights[param.name] = True
            if len(names) != 0:
                weight_group.attrs['weight_names'] = commons.encode(names)


def import_model(filename, session,
                 verbose=True,
                 **kwargs):
    if verbose:
        print('importing model from {}'
              .format(colors.green(filename)))
    pkl = gzip.open(filename, 'rb')
    meta, data = pickle.load(pkl, **kwargs)
    pkl.close()
    with io.StringIO(meta) as sio:
        saver = tf.train.import_meta_graph(sio, **kwargs)
    with io.StringIO(data) as sio:
        saver.restore(session, sio)


def export_model(filename, session,
                 verbose=True,
                 **kwargs):
    if verbose:
        print('exporting model to {}'
              .format(colors.green(filename)))
    if saver is None:
        saver = tf.train.Saver()
    pkl = gzip.open(filename, 'wb')
    meta = tf.train.export_meta_graph(**kwargs)
    with io.StringIO() as sio:
        saver.save(session, sio)
        data = sio.getvalue()
        pkl.dummy([meta, data])
    pkl.close()


#----- tensorflow session run -----#
def run(session, operations,
        feed_dict=None,
        options=None,
        run_metadata=None):
    return session.run(operations,
                       feed_dict,
                       options,
                       run_metadata)


#----- tensorflow engine -----#
def session(target='',
            graph=None,
            config=None,
            initializers=None,
            debug=False,
            address=None):
    sess = tf.Session(target, graph, config)
    if debug:
        if address is not None:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess,
                                                           address,
                                                           send_traceback_and_source_code=False)
        else:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if initializers is not None:
        sess.run(initializers)
    return sess


def close_session(session):
    if session is not None:
        session.close()


def close_summary_writer(writer):
    if writer is not None:
        return writer.close()


def summary_merge():
    return tf.summary.merge_all()


def summary_writer(logdir,
                   graph=None,
                   max_queue=10,
                   flush_secs=120,
                   graph_def=None,
                   filename_suffix=None):
    return tf.summary.FileWriter(logdir,
                                 graph,
                                 max_queue,
                                 flush_secs,
                                 graph_def,
                                 filename_suffix)


def add_summary(filewriter, summary, global_step=None):
    return filewriter.add_summary(summary, global_step)

def device(name_or_function):
    return tf.device(name_or_function)


def idle():
    tf.no_op()

def extract_patches(x,
                    ksize,
                    strides,
                    rates,
                    padding,
                    name=None):
    return tf.extract_image_patches(x,
                              ksize,
                              strides,
                              rates,
                              padding,
                              name)


def moving_average_update(x,
                          value,
                          decay,
                          zero_debias=True,
                          name=None):
    return moving_averages.assign_moving_average(x,
                                                 value,
                                                 decay,
                                                 zero_debias,
                                                 name)

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def make_feature(features):
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

def feature_writer(filename):
    return tf.python_io.TFRecordWriter(filename)

def feature_reader(filenames):
    reader = tf.TFRecordReader()
    return reader
