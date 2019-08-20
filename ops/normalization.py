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

from . import mm, helper, actives, core
from .. import status, helpers

def instance_norm(input_shape,
                  offset_initializer='zeros',
                  scale_initializer='ones',
                  offset_regularizer=None,
                  scale_regularizer=None,
                  cpuid=0,
                  epsilon=core.epsilon,
                  act=None,
                  trainable=True,
                  collections=None,
                  summary='histogram',
                  check_input_shape=True,
                  reuse=False,
                  name=None,
                  scope=None):
    helper.check_input_shape(input_shape)
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'instance_norm',
                                             reuse)
    input_len = len(input_shape)
    axis = helper.normalize_axes(input_shape)
    neurons = input_shape[axis]
    axes = list(range(input_len))
    del axes[axis]
    del axes[0]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('offset',
                           name,
                           neurons,
                           core.float32,
                           offset_initializer,
                           offset_regularizer,
                           cpuid,
                           trainable,
                           collections,
                           summary,
                           reuse,
                           scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('scale',
                          name,
                          neurons,
                          core.float32,
                          scale_initializer,
                          scale_regularizer,
                          cpuid,
                          trainable,
                          collections,
                          summary,
                          reuse,
                          scope)
    act = actives.get(act)
    def _instance_norm(x):
        with ops_scope:
            mean, variance = core.moments(x, axes, keepdims=True)
            normalized = (x - mean) / core.sqrt(variance + epsilon)
            if scale is not None:
                normalized = scale * normalized
            if offset is not None:
                normalized = normalized + offset
            return  act(normalized)
    return _instance_norm


""" code inspired by (borrowed from):
        https://github.com/tensorflow/magenta/blob/master
              /magenta/models/image_stylization/ops.py
"""
def conditional_instance_norm(input_shape,
                              bank_size,
                              offset_initializer='zeros',
                              scale_initializer='ones',
                              offset_regularizer=None,
                              scale_regularizer=None,
                              cpuid=0,
                              epsilon=core.epsilon,
                              act=None,
                              trainable=True,
                              collections=None,
                              summary='histogram',
                              check_input_shape=True,
                              reuse=False,
                              name=None,
                              scope=None):
    helper.check_input_shape(input_shape)
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'conditional_instance_norm',
                                             reuse)
    input_len = len(input_shape)
    axis = helper.normalize_axes(input_shape)
    neurons = [bank_size, input_shape[axis]]
    axes = list(range(input_len))
    del axes[axis]
    del axes[0]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('offset',
                           name,
                           neurons,
                           core.float32,
                           offset_initializer,
                           offset_regularizer,
                           cpuid,
                           trainable,
                           collections,
                           summary,
                           reuse,
                           scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('scale',
                          name,
                          neurons,
                          core.float32,
                          scale_initializer,
                          scale_regularizer,
                          cpuid,
                          trainable,
                          collections,
                          summary,
                          reuse,
                          scope)
    act = actives.get(act)
    def _condition_on(labels):
        select_scale = core.gather(scale, labels)
        select_offset = core.gather(offset, labels)
        select_scale = core.expand_dims(core.expand_dims(select_scale, 1), 1)
        select_offset = core.expand_dims(core.expand_dims(select_offset, 1), 1)
        return select_scale, select_offset

    def _conditional_instance_norm(x):
        x, labels = helper.split_inputs(x)
        with ops_scope:
            mean, variance = core.moments(x, axes, keepdims=True)
            normalized = (x - mean) / core.sqrt(variance + epsilon)
            select_scale, select_offset = _condition_on(labels)
            if select_scale is not None:
                normalized = select_scale * normalized
            if select_offset is not None:
                normalized = normalized + select_offset
            return  act(normalized)
    return _conditional_instance_norm


# @helpers.typecheck(input_shape=list,
#                    momentum=float,
#                    trainable=bool,
#                    fused=bool,
#                    collections=str,
#                    summary=str,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def batch_norm(input_shape,
               axis=None,
               momentum=0.99,
               offset_initializer='zeros',
               scale_initializer='ones',
               offset_regularizer=None,
               scale_regularizer=None,
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               cpuid=0,
               epsilon=core.epsilon,
               act=None,
               trainable=True,
               collections=None,
               summary='histogram',
               check_input_shape=True,
               reuse=False,
               name=None,
               scope=None):
    """ batch normalization layer
        Attributes
        ==========
        input_shape : list / tuple
                      input tensor shape
        momentum : float | None
                   momentum to update moving mean and variance
                   if None, moving mean and variance
                   will not be updated
        offset_initializer : string / callable function | None | bool
                             initializer to initialize offset
                             if False, offset will be ignored
                             (output will not be centered)
        offset_regularizer : string
                             penalty for offset
        scale_initializer : string / callable function | None | bool
                            initializer to initialize scale
                            if False, scale will be ignored
        scale_regularizer : string
                            penalty for scale
        moving_mean_initializer : string / callable function | None
                                  initializer for moving_mean
        moving_variance_initializer : string / callable function | None
                                      initializer for moving_variance
        is_training : bool
        fused : bool
                use fused_batch_normal if true
    """
    if check_input_shape:
        helper.check_input_shape(input_shape)
    if helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'batch_norm',
                                             reuse)
    if axis is None:
        axis = list(range(len(input_shape)-1))
    if fused:
        axis = [0 ,1, 2]
    # if not isinstance(axis, (list, tuple)):
    #     axis = [axis]
    neurons = input_shape[core.caxis]

    offset = None
    if not isinstance(offset_initializer, bool) or \
       offset_initializer is not False:
        offset = mm.malloc('offset',
                           name,
                           neurons,
                           core.float32,
                           offset_initializer,
                           offset_regularizer,
                           cpuid,
                           trainable,
                           collections,
                           summary,
                           reuse,
                           scope)
    scale = None
    if not isinstance(scale_initializer, bool) or \
       scale_initializer is not False:
        scale = mm.malloc('scale',
                          name,
                          neurons,
                          core.float32,
                          scale_initializer,
                          scale_regularizer,
                          cpuid,
                          trainable,
                          collections,
                          summary,
                          reuse,
                          scope)

    moving_mean = None
    moving_mean = mm.malloc('moving-mean',
                            name,
                            neurons,
                            core.float32,
                            moving_mean_initializer,
                            None,
                            cpuid,
                            False,
                            collections,
                            summary,
                            reuse,
                            scope)

    moving_variance = None
    moving_variance = mm.malloc('moving-variance',
                                name,
                                neurons,
                                core.float32,
                                moving_variance_initializer,
                                None,
                                cpuid,
                                False,
                                collections,
                                summary,
                                reuse,
                                scope)
    act = actives.get(act)
    def _train(x, fused):
        if fused is True:
            # fused_batch_norm(x, scale, offset, mean=None, variance=None,
            #                  epsionl=0.001, is_training=True, name=None)
            # x must be 4-d tensor
            # mean / variance used for inference
            x_shape = core.shape(x)
            for _ in range(4 - x.get_shape().ndims):
                x = core.expand_dims(x, 1)
            x, mean, variance = core.fused_batch_norm(x, scale,
                                                      offset,
                                                      is_training=True,
                                                      epsilon=epsilon)
            x = core.reshape(x, x_shape)
        else:
            mean, variance = core.moments(x, axis, keepdims=True)
            # batch_normalize(x, mean, variance, offset,
            #                 scale, variance_epsilon, name)
            x = core.batch_norm(x, mean, variance,
                                offset, scale, epsilon)
            mean = core.squeeze(mean)
            variance = core.squeeze(variance)
        if momentum is not None:
            update_mean=core.moving_average_update(moving_mean, mean, momentum)
            update_variance=core.moving_average_update(moving_variance, variance, momentum)
            core.add_to_collection(core.Collections.update_ops, update_mean)
            core.add_to_collection(core.Collections.update_ops, update_variance)
        return act(x)

    def _infer(x, fused):
        if fused is True:
            x_shape = [-1] + core.shape(x)[1:]
            for _ in range(4 - x.get_shape().ndims):
                x = core.expand_dims(x, 1)
            x, mean, variance = core.fused_batch_norm(x, scale,
                                 offset, moving_mean, moving_variance,
                                 epsilon, is_training=False)
            x = core.reshape(x, x_shape)
        else:
            mean, variance = core.moments(x, axis, keepdims=True)
            x = core.batch_norm(x, moving_mean, moving_variance,
                                         offset, scale, epsilon)

        return act(x)

    def _batch_norm(x, is_training, fused):
        with ops_scope:
            if is_training:
                return _train(x, fused)
            else:
                return _infer(x, fused)
    return _batch_norm
