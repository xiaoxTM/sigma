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

import numpy as np
from . import helper, core, mm
from .. import helpers

import tensorflow as tf

def embedding(table_shape,
              strategy='mod',
              dtype=core.float32,
              initializer='glorot_uniform',
              regularizer=None,
              cpuid=0,
              trainable=True,
              collections=None,
              summary='histogram',
              reuse=False,
              name=None,
              scope=None):
    """ table shape should be:
        table-length x embedding-length
        normally, table-length is the number of class
                  emebdding-length

        // TODO: test
    """
    # assign scope returns
    # - ops_scope       : [scope/]layername/layertype
    # - name_with_ltype : layername/layertype
    # - name            : original name | layername
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'embedding',
                                             reuse)
    embeddings = mm.malloc('embedding', name, table_shape, dtype,
                           initializer, regularizer, cpuid, trainable,
                           collections, summary, reuse, scope)
    def _embedding(ids):
        with ops_scope:
            if core.dtype(ids) != core.int32:
                ids = core.cast(ids, core.int32)
            return core.gather(embeddings, ids)
    return _embedding


# @helpers.typecheck(input_shape=list, reuse=bool, name=str, scope=str)
def flatten(input_shape,
            reuse=False,
            name=None,
            scope=None):
    helper.check_input_shape(input_shape)
    ops_scope, _, name = helper.assign_scope(name, scope, 'flatten', reuse)
    output_shape = [-1, np.prod(input_shape[1:])]
    def _flatten(x):
        with ops_scope:
            return core.reshape(x, output_shape)
    return _flatten, output_shape


# @helpers.typecheck(output_shape=list, reuse=bool, name=str, scope=str)
def reshape(output_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, _, name = helper.assign_scope(name, scope, 'reshape', reuse)
    def _reshape(x):
        with ops_scope:
            return core.reshape(x, output_shape)
    return _reshape, output_shape


# @helpers.typecheck(input_shape=list, axis=int, reuse=bool, name=str, scope=str)
def expand_dims(input_shape,
                axis,
                reuse=False,
                name=None,
                scope=None):
    helper.check_input_shape(input_shape)
    ops_scope, _, _ = helper.assign_scope(name,
                                          scope,
                                          'expand_dims',
                                          reuse)
    output_shape = input_shape[:]
    output_shape.insert(axis if axis >= 0 else axis + 1, 1)
    def _expand_dims(x):
        with ops_scope:
            return core.expand_dims(x, axis)
    return _expand_dims, output_shape


# @helpers.typecheck(input_shape=list,
#                    axis=int,
#                    drop=bool,
#                    flatten=bool,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def maskout(input_shape,
            axis=-2,
            drop=False,
            flatten=True,
            reuse=False,
            name=None,
            scope=None):
    """ typical input_shape form:
        [batch-size, nclass, depth]

        flatten works ONLY when drop is `False`
    """
    helper.check_input_shape(input_shape)
    ops_scope, name_with_ltype, _ = helper.assign_scope(name,
                                                        scope,
                                                        'maskout',
                                                        reuse)
    output_shape = input_shape[:]
    index_shape = output_shape[:]
    axis = helper.normalize_axes(input_shape, axis)
    if drop:
        #prepare_scope = '{}/preprocess'.format(name_with_ltype)
        #if scope is not None:
        #    prepare_scope = '{}/{}'.format(scope, prepare_scope)
        index_shape[axis] = 1
        output_shape.pop(axis)
#        with core.name_scope(prepare_scope):
#            def _index(i, index=None):
#                if index is None:
#                    index = core.range(input_shape[i])
#                shape = ones[:]
#                shape[i] = input_shape[i]
#                index = core.reshape(index, shape)
#                multiples = [int(x / y) for x, y in zip(index_shape, shape)]
#                index = core.reshape(core.tile(index, multiples), (-1,1))
#                return index
#            indexlist = [_index(i) for i in range(len(input_shape)) \
#                         if i != axis]
#
        def _maskout(x, index=None):
            with ops_scope:
                if index is None:
                    if axis != len(input_shape) - 1:
                        xnorm = core.norm(x, -1, safe=False)
                        index = core.argmax(xnorm, -1, dtype=core.int32)
                    else:
                        raise ValueError('index cannot be None')
                x = core.gather(x, index, axis=axis)
                return x
    else:
        # no need for batch-size axis, therefore -1
        tiles = [1] * (len(input_shape))
        tiles[core.axis] = input_shape[core.axis]
        if flatten:
            output_shape[-1] *= output_shape[axis]
            output_shape.pop(axis)
        def _maskout(x, index=None):
            with ops_scope:
                if index is None:
                    #print('axis: ', axis)
                    #print('input shape length: ', len(input_shape))
                    ## if index not given, use the max `NORM` as index
                    if axis != len(input_shape) - 1:
                        # x shape: [batch-size, nclass, depth]
                        # xnorm shape: [batch-size, nclass]
                        xnorm = core.norm(x, -1, safe=False)
                        # index shape: [batch-size]
                        index = core.argmax(xnorm, -1, dtype=core.int32)
                        # [batch-size] => [batch-size, nclass]
                        index = core.one_hot(index, input_shape[-2])
                    else:
                        raise ValueError('element cannot be None')
                # onehot to from
                # [batch-size]
                # to
                # [batch-size, nclass]
                # tile to [batch-size, nclass, depth]
                index = core.tile(core.expand_dims(index, -1), tiles)
                index = core.cast(index, core.float32)
                x = core.multiply(x, index)
                if flatten:
                    #x = core.flatten(x)
                    x = core.reshape(x, output_shape)
                return x
    return _maskout, output_shape
