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
from .. import helpers, colors

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
            check_input_shape=True,
            reuse=False,
            name=None,
            scope=None):
    if check_input_shape:
        helper.check_input_shape(input_shape)
    ops_scope, _, name = helper.assign_scope(name, scope, 'flatten', reuse)
    output_shape = [input_shape[0], np.prod(input_shape[1:])]
    def _flatten(x):
        with ops_scope:
            return core.reshape(x, output_shape)
    return _flatten, output_shape


# @helpers.typecheck(target_shape=list, reuse=bool, name=str, scope=str)
def reshape(target_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, _, name = helper.assign_scope(name, scope, 'reshape', reuse)
    def _reshape(x):
        with ops_scope:
            return core.reshape(x, target_shape)
    return _reshape, target_shape


# @heplers.typecheck(input_shape=list, conjugate=bool, name=str, scope=str)
def transpose(input_shape,
              perm,
              conjugate=False,
              check_input_shape=True,
              reuse=False,
              name=None,
              scope=None):
    #helper.check_input_shape(input_shape)
    ops_scope, _, name = helper.assign_scope(name,
                                             scope,
                                             'transpose',
                                             reuse)
    if not isinstance(perm, (list, tuple)):
        raise TypeError('`perm` must be list/tuple, given `{}`'
                        .format(colors.red(type(perm))))
    if len(input_shape) != len(perm):
        raise AttributeError('`input_shape` and `perm` must have same length. given `{}` vs `{}`'
                             .format(colors.red(len(input_shape)), colors.blue(len(perm))))
    output_shape = []
    for p in perm:
        output_shape.append(input_shape[p])
    def _transpose(x):
        with ops_scope:
            return core.transpose(x,
                                  perm=perm,
                                  conjugate=conjugate)
    return _transpose, output_shape


# @helpers.typecheck(input_shape=list, axis=int, reuse=bool, name=str, scope=str)
def expand_dims(input_shape,
                axis,
                check_input_shape=True,
                reuse=False,
                name=None,
                scope=None):
    if check_input_shape:
        helper.check_input_shape(input_shape)
    ops_scope, _, _ = helper.assign_scope(name,
                                          scope,
                                          'expand_dims',
                                          reuse)
    output_shape = input_shape[:]
    naxis = helper.normalize_axes(input_shape, axis)
    output_shape.insert(axis if axis >= 0 else naxis+1, 1)
    def _expand_dims(x):
        with ops_scope:
            return core.expand_dims(x, axis)
    return _expand_dims, output_shape


# @helpers.typecheck(kpeep=float,
#                    noise_shape=list,
#                    aslayer=bool,
#                    reuse=bool,
#                    name=str,
#                    scope=str)
def dropout(pkeep,
            noise_shape=None,
            seed=None,
            aslayer=False,
            reuse=False,
            name=None,
            scope=None):
    # if aslayer:
    #     ops_scope, _, name = helper.assign_scope(name,
    #                                              scope,
    #                                              'dropout',
    #                                              reuse)
    #     def _dropout(x):
    #         with ops_scope:
    #             return core.dropout(x, pkeep, noise_shape, seed, name)
    # else:
    #     def _dropout(x):
    #         return core.dropout(x, pkeep, noise_shape, seed, name)
    # return _dropout
    def _dropout(x):
        with helper.maybe_layer(aslayer, name, scope, 'dropout', reuse):
            return core.dropout(x, pkeep, noise_shape, seed, name)
    return _dropout
