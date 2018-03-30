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

def embedding(table_shape,
              strategy='mod',
              dtype=core.float32,
              initializer='glorot_uniform',
              regularizer=None,
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
                           initializer, regularizer, trainable,
                           collections, summary, reuse, scope)
    def _embedding(ids):
        with ops_scope:
            if core.dtype(ids) != core.int32:
                ids = core.cast(ids, core.int32)
            return core.gather(embeddings, ids)
    return _embedding


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


def reshape(output_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, _, name = helper.assign_scope(name, scope, 'reshape', reuse)
    def _reshape(x):
        with ops_scope:
            return core.reshape(x, output_shape)
    return _reshape, output_shape


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


def maskout(input_shape,
            indices=None,
            axis=-2,
            drop=False,
            flatten=True,
            reuse=False,
            name=None,
            scope=None):
    """ typical input_shape form:
        [batch-size, nclass, depth]
    """
    helper.check_input_shape(input_shape)
    ops_scope, name_with_ltype, _ = helper.assign_scope(name,
                                                        scope,
                                                        'maskout',
                                                        reuse)
    output_shape = input_shape[:]
    index_shape = output_shape[:]
    ones = [1] * len(input_shape)
    axis = helper.normalize_axes(input_shape, axis)
    if drop:
        prepare_scope = '{}/preprocess'.format(name_with_ltype)
        if scope is not None:
            prepare_scope = '{}/{}'.format(scope, prepare_scope)
        if indices is None:
            index_shape[axis] = 1
            output_shape.pop(axis)
            nelems = 1
        else:
            nelems = len(indices)
            index_shape[axis] = nelems
            output_shape[axis] = nelems
            if flatten:
                output_shape[-1] *= output_shape[axis]
                output_shape.pop(axis)
        with core.name_scope(prepare_scope):
            def _index(i, index=None):
                if index is None:
                    index = core.range(input_shape[i])
                shape = ones[:]
                shape[i] = input_shape[i]
                index = core.reshape(index, shape)
                multiples = [int(x / y) for x, y in zip(index_shape, shape)]
                index = core.reshape(core.tile(index, multiples), (-1,1))
                return index
            indexlist = [_index(i) for i in range(len(input_shape)) \
                         if i != axis]

        def _maskout(x, elements=None):
            with ops_scope:
                if elements is None:
                    if axis != len(input_shape) - 1:
                        xnorm = core.norm(x, -1, safe=False)
                        elements = core.argmax(xnorm, -1, dtype=core.int32)
                    elements = _index(0, elements)
                positions = indexlist[:]
                positions.insert(axis, elements)
                positions = core.concat(positions, axis=1)
                x = core.gather_nd(x, positions)
                reshape_target = [-1] + output_shape[1:]
                return core.reshape(x, reshape_target)
    else:
        tiles = ones
        tiles[core.axis] = input_shape[core.axis]
        if flatten:
            output_shape[-1] *= output_shape[axis]
            output_shape.pop(axis)
        def _maskout(x, elements=None):
            with ops_scope:
                if elements is None:
                    if axis != len(input_shape) - 1:
                        # x shape: [batch-size, nclass, depth]
                        # xnorm shape: [batch-size, nclass]
                        xnorm = core.norm(x, -1, safe=False)
                        # elements shape: [batch-size]
                        elements = core.argmax(xnorm, -1, dtype=core.int32)
                # onehot to from
                # [batch-size]
                # to
                # [batch-size, nclass]
                elements = core.one_hot(elements, input_shape[-2])
                # tile to [batch-size, nclass, depth]
                elements = core.tile(core.expand_dims(elements, -1), tiles)
                x = x * elements
                if flatten:
                    x = core.flatten(x)
                return x
    return _maskout, output_shape
