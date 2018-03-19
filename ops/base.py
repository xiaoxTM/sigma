import numpy as np
from . import helper, core, mm

def embedding(table_shape,
              strategy='mod',
              dtype=core.float32,
              initializer='glorot_uniform',
              regularizer=None,
              trainable=True,
              collections=None,
              summarize=True,
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
                           collections, reuse, scope)
    if summarize and not reuse:
        tf.summary.histogram(embeddings.name, table)
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
    ops_scope, _, name = helper.assign_scope(name, scope, 'flatten', reuse)
    output_shape = [-1, np.prod(input_shape[1:])]
    def _flatten(x):
        with ops_scope:
            return core.reshape(x, output_shape, name)
    return _flatten, output_shape


def reshape(output_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, _, name = helper.assign_scope(name, scope, 'reshape', reuse)
    def _reshape(x):
        with ops_scope:
            return core.reshape(x, output_shape, name)
    return _reshape, output_shape


def expand_dims(input_shape,
                axis,
                reuse=False,
                name=None,
                scope=None):
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
            indices,
            axis=-1,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, _, _ = helper.assign_scope(name,
                                          scope,
                                          'maskout',
                                          reuse)
    axis = helper.normalize_axes(input_shape, axis)
    output_shape = input_shape[:]
    index_shape = input_shape[:]
    if indices is None:
        index_shape[axis] = 1
        output_shape.pop(axis)
        nelems = 1
    else:
        nelems = len(indices)
        output_shape[axis] = nelems
        index_shape[axis] = nelems

    ones = [1] * len(input_shape)
    def _index(i, index=None):
        if index is None:
            index = core.range(input_shape[i])
        shape = ones[:]
        shape[i] = input_shape[i]
        index = core.reshape(index, shape)
        multiples = [int(x / y) for x, y in zip(index_shape, shape)]
        index = core.reshape(core.tile(index, multiples), (-1,1))
        return index
        
    indexlist = [_index(i) for i in range(len(input_shape)) if i != axis]
    def _maskout(x, elements=None):
        with ops_scope:
            if elements is None:
                if axis != len(input_shape) - 1:
                    xnorm = core.norm(x, -1)
                elements = core.argmax(xnorm, -1, dtype=core.int32)
                elements = _index(0, elements)
            positions = indexlist[:]
            positions.insert(axis, elements)
            positions = core.concat(positions, axis=1)
            x = core.gather_nd(x, positions, name=name)
            return core.reshape(x, output_shape)
    return _maskout, output_shape
