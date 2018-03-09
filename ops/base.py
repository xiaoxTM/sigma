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