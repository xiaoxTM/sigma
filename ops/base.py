from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

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
    ops_scope, name = helper.assign_scope(name,
                                          scope,
                                          'embedding',
                                          reuse)
    embeddings = mm.malloc('embedding', table_shape, dtype,
                           initializer, regularizer, trainable,
                           collections, reuse, name, scope)
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
    ops_scope, name = helper.assign_scope(name, scope, 'flatten', reuse)
    output_shape = [-1, np.prod(input_shape[1:])]
    def _flatten(x):
        with ops_scope:
            return core.reshape(x, output_shape, name)
    return _flatten, output_shape


def reshape(output_shape,
            reuse=False,
            name=None,
            scope=None):
    ops_scope, name = helper.assign_scope(name, scope, 'reshape', reuse)
    def _reshape(x):
        with ops_scope:
            return core.reshape(x, output_shape, name)
    return _reshape, output_shape
