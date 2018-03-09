from .. import ops
from . import core

@core.layer
def total_variation_regularize(inputs,
                               reuse=False,
                               name=None,
                               scope=None):
    shape = ops.core.shape(inputs)
    return ops.regularizers.total_variation_regularizer(shape,
                                                        reuse,
                                                        name,
                                                        scope)(inputs)

tvr = total_variation_regularize
