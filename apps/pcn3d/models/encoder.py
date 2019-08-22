import sigma
from sigma import ops, layers

from .block import _block

def encode(name, inputs, primary_size=16, reuse=False, is_training=True):
    with sigma.defaults(reuse=reuse):
        #  inputs: [batch-size, N, 3]
        #=>     x: [batch-size, N, 64]
        # typically, N = 2048
        x = _block('{}_64d'.format(name), inputs, 64, act='relu', is_training=is_training)

        #       x: [batch-size, N, 64]
        #=>     x: [batch-size, N, 128]
        x = _block('{}_128d'.format(name), x, 128, act='relu', is_training=is_training)

        #       x: [batch-size, N, 128]
        #=>     x: [batch-size, N, 1024]
        #=>     x: [batch-size, 1, 1024] * primary_size (by max_pool1d)
        N = ops.core.shape(x)[1]
        capsules = [_block('{}_{}'.format(name, idx), x, 1024, act=None, pool=N, is_training=is_training) for idx in range(primary_size)]

        #       x: [batch-size, 1, 1024] * primary_size (by max_pool1d)
        #=>     x: [batch-size, primary_size, 1024]
        x = layers.merge.concat(capsules, axis=1, name='{}_concat'.format(name))
        #ops.core.summarize('{}_concat'.format(name), x)

        #       x: [batch-size, primary_size, 1024]
        #=>     x: [batch-size, 1024, primary_size]
        x = layers.base.transpose(x, (0, 2, 1), name='{}_transpose'.format(name))
        x = layers.actives.squash(x, 1, name='{}_squash'.format(name), epsilon=1e-9)
        #ops.core.summarize('{}_squash'.format(name), x)

        return x
