import sigma
from sigma import layers, ops

def _block(name, inputs, neurons, channels=1, act=None, pool=None, reuse=False, is_training=True):
    with sigma.defaults(reuse=reuse):
        x = layers.convs.conv1d(inputs, neurons, channels, name='{}_conv1d'.format(name))
        ops.core.summarize('{}_conv2d'.format(name), x)
        x = layers.norms.batch_norm(x, is_training, act=act, name='{}_batchnorm'.format(name), reuse=reuse)
        ops.core.summarize('{}_batchnorm'.format(name), x)
        if pool is not None:
            x = layers.pools.max_pool1d(x, pool, name='{}_maxpool1d'.format(name))
            ops.core.summarize('{}_maxpool1d'.format(name), x)
        return x

def _block_conv2d(inputs, channels, kshape=3, stride=1, padding='valid', act=None, reuse=False, is_training=False, name=None, reshape=True):
    # inputs: [batch-size, npoints, 3, channels]
    npoints = ops.core.shape(inputs)[1]
    with sigma.defaults(reuse=reuse):
        #    [batch-size, npoints, 3, channels]
        #=>  [batch-size, npoints-2, 1, 3*channels]
        x = layers.convs.conv2d(inputs, 3*channels, kshape=kshape, stride=stride, padding=padding, act=act, name='{}_conv2d'.format(name))
        x = layers.norms.batch_norm(x, is_training, act=act, name='{}_batchnorm'.format(name))
        if reshape:
            x = layers.base.reshape(x, [-1, npoints-kshape+1, 3, channels], name='{}_reshape'.format(name))
        else:
            # [batch-size, npoints, 1, 3*channels]
            x = layers.base.reshape(x, [-1, npoints-kshape+1, 3*channels], name='{}_reshape'.format(name))
    return x
