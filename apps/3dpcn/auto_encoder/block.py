import sigma
from sigma import layers, ops

def _block(name, inputs, neurons, channels=1, act=None, pool=None, reuse=False):
    with sigma.defaults(reuse=reuse):
        x = layers.convs.conv1d(inputs, neurons, channels, name='{}_conv1d'.format(name))
        ops.core.summarize('{}_conv2d'.format(name), x)
        x = layers.norms.batch_norm(x, act=act, name='{}_batchnorm'.format(name))
        ops.core.summarize('{}_batchnorm'.format(name), x)
        if pool is not None:
            x = layers.pools.max_pool1d(x, pool, name='{}_maxpool1d'.format(name))
            ops.core.summarize('{}_maxpool1d'.format(name), x)
        return x
