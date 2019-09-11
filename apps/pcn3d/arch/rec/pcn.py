import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers
from sigma.ops import core, mm
from sigma.apps.pcn3d import dataset
from . import ops
import tensorflow as tf
import numpy as np

def project(inputs, channels, ksize, name, padding='VALID', stride=[1,1], is_training=True, reuse=False):
    x = layers.convs.conv2d(inputs, channels, ksize, stride, padding, reuse=reuse, name='{}_conv2d'.format(name))
    x = layers.norms.batch_norm(x, is_training, momentum=0.9, act='relu', reuse=reuse, name='{}_batchnorm'.format(name))
    return x


def transform_net(inputs, is_training, batch_size, is_feature=False, reuse=False, name='transform', scope=None):
    if is_feature:
        _, npoints, _, k = core.shape(inputs)
        #expand = layers.base.transpose(inputs, (0, 1, 3, 2), name='{}_transpose'.format(name))
        expand = inputs
        kshape = [1, 1]
    else:
        # inputs: [batch-size, npoints, 3]
        _, npoints, k = core.shape(inputs)
        kshape = [1, 3]
        #=> [batch-size, npoints, 3, 1]
        expand = layers.base.expand_dims(inputs, -1, name='{}_expand_dims'.format(name))

    with sigma.defaults(padding='valid', stride=1, reuse=reuse, scope=scope):
        # [batch-size, npoints, 1, 64]
        x = layers.convs.conv2d(expand, 64, kshape=kshape, name='{}_conv2d-0'.format(name))
        x = layers.norms.batch_norm(x, is_training, act='relu', momentum=0.9, name='{}_batch_norm-0'.format(name))

        # [batch-size, npoints, 1, 128]
        x = layers.convs.conv2d(x, 128, kshape=[1, 1], name='{}_conv2d-1'.format(name))
        x = layers.norms.batch_norm(x, is_training, act='relu', momentum=0.9, name='{}_batch_norm-1'.format(name))

        # [batch-size, npoints, 1, 1024]
        x = layers.convs.conv2d(x, 1024, kshape=[1, 1], name='{}_conv2d-2'.format(name))
        x = layers.norms.batch_norm(x, is_training, act='relu', momentum=0.9, name='{}_batch_norm-2'.format(name))

        # [batch-size, 1, 1, 1024]
        x = layers.pools.max_pool2d(x, pshape=[npoints, 1], name='{}_max_pool2d'.format(name))

        # [batch-size, 1024]
        x = layers.base.reshape(x, [batch_size, -1], name='{}_reshape-0'.format(name))

        # [batch-size, 512]
        x = layers.convs.dense(x, 512, name='{}_dense-0'.format(name))
        x = layers.norms.batch_norm(x, is_training, act='relu', momentum=0.9, name='{}_batch_norm-3'.format(name))

        # [batch-size, 256]
        x = layers.convs.dense(x, 256, name='{}_dense-1'.format(name), reuse=reuse)
        x = layers.norms.batch_norm(x, is_training, act='relu', momentum=0.9, name='{}_batch_norm-4'.format(name))

        weights = mm.malloc(name='{}_weights'.format(name),
                            layername='pose_transform',
                            shape=[256, k*k],
                            dtype=core.float32,
                            reuse=reuse,
                            initializer='zeros')
        bias = mm.malloc(name='{}_bias'.format(name),
                         layername='pose_transform',
                         shape=[k*k,],
                         reuse=reuse,
                         dtype=core.float32,
                         initializer='zeros')
        bias += core.constant(np.eye(k).flatten(), dtype=core.float32, name='{}_eyes'.format(name))
        transform = core.matmul(x, weights, name='{}_matmul'.format(name))
        transform = core.bias_add(transform, bias, name='{}_bias_add'.format(name))

        return layers.base.reshape(transform, [batch_size, k, k], name='{}_reshape-1'.format(name))


def build_net(inputs, labels, loss='cce', batch_size=32, nclass=16, reuse=False, is_training=True, **kwargs):
    _, npoints, dims = core.shape(inputs)
    with core.device('/gpu:0'):
        if is_training:
            inputs = dataset.rotate_point_cloud(inputs, batch_size=batch_size)
            inputs = dataset.jitter_point_cloud(inputs, batch_size=batch_size)

        # [batch-size, npoints, 3, 1]
        inputs = layers.base.expand_dims(inputs, -1, name='expand_dims-0', reuse=reuse)
        pcts = []
        for i in range(3):
            # [batch-size, npoints, 1, 64]
            pct = project(inputs, 64, [1, 3], name='project_0-{}'.format(i), is_training=is_training, reuse=reuse)
            # [batch-size, npoints, 1, 64]
            pct = project(pct, 64, [1, 1], name='project_1-{}'.format(i), is_training=is_training, reuse=reuse)
            # [batch-size, npoints, 1, 128]
            pct = project(pct, 128, [1, 1], name='project_2-{}'.format(i), is_training=is_training, reuse=reuse)
            # [batch-size, npoints, 1, 1024]
            pct = project(pct, 1024, [1, 1], name='project_3-{}'.format(i), is_training=is_training, reuse=reuse)
            # [batch-size, 1, 1, 1024]
            pct = layers.pools.max_pool2d(pct, [npoints, 1], name='max_pool2d-{}'.format(i), reuse=reuse)
            # [batch-size, 1024]
            pct = layers.base.reshape(pct, [batch_size, 1, 1024], name='reshape-{}'.format(i), reuse=reuse)

            pcts.append(pct)

        pct = layers.merge.concat(pcts, axis=1, name='concat', reuse=reuse)

        #  [batch-size, 6, 512]
        #=>[batch-size, 16, 40]
        pct = layers.capsules.dense(pct, 128, 8, iterations=3, name='cap_dense-0', reuse=reuse)
        pct = layers.norms.batch_norm(pct, is_training, momentum=0.9, name='batch_norm', reuse=reuse)

        pct = layers.capsules.dense(pct, 40, 16, iterations=3, name='cap_dense-1', reuse=reuse)
        pct = layers.capsules.norm(pct, axis=1, name='cap_norm', reuse=reuse)

        loss_op = layers.losses.get(loss, pct, labels, name='recognition_losses', reuse=reuse)

        metric_op = layers.metrics.accuracy([pct, labels])
    return pct, loss_op, metric_op
