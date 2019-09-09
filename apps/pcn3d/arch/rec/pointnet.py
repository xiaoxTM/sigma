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
    x = layers.norms.batch_norm(x, is_training, momentum=0.9, reuse=reuse, name='{}_batchnorm'.format(name))
    x = layers.actives.relu(x, name='{}_act'.format(name), reuse=reuse)
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
        x = layers.norms.batch_norm(x, is_training, momentum=0.9, name='{}_batch_norm-0'.format(name))
        x = layers.actives.relu(x, name='{}_act-0'.format(name))

        # [batch-size, npoints, 1, 128]
        x = layers.convs.conv2d(x, 128, kshape=[1, 1], name='{}_conv2d-1'.format(name))
        x = layers.norms.batch_norm(x, is_training, momentum=0.9, name='{}_batch_norm-1'.format(name))
        x = layers.actives.relu(x, name='{}_act-1'.format(name))

        # [batch-size, npoints, 1, 1024]
        x = layers.convs.conv2d(x, 1024, kshape=[1, 1], name='{}_conv2d-2'.format(name))
        x = layers.norms.batch_norm(x, is_training, momentum=0.9, name='{}_batch_norm-2'.format(name))
        x = layers.actives.relu(x, name='{}_act-2'.format(name))

        # [batch-size, 1, 1, 1024]
        x = layers.pools.max_pool2d(x, pshape=[npoints, 1], name='{}_max_pool2d'.format(name))

        # [batch-size, 1024]
        x = layers.base.reshape(x, [batch_size, -1], name='{}_reshape-0'.format(name))

        # [batch-size, 512]
        x = layers.convs.dense(x, 512, name='{}_dense-0'.format(name))
        x = layers.norms.batch_norm(x, is_training, momentum=0.9, name='{}_batch_norm-3'.format(name))
        x = layers.actives.relu(x, name='{}_act-3'.format(name))

        # [batch-size, 256]
        x = layers.convs.dense(x, 256, name='{}_dense-1'.format(name), reuse=reuse)
        x = layers.norms.batch_norm(x, is_training, momentum=0.9, name='{}_batch_norm-4'.format(name))
        x = layers.actives.relu(x, name='{}_act-4'.format(name))

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
        # [batch-size, 3, 3]
        transform_input = transform_net(inputs, is_training, batch_size, name='transform_input', scope='transforms', reuse=reuse)
        # [batch-size, npoints, 3]
        pct = core.matmul(inputs, transform_input, name='matmul-0')

        # [batch-size, npoints, 3, 1]
        pct = layers.base.expand_dims(pct, -1, name='expand_dims-0', reuse=reuse)

        # [batch-size, npoints, 1, 64]
        pct = project(pct, 64, [1, 3], name='project_0', is_training=is_training, reuse=reuse)
        # [batch-size, npoints, 1, 64]
        pct = project(pct, 64, [1, 1], name='project_1', is_training=is_training, reuse=reuse)

        # [batch-size, 64, 64]
        transform_feature = transform_net(pct, is_training, batch_size, is_feature=True, name='transform_feature', scope='transforms', reuse=reuse)
        # [batch-size, npoints, 64]
        pct = core.squeeze(pct, axis=2, name='squeeze')
        # [batch-size, npoints, 64]
        pct = core.matmul(pct, transform_feature, name='matmul-1')
        # [batch-size, npoints, 1, 64]
        pct = layers.base.expand_dims(pct, 2, name='expand_dims-1', reuse=reuse)

        # [batch-size, npoints, 1, 64]
        pct = project(pct, 64, [1, 1], name='project-2', is_training=is_training, reuse=reuse)
        # [batch-size, npoints, 1, 128]
        pct = project(pct, 128, [1, 1], name='project-3', is_training=is_training, reuse=reuse)
        # [batch-size, npoints, 1, 1024]
        pct = project(pct, 1024, [1, 1], name='project-4', is_training=is_training, reuse=reuse)

        # [batch-size, 1, 1, 1024]
        pct = layers.pools.max_pool2d(pct, [npoints, 1], name='max_pool2d', reuse=reuse)
        # [batch-size, 1024]
        pct = layers.base.reshape(pct, [batch_size, -1], name='reshape', reuse=reuse)

        # [batch-size, 512]
        pct = layers.convs.dense(pct, 512, name='dense-0', reuse=reuse)
        pct = layers.norms.batch_norm(pct, is_training, momentum=0.9, reuse=reuse, name='batch_norm-0')
        pct = layers.actives.relu(pct, name='act-0')

        # [batch-size, 256]
        pct = layers.convs.dense(pct, 256, name='dense-1', reuse=reuse)
        pct = layers.norms.batch_norm(pct, is_training, momentum=0.9, reuse=reuse, name='batch_norm-1')
        pct = layers.actives.relu(pct, name='act-1')

        # [batch-size, 40]
        pct = layers.convs.dense(pct, 40, name='dense-2', reuse=reuse)
        #pct = layers.norms.batch_norm(pct, is_training, momentum=0.9, reuse=reuse, name='batch_norm-2')
        #pct = layers.actives.tanh(pct, name='act-2')

        k = core.shape(transform_feature)[2]
        matrix_diff = tf.matmul(transform_feature, tf.transpose(transform_feature, perm=(0, 2, 1)))
        matrix_diff -= tf.constant(np.eye(k), dtype=tf.float32)
        matloss = tf.nn.l2_loss(matrix_diff)

        loss_op = layers.losses.get(loss, pct, labels, name='recognition_losses', reuse=reuse)
        loss_op += (matloss * 0.001)

        metric_op = layers.metrics.accuracy([pct, labels])
    return loss_op, metric_op
