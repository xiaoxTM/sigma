import sys
sys.path.append('/home/xiaox/studio/src/git-series')

import sigma
from sigma import layers
from sigma.ops import core, initializers, mm, helper
from sigma.apps.pcn3d import dataset
from . import ops

def vote(inputs, depth=None):
    if not isinstance(inputs, (list, tuple)):
        raise TypeError('`inputs` must be list/tuple. given {}'.format(type(inputs)))
    input_shapes = [core.shape(s) for s in inputs]
    helper.check_shape_consistency(input_shapes)
    if depth is None:
        depth = core.shape(inputs[0])[-1]
    xs = [core.one_hot(core.argmax(x, axis=-1), depth=depth) for x in inputs]
    return core.add(xs)

def pose_transform(inputs, is_training=True, reuse=False, name=None):
    ''' inputs: [batch-size, npoints, channels]
        return: [batch-size, channels, channels]
    '''
    batch_size, npoints, channels = core.shape(inputs)

    with sigma.defaults(reuse=reuse, is_training=is_training, padding='valid'):
        #   [batch-size, npoints, channels, 1]
        x = layers.base.expand_dims(inputs, axis=-1, name='{}_transform_expand_dims'.format(name))

        #   [batch-size, npoints, channels, 1]
        #=> [batch-size, npoints, 1, 64]
        x = layers.convs.conv2d(x, 64, [1, channels], name='{}_transform_conv2d_0'.format(name))
        x = layers.norms.batch_norm(x, momentum=0.9, act='relu', name='{}_transform_batchnorm_0'.format(name))

        #   [batch-size, npoints, 1, 128]
        x = layers.convs.conv2d(x, 128, [1, 1], name='{}_transform_conv2d_1'.format(name))
        x = layers.norms.batch_norm(x, momentum=0.9, act='relu', name='{}_transform_batchnorm_1'.format(name))

        #   [batch-size, npoints, 1, 1024]
        x = layers.convs.conv2d(x, 1024, [1, 1], name='{}_transform_conv2d_2'.format(name))
        x = layers.norms.batch_norm(x, momentum=0.9, act='relu', name='{}_transform_batchnorm_2'.format(name))

        #   [batch-size, 1, 1, 1024]
        x = layers.pools.max_pool2d(x, [npoints, 1], name='{}_transform_maxpool2d'.format(name))

        #   [batch-size, 1024]
        x = layers.base.reshape(x, [batch_size, 1024], name='{}_transform_reshape'.format(name))

        #   [batch-size, 512]
        x = layers.convs.dense(x, 512, name='{}_transform_dense_0'.format(name))
        x = layers.norms.batch_norm(x, momentum=0.9, act='relu', name='{}_transform_batchnorm_3'.format(name))

        #   [batch-size, 256]
        x = layers.convs.dense(x, 256, name='{}_transform_dense_1'.format(name))
        x = layers.norms.batch_norm(x, momentum=0.9, act='relu', name='{}_transform_batchnorm_4'.format(name))

        weights = mm.malloc('weights',
                            'pose_transform',
                            [256, channels*channels],
                            dtype=core.float32,
                            initializer='zeros',
                            scope=name,
                            reuse=reuse)
        bias = mm.malloc('bias',
                         'pose_transform',
                         [channels*channels],
                         dtype=core.float32,
                         initializer=initializers.constant([1,0,0,0,1,0,0,0,1]),
                         scope=name,
                         reuse=reuse)
        transform = core.bias_add(core.matmul(x, weights), bias)
        x = core.reshape(transform, [batch_size, channels, channels])
        return core.matmul(inputs, x, name='{}_matmul'.format(name))


def mean_accuracy(xs, labels, reuse=False):
    pred = vote(xs)
    return pred, layers.metrics.accuracy([pred, labels], reuse=reuse)


def build_net(inputs, labels, loss='margin_loss', batch_size=32, nclass=40, is_training=True, reuse=False, **kwargs):
    with core.device('/gpu:3'):
        if is_training:
            inputs = dataset.rotate_point_cloud(inputs, batch_size=batch_size)
            inputs = dataset.jitter_point_cloud(inputs, batch_size=batch_size)
            core.summarize('inputs', inputs, reuse=reuse)

    xs = [None] * 4

    with core.device('/gpu:0'):
        #   [batch-size, npoints, 3]
        #=> [batch-size, 16, 512]
        x = pose_transform(inputs, is_training, reuse, name='br0')
        x = layers.base.transpose(x, (0, 2, 1), name='transpose_0')
        x = ops.permutation_transform(x, 512, 16, mode='max', reuse=reuse, name='permutation_transform_0', act='squash')
        core.summarize('permutation_transform_0', x, reuse=reuse)

        x = layers.capsules.dense(x, nclass, 16, iterations=3, name='caps_dense_0', reuse=reuse)
        core.summarize('caps_dense_0', x, reuse=reuse)
        x = layers.capsules.norm(x, axis=1, name='caps_norm_0', reuse=reuse)
        core.summarize('caps_norm_0', x, reuse=reuse)

        l = layers.losses.get(loss, x, labels, reuse=reuse, name='{}_0'.format(loss))
        core.summarize('loss_0', l, 'scalar')
        xs[0] = x
        losses = l

    with core.device('/gpu:1'):
        #   [batch-size, 3, npoints]
        #=> [batch-size, 16, 512]
        x = pose_transform(inputs, is_training, reuse, name='br1')
        x = layers.base.transpose(x, (0, 2, 1), name='transpose_1')
        x = ops.permutation_transform(x, 512, 16, mode='max', reuse=reuse, name='permutation_transform_1', act='squash')
        core.summarize('permutation_transform_1', x)

        x = layers.capsules.dense(x, nclass, 16, iterations=3, name='caps_dense_1', reuse=reuse)
        core.summarize('caps_dense_1', x, reuse=reuse)
        x = layers.capsules.norm(x, axis=1, name='caps_norm_1', reuse=reuse)
        core.summarize('caps_norm_1', x, reuse=reuse)

        l = layers.losses.get(loss, x, labels, reuse=reuse, name='{}_1'.format(loss))
        core.summarize('loss_1', l, 'scalar')
        xs[1] = x
        losses += l

    with core.device('/gpu:2'):
        #   [batch-size, 3, npoints]
        #=> [batch-size, 16, 512]
        x = pose_transform(inputs, is_training, reuse, name='br2')
        x = layers.base.transpose(x, (0, 2, 1), name='transpose_2')
        x = ops.permutation_transform(x, 512, 16, mode='max', reuse=reuse, name='permutation_transform_2', act='squash')
        core.summarize('permutation_transform_2', x)

        x = layers.capsules.dense(x, nclass, 16, iterations=3, name='caps_dense_2', reuse=reuse)
        core.summarize('caps_dense_2', x, reuse=reuse)
        x = layers.capsules.norm(x, axis=1, name='caps_norm_2', reuse=reuse)
        core.summarize('caps_norm_2', x, reuse=reuse)

        l = layers.losses.get(loss, x, labels, reuse=reuse, name='{}_2'.format(loss))
        core.summarize('loss_2', l, 'scalar')
        xs[2] = x
        losses += l

    with core.device('/gpu:3'):
        #   [batch-size, 3, npoints]
        #=> [batch-size, 16, 512]
        x = pose_transform(inputs, is_training, reuse, name='br3')
        x = layers.base.transpose(x, (0, 2, 1), name='transpose_3')
        x = ops.permutation_transform(x, 512, 16, mode='max', reuse=reuse, name='permutation_transform_3', act='squash')
        core.summarize('permutation_transform_3', x)

        x = layers.capsules.dense(x, nclass, 16, iterations=3, name='caps_dense_3', reuse=reuse)
        core.summarize('caps_dense_3', x, reuse=reuse)
        x = layers.capsules.norm(x, axis=1, name='caps_norm_3', reuse=reuse)
        core.summarize('caps_norm_3', x, reuse=reuse)

        l = layers.losses.get(loss, x, labels, reuse=reuse, name='{}_3'.format(loss))
        core.summarize('loss_3', l, 'scalar')
        xs[3] = x
        losses += l

    # xs = [[batch-size, nclass],
    #       [batch-size, nclass],
    #       [batch-size, nclass],
    #       [batch-size, nclass]]
    #
    #=>    [batch-size, 16, 2048]
    pred, acc = mean_accuracy(xs, labels)

    return pred, losses, acc
